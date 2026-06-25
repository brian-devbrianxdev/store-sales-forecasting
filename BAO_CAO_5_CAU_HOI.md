# Báo cáo trả lời 5 câu hỏi — Store Sales Forecasting

## Bối cảnh bài toán

Cuộc thi Kaggle *Store Sales – Time Series Forecasting*: dự báo doanh số 16 ngày (2017-08-16 → 08-31) cho từng cặp (cửa hàng, ngành hàng) — 1.782 chuỗi thời gian. Độ đo là RMSLE. Giải pháp = 5 chân mô hình bổ trợ nhau, rồi trộn theo phương sai tối thiểu (minimum-variance blend) trong không gian `log1p`. Kết quả cuối: LB **0.37379**.

5 chân: GBT family (darts LightGBM/XGBoost/CatBoost), Chronos-2 (foundation model), LightGBM-v8, TSMixer, TiDE.

---

## Câu 1 — Cấu trúc giải thuật / Lan truyền xuôi (forward)

**Mức tổng thể (pipeline forward):** [cli.py](src/store_sales/cli.py) → từng leg → [ensemble/build.py](src/store_sales/ensemble/build.py)

```
Raw CSV → Feature engineering → 5 leg dự báo độc lập → Min-variance blend → submission.csv
```

**Mức mạng nơ-ron (TSMixer / TiDE)** — đây mới là "lan truyền xuôi" đúng nghĩa, code ở [neural_ts.py](src/store_sales/models/neural_ts.py):

- Đầu vào một cửa sổ quá khứ `input_chunk_length = 90` ngày (TSMixer) cùng covariates → mạng → xuất thẳng H = 16 ngày tương lai (dự báo trực tiếp đa-bước, `output_chunk_length=16`).
- TSMixer forward: chồng 8 khối (`num_blocks=8`), mỗi khối gồm time-mixing MLP (trộn theo chiều thời gian) + feature-mixing MLP (trộn theo chiều đặc trưng), kèm chuẩn hóa + dropout(0.2) + residual; lớp tuyến tính cuối chiếu ra 16 ngày. [neural_ts.py:182-191](src/store_sales/models/neural_ts.py#L182-L191)
- TiDE forward: encoder MLP → bottleneck → decoder MLP, dùng static covariates. [neural_ts.py:173-181](src/store_sales/models/neural_ts.py#L173-L181)

Các chân cây (GBT): "forward" = đi qua chuỗi cây quyết định, mỗi cây cộng dồn dự báo (xem Câu 5). Chronos-2: zero-shot, forward = một lượt qua Transformer, không huấn luyện.

---

## Câu 2 — Lan truyền ngược (backward)

Chỉ các chân nơ-ron (TSMixer/TiDE/NHiTS) dùng backpropagation đúng nghĩa, qua PyTorch (darts):

- Hàm mất mát: MSE trên mục tiêu `log1p(sales)`.
- Backprop + tối ưu: optimizer Adam, learning rate 0.001, lịch học CosineAnnealingLR theo số epoch. [neural_ts.py:188-190](src/store_sales/models/neural_ts.py#L188-L190); huấn luyện 30 epoch, batch 1024. Gradient được lan ngược qua các khối mixing để cập nhật trọng số.

**Phân biệt rõ cho thầy (điểm dễ bị hỏi vặn):**

- GBT (LightGBM/XGBoost/CatBoost) **không** dùng backprop — mà là gradient boosting: mỗi cây mới fit theo gradient của hàm mất mát (ở đây objective tweedie, power 1.2–1.3) so với dự báo hiện tại, rồi cộng dồn. Đây là "đạo hàm" nhưng theo kiểu boosting, không phải lan truyền ngược qua mạng.
- Chronos-2: zero-shot, không có pha huấn luyện/backward nào cả.
- Tầng blend (ensemble): không học bằng gradient — dùng nghiệm đóng (closed-form) `w = Σ⁻¹·1 / (1ᵀ·Σ⁻¹·1)`. [blend.py:57-75](src/store_sales/ensemble/blend.py#L57-L75)

---

## Câu 3 — Input là gì

**Dữ liệu thô** ([data_loading.py](src/store_sales/io/data_loading.py), thư mục [data/](data/)):

- `train.csv` / `test.csv`: id, date, store_nbr, family, sales, onpromotion
- `oil.csv`: giá dầu WTI hằng ngày (Ecuador là nền kinh tế phụ thuộc dầu)
- `holidays_events.csv`: lịch lễ/sự kiện (national/regional/local, transferred…)
- `stores.csv`: metadata cửa hàng (city, state, type, cluster)
- `transactions.csv`: số giao dịch

**Đặc trưng đưa vào mô hình (đã feature-engineering):**

- Tự hồi quy (cho GBT): lag doanh số (1→21, 28…364), rolling mean/std/max (cửa sổ 7…168), EWM (halflife 7/28), theo từng horizon. [lgbm_features.py](src/store_sales/features/lgbm_features.py)
- Lịch: day-of-week, month, day, payday (ngày 15 & cuối tháng), cờ ngày lễ quốc gia, Fourier day-of-year (sin/cos bậc 1–3). [neural_ts.py:86-96](src/store_sales/models/neural_ts.py#L86-L96)
- Dầu (động lực học): lợi suất 7/28 ngày, lag 16/28/56, volatility 28 ngày. [common_features.py:39-72](src/store_sales/features/common_features.py#L39-L72)
- Ngày lễ (khoảng cách): số ngày tới/kể từ ngày lễ đặc biệt gần nhất, cờ transferred. [common_features.py:115-169](src/store_sales/features/common_features.py#L115-L169)
- Static covariates (cho NN): family, city, state, type, cluster, store.

Cấu hình chung: horizon H=16, context Chronos 512, mục tiêu = `log1p(sales)`. Phân loại covariate cho NN: future covs (onpromotion, oil, lịch), past covs (transactions), static covs. ([config.yaml](config.yaml))

---

## Câu 4 — Output / Solution

- Mỗi leg xuất một file submission (id, sales) — dự báo doanh số 16 ngày, đã chuyển ngược từ log về thực bằng `expm1` và clip ≥ 0. [neural_ts.py:266-267](src/store_sales/models/neural_ts.py#L266-L267)
- Tổng hợp: blend phương sai tối thiểu trộn 5 leg trong không gian `log1p` → file cuối `submission_fam_cov_v8_tsm_tide_5way.csv`.
- Kết quả (solution): RMSLE trên public leaderboard = **0.37379** (nhỉnh hơn baseline tác giả tham chiếu 0.37408).

Công thức trộn (không cần nhãn thật): `Cov_ij = (σ_i² + σ_j² − D_ij²)/2`, rồi `w = Σ⁻¹·1 / (1ᵀ·Σ⁻¹·1)`. σ là RMSLE leaderboard của từng leg; D là sai khác RMS giữa dự báo 2 leg. [blend.py:28-75](src/store_sales/ensemble/blend.py#L28-L75)

---

## Câu 5 — Tất cả các operator

Chia 3 nhóm:

**A. Operator tiền xử lý / đặc trưng** ([features/](src/store_sales/features/))

- shift (lag), rolling (mean/std/max), ewm (EWM), pct_change (lợi suất dầu), rolling.std (volatility)
- Fourier: sin/cos của day-of-year
- OneHotEncoder (biến hạng mục), Scaler / StaticCovariatesTransformer (chuẩn hóa, darts)
- InvertibleMapper: `log1p` ↔ `expm1` (biến đổi mục tiêu, khả nghịch). [darts_family.py:44-50](src/store_sales/models/darts_family.py#L44-L50)
- clip, ffill/bfill/fillna (vá khuyết, chống rò rỉ dữ liệu)

**B. Operator trong mô hình**

- GBT: phép tách nút cây (split), cộng dồn cây (gradient boosting), objective tweedie; ba thư viện LightGBM/XGBoost/CatBoost. ([darts_family.py](src/store_sales/models/darts_family.py))
- TSMixer: time-mixing MLP, feature-mixing MLP, LayerNorm, Dropout, residual add, Linear projection. [neural_ts.py:182-191](src/store_sales/models/neural_ts.py#L182-L191)
- TiDE: encoder/decoder MLP, dense projection, dropout. [neural_ts.py:173-181](src/store_sales/models/neural_ts.py#L173-L181)
- Chronos-2: self-attention Transformer (zero-shot). ([chronos2.py](src/store_sales/models/chronos2.py))
- Tối ưu cho NN: Adam, CosineAnnealingLR, loss MSE. [neural_ts.py:188-190](src/store_sales/models/neural_ts.py#L188-L190)

**C. Operator ở tầng ensemble** ([blend.py](src/store_sales/ensemble/blend.py))

- Tái dựng ma trận hiệp phương sai (reconstruct_cov). [blend.py:28-54](src/store_sales/ensemble/blend.py#L28-L54)
- Nghịch đảo ma trận np.linalg.inv. [blend.py:70-73](src/store_sales/ensemble/blend.py#L70-L73)
- Tính trọng số phương sai tối thiểu (tích ma trận–vector), cho phép trọng số âm để triệt tiêu sai số chung. [blend.py:57-75](src/store_sales/ensemble/blend.py#L57-L75)
- Chiếu trọng số không âm + trộn theo alpha (family_alpha=0.75). [blend.py:78-113](src/store_sales/ensemble/blend.py#L78-L113)
- Tổng có trọng số các dự báo log + `expm1` + `clip(min=0)`. [blend.py:121-140](src/store_sales/ensemble/blend.py#L121-L140)
