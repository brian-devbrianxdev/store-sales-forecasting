# Báo cáo trả lời 5 câu hỏi — Store Sales Forecasting

> **Bài toán:** Cuộc thi Kaggle *Store Sales – Time Series Forecasting*. Dự báo doanh số **16 ngày** (2017-08-16 → 2017-08-31) cho từng cặp **(cửa hàng, ngành hàng)** — tổng cộng 1.782 chuỗi thời gian. Độ đo: **RMSLE**.
>
> **Giải pháp (solution):** KHÔNG phải một mạng nơ-ron đơn lẻ, mà là một **ensemble (tổ hợp) 5 chân mô hình** bổ trợ nhau, rồi **trộn theo phương sai tối thiểu (minimum-variance blend)** trong không gian `log1p`. Kết quả cuối: **LB RMSLE = 0.37379**.
>
> 5 chân: GBT family (darts LightGBM/XGBoost/CatBoost), Chronos-2 (foundation model), LightGBM-v8, TSMixer, TiDE.

---

## Câu 1 — Cấu trúc giải thuật / Lan truyền xuôi (forward propagation)

### Mức tổng thể (pipeline forward)
Luồng đi qua: `cli.py` → từng leg trong `models/` → `ensemble/build.py`

```
Raw CSV → Feature engineering → 5 leg dự báo độc lập → Min-variance blend → submission.csv
```

### Mức mạng nơ-ron (TSMixer / TiDE) — đây mới là "lan truyền xuôi" đúng nghĩa
Code: `src/store_sales/models/neural_ts.py`

- Đầu vào một cửa sổ quá khứ `input_chunk_length = 90` ngày (TSMixer) cùng covariates → mạng → xuất thẳng **H = 16 ngày** tương lai (dự báo trực tiếp đa-bước, `output_chunk_length = 16`).
- **TSMixer forward:** chồng **8 khối** (`num_blocks = 8`), mỗi khối gồm:
  - *time-mixing MLP* (trộn theo chiều thời gian)
  - *feature-mixing MLP* (trộn theo chiều đặc trưng)
  - kèm chuẩn hóa (LayerNorm) + Dropout(0.2) + kết nối residual
  - lớp tuyến tính cuối chiếu ra 16 ngày.
- **TiDE forward:** encoder MLP → bottleneck → decoder MLP, có dùng static covariates.

### Các chân khác
- **GBT (cây):** "forward" = đi qua chuỗi cây quyết định, mỗi cây cộng dồn dự báo.
- **Chronos-2:** zero-shot, forward = một lượt qua Transformer, **không huấn luyện**.

---

## Câu 2 — Lan truyền ngược (backward propagation)

Chỉ **các chân nơ-ron (TSMixer / TiDE / NHiTS)** mới dùng backpropagation đúng nghĩa, qua PyTorch (darts):

- **Hàm mất mát:** MSE trên mục tiêu `log1p(sales)`.
- **Backprop + tối ưu:** optimizer **Adam**, learning rate `0.001`, lịch học `CosineAnnealingLR` theo số epoch. Huấn luyện 30 epoch, batch size 1024. Gradient được lan ngược qua các khối mixing để cập nhật trọng số.

### Phân biệt rõ (điểm dễ bị hỏi vặn)
- **GBT (LightGBM / XGBoost / CatBoost):** *KHÔNG* dùng backprop — mà là **gradient boosting**: mỗi cây mới fit theo *gradient của hàm mất mát* (objective `tweedie`, power 1.2–1.3) so với dự báo hiện tại, rồi cộng dồn. Có "đạo hàm" nhưng theo kiểu boosting, không phải lan truyền ngược qua mạng nơ-ron.
- **Chronos-2:** zero-shot, **không có pha huấn luyện/backward** nào.
- **Tầng blend (ensemble):** không học bằng gradient — dùng **nghiệm đóng (closed-form)**:

  ```
  w = Σ⁻¹·1 / (1ᵀ·Σ⁻¹·1)
  ```

---

## Câu 3 — Input là gì

### Dữ liệu thô (thư mục `data/`, đọc bởi `io/data_loading.py`)
| File | Nội dung |
|------|----------|
| `train.csv` / `test.csv` | `id, date, store_nbr, family, sales, onpromotion` |
| `oil.csv` | Giá dầu WTI hằng ngày (Ecuador phụ thuộc dầu mỏ) |
| `holidays_events.csv` | Lịch lễ/sự kiện (national/regional/local, transferred…) |
| `stores.csv` | Metadata cửa hàng (city, state, type, cluster) |
| `transactions.csv` | Số giao dịch |

### Đặc trưng đưa vào mô hình (sau feature engineering)
- **Tự hồi quy (cho GBT):** lag doanh số (1→21, 28…364), rolling mean/std/max (cửa sổ 7…168), EWM (halflife 7/28), theo từng horizon.
- **Lịch:** day-of-week, month, day, payday (ngày 15 & cuối tháng), cờ ngày lễ quốc gia, **Fourier day-of-year** (sin/cos bậc 1–3).
- **Dầu (động lực học):** lợi suất 7/28 ngày, lag 16/28/56, volatility 28 ngày.
- **Ngày lễ (khoảng cách):** số ngày tới/kể từ ngày lễ đặc biệt gần nhất, cờ transferred.
- **Static covariates (cho NN):** family, city, state, type, cluster, store.

### Cấu hình chung
- Horizon `H = 16`, context Chronos `512`.
- Mục tiêu = `log1p(sales)`.
- Phân loại covariate cho NN: future covs (onpromotion, oil, lịch), past covs (transactions), static covs.

---

## Câu 4 — Output là gì (solution)

- **Mỗi leg** xuất một file submission `(id, sales)` — dự báo doanh số 16 ngày, đã chuyển ngược từ log về thực bằng `expm1` và clip ≥ 0.
- **Tổng hợp:** blend phương sai tối thiểu trộn 5 leg trong không gian `log1p` → file cuối **`submission_fam_cov_v8_tsm_tide_5way.csv`**.
- **Kết quả (solution):** RMSLE trên public leaderboard = **0.37379** (nhỉnh hơn baseline tham chiếu 0.37408).

### Công thức trộn (không cần nhãn thật)
```
Cov_ij = (σ_i² + σ_j² − D_ij²) / 2
w      = Σ⁻¹·1 / (1ᵀ·Σ⁻¹·1)
```
- `σ` = RMSLE leaderboard của từng leg.
- `D` = sai khác RMS giữa dự báo của 2 leg (2 leg càng bất đồng → sai số càng ít tương quan → trộn càng lợi).

---

## Câu 5 — Tất cả các operator

### A. Operator tiền xử lý / đặc trưng (`features/`)
- `shift` (lag), `rolling` (mean/std/max), `ewm` (EWM)
- `pct_change` (lợi suất dầu), `rolling.std` (volatility)
- Fourier: `sin` / `cos` của day-of-year
- `OneHotEncoder` (biến hạng mục)
- `Scaler` / `StaticCovariatesTransformer` (chuẩn hóa, darts)
- `InvertibleMapper`: `log1p` ↔ `expm1` (biến đổi mục tiêu, khả nghịch)
- `clip`, `ffill` / `bfill` / `fillna` (vá khuyết, chống rò rỉ dữ liệu)

### B. Operator trong mô hình
- **GBT:** phép tách nút cây (split), cộng dồn cây (gradient boosting), objective **tweedie**; ba thư viện LightGBM / XGBoost / CatBoost.
- **TSMixer:** time-mixing MLP, feature-mixing MLP, LayerNorm, Dropout, residual add, Linear projection.
- **TiDE:** encoder/decoder MLP, dense projection, dropout.
- **Chronos-2:** self-attention Transformer (zero-shot).
- **Tối ưu cho NN:** Adam, CosineAnnealingLR, loss MSE.

### C. Operator ở tầng ensemble (`ensemble/blend.py`)
- Tái dựng ma trận hiệp phương sai (`reconstruct_cov`)
- Nghịch đảo ma trận `np.linalg.inv`
- Tính trọng số phương sai tối thiểu (tích ma trận–vector), cho phép **trọng số âm** để triệt tiêu sai số chung
- Chiếu trọng số không âm + trộn theo `alpha` (`family_alpha = 0.75`)
- Tổng có trọng số các dự báo log + `expm1` + `clip(min=0)`

---

*Báo cáo dựa trên mã nguồn repo `store-sales-forecasting` — các file chính: `cli.py`, `models/neural_ts.py`, `models/darts_family.py`, `features/common_features.py`, `features/lgbm_features.py`, `ensemble/blend.py`, `config.yaml`.*
