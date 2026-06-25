# Báo cáo 5 câu hỏi — Dự báo doanh số siêu thị (Store Sales Forecasting)

## Bài toán đang giải là gì?

Đây là cuộc thi trên Kaggle. Một chuỗi siêu thị ở Ecuador muốn **đoán trước 16 ngày tới mỗi quầy hàng sẽ bán được bao nhiêu**.

- "Mỗi quầy hàng" = một cặp **(cửa hàng, ngành hàng)**, ví dụ: *cửa hàng số 3 – ngành hàng nước giải khát*. Có tất cả **1.782 cặp** như vậy.
- Cần đoán doanh số cho **16 ngày** (từ 16/8 đến 31/8/2017).
- Đoán càng sát thực tế càng tốt. Sai số được chấm bằng một thước đo tên là **RMSLE** (số càng nhỏ càng giỏi).

**Cách làm của bài này:** thay vì tin vào một mô hình duy nhất, ta dùng **5 mô hình khác nhau** cùng đoán, rồi **lấy trung bình có trọng số** kết quả của chúng. Giống như hỏi ý kiến 5 chuyên gia rồi tổng hợp lại — thường chính xác hơn nghe một người. Kết quả cuối đạt điểm **0.37379** (rất tốt).

5 mô hình (gọi là 5 "chân"):
1. **GBT family** — nhóm mô hình cây quyết định (LightGBM/XGBoost/CatBoost)
2. **Chronos-2** — mô hình AI khổng lồ đã được huấn luyện sẵn
3. **LightGBM-v8** — một mô hình cây khác, tinh chỉnh riêng
4. **TSMixer** — mạng nơ-ron
5. **TiDE** — mạng nơ-ron

---

## Câu 1 — Cấu trúc giải thuật / Lan truyền xuôi (forward)

> **"Lan truyền xuôi" hiểu đơn giản:** là quá trình **đưa dữ liệu vào → mô hình tính toán → cho ra dự đoán**. Giống như bỏ nguyên liệu vào máy xay rồi nhận ly sinh tố ở đầu ra.

**Toàn bộ quy trình (nhìn từ trên xuống):**

```
Dữ liệu thô  →  Tạo đặc trưng  →  5 mô hình cùng đoán  →  Trộn kết quả  →  File nộp bài
   (CSV)        (lịch, dầu...)      (mỗi cái 1 kiểu)      (trung bình)    (submission.csv)
```

Quy trình này được điều phối ở [cli.py](src/store_sales/cli.py), rồi gộp lại ở [ensemble/build.py](src/store_sales/ensemble/build.py).

**Riêng 2 mạng nơ-ron (TSMixer / TiDE)** — đây mới là "lan truyền xuôi" theo đúng nghĩa kỹ thuật. Code ở [neural_ts.py](src/store_sales/models/neural_ts.py):

- Đưa vào dữ liệu **90 ngày gần nhất** (cùng vài thông tin phụ như khuyến mãi, giá dầu, ngày lễ) → mạng tính toán → **xuất ra luôn dự đoán cho 16 ngày tới một lần** (không đoán từng ngày một).
- **TSMixer** giống một dây chuyền có 8 công đoạn ([neural_ts.py:182-191](src/store_sales/models/neural_ts.py#L182-L191)). Mỗi công đoạn làm 2 việc: trộn thông tin **theo thời gian** (hôm nay liên quan hôm qua thế nào) và trộn thông tin **giữa các đặc trưng** (khuyến mãi liên quan giá dầu thế nào). Cuối dây chuyền cho ra con số 16 ngày.
- **TiDE** đơn giản hơn: nén thông tin lại rồi bung ra thành dự đoán ([neural_ts.py:173-181](src/store_sales/models/neural_ts.py#L173-L181)).

**Các mô hình còn lại:**
- **GBT (mô hình cây):** "lan truyền xuôi" = dữ liệu đi qua một loạt cây quyết định (kiểu sơ đồ "nếu... thì..."), mỗi cây góp một phần vào con số cuối.
- **Chronos-2:** chỉ đưa dữ liệu vào và nhận kết quả, **không cần dạy lại** (xem Câu 2).

---

## Câu 2 — Lan truyền ngược (backward)

> **"Lan truyền ngược" hiểu đơn giản:** là cách mô hình **tự sửa sai khi học**. Sau khi đoán xong, nó so với đáp án thật, thấy lệch bao nhiêu, rồi **điều chỉnh ngược lại các thông số bên trong** cho lần sau đoán đúng hơn. Giống học sinh làm bài, dò đáp án, rồi rút kinh nghiệm.

Trong bài này, **chỉ có 2 mạng nơ-ron (TSMixer / TiDE) là học theo kiểu "lan truyền ngược" thật sự:**

- **Đo sai số:** so dự đoán với doanh số thật, tính độ lệch (hàm "loss").
- **Sửa lại:** dùng thuật toán tên **Adam** để chỉnh dần các thông số, học đi học lại **30 lượt (epoch)**. Tốc độ học được giảm dần cho mượt. ([neural_ts.py:188-190](src/store_sales/models/neural_ts.py#L188-L190))

**Lưu ý quan trọng (thầy hay hỏi vặn chỗ này):** không phải cả 5 mô hình đều "lan truyền ngược".

- **GBT (cây):** học theo kiểu **khác** — gọi là *boosting*. Nó trồng cây sau để **sửa lỗi của cây trước**, cộng dồn dần. Có dùng "độ lệch" nhưng **không phải** lan truyền ngược như mạng nơ-ron. ([blend.py có giải thích phần trộn](src/store_sales/ensemble/blend.py#L57-L75))
- **Chronos-2:** **không học gì cả** trong bài này — nó là AI đã được huấn luyện sẵn từ trước, ta chỉ việc dùng (gọi là "zero-shot").
- **Bước trộn 5 kết quả:** cũng **không học** — chỉ giải một công thức toán có sẵn để tìm tỉ lệ trộn tốt nhất. ([blend.py:57-75](src/store_sales/ensemble/blend.py#L57-L75))

---

## Câu 3 — Input (Đầu vào) là gì?

**1. Dữ liệu thô** — các file Excel/CSV của cuộc thi ([data_loading.py](src/store_sales/io/data_loading.py), thư mục [data/](data/)):

| File | Chứa gì |
|------|---------|
| `train.csv` / `test.csv` | Lịch sử bán hàng: ngày, cửa hàng, ngành hàng, doanh số, có khuyến mãi hay không |
| `oil.csv` | Giá dầu mỗi ngày (kinh tế Ecuador phụ thuộc dầu → ảnh hưởng sức mua) |
| `holidays_events.csv` | Lịch ngày lễ, sự kiện |
| `stores.csv` | Thông tin cửa hàng: thành phố, tỉnh, loại, nhóm |
| `transactions.csv` | Số lượt giao dịch mỗi ngày |

**2. Đặc trưng được tạo thêm** — từ dữ liệu thô, ta "chế biến" ra các gợi ý hữu ích cho mô hình:

- **Quá khứ bán hàng:** doanh số 1–21 ngày trước, trung bình tuần/tháng trước... (giúp mô hình thấy xu hướng). [lgbm_features.py](src/store_sales/features/lgbm_features.py)
- **Thông tin lịch:** thứ mấy, tháng mấy, ngày trả lương (15 và cuối tháng), có phải ngày lễ không. [neural_ts.py:86-96](src/store_sales/models/neural_ts.py#L86-L96)
- **Giá dầu:** dầu tăng/giảm bao nhiêu, biến động mạnh hay nhẹ. [common_features.py:39-72](src/store_sales/features/common_features.py#L39-L72)
- **Gần ngày lễ:** còn mấy ngày tới lễ / vừa qua lễ mấy ngày (dân hay mua sắm quanh lễ). [common_features.py:115-169](src/store_sales/features/common_features.py#L115-L169)
- **Thông tin cố định của cửa hàng:** thành phố, loại, nhóm...

Tất cả thiết lập chung (đoán 16 ngày, mục tiêu...) nằm ở [config.yaml](config.yaml).

> Một mẹo nhỏ: thay vì đoán thẳng doanh số, mô hình đoán theo dạng `log` (nén số lớn lại) cho ổn định, rồi đổi ngược về số thật ở cuối.

---

## Câu 4 — Output (Đầu ra) là gì?

- Mỗi mô hình tạo ra một **file kết quả** gồm 2 cột: *mã dòng* và *doanh số dự đoán* cho 16 ngày. Số dự đoán được đổi về đơn vị thật và **không cho âm** (không thể bán âm hàng). [neural_ts.py:266-267](src/store_sales/models/neural_ts.py#L266-L267)
- Sau đó **trộn kết quả của cả 5 mô hình** lại thành một file cuối: `submission_fam_cov_v8_tsm_tide_5way.csv`.
- Nộp file này lên Kaggle → được chấm điểm **0.37379** (điểm rất tốt, nhỉnh hơn cả bài tham khảo gốc).

**Cách trộn dựa trên ý tưởng đơn giản:** mô hình nào **đoán giỏi hơn** thì cho **trọng số cao hơn**; và nếu 2 mô hình hay **sai giống nhau** thì giảm bớt để tránh "cùng sai một kiểu". Máy tự tính ra tỉ lệ trộn tối ưu bằng một công thức toán (không cần biết đáp án thật). [blend.py:28-75](src/store_sales/ensemble/blend.py#L28-L75)

---

## Câu 5 — Tất cả các "operator" (phép xử lý) gồm những gì?

Gom thành 3 nhóm:

**A. Nhóm xử lý / chế biến dữ liệu** ([features/](src/store_sales/features/))

- Lấy số liệu của các ngày trước (lag), tính trung bình động (rolling), trung bình có ưu tiên ngày gần (EWM)
- Tính phần trăm tăng/giảm giá dầu, mức độ biến động
- Biến "thứ trong năm" thành dạng sóng tuần hoàn (sin/cos) để mô hình hiểu tính mùa vụ
- Chuyển chữ thành số (ví dụ tên thành phố → mã số), chuẩn hóa số liệu về cùng thang
- Nén số bằng `log` rồi đổi ngược lại ([darts_family.py:44-50](src/store_sales/models/darts_family.py#L44-L50))
- Vá các ô bị thiếu dữ liệu (điền giá trị gần nhất), chặn giá trị âm

**B. Nhóm bên trong các mô hình**

- **GBT (cây):** chia nhánh "nếu... thì...", cộng dồn nhiều cây để ra kết quả ([darts_family.py](src/store_sales/models/darts_family.py))
- **TSMixer:** các lớp trộn theo thời gian và theo đặc trưng, có chuẩn hóa và "bỏ bớt ngẫu nhiên" (dropout) để tránh học vẹt ([neural_ts.py:182-191](src/store_sales/models/neural_ts.py#L182-L191))
- **TiDE:** lớp nén — lớp bung để tạo dự đoán ([neural_ts.py:173-181](src/store_sales/models/neural_ts.py#L173-L181))
- **Chronos-2:** dùng cơ chế "attention" của AI ngôn ngữ lớn ([chronos2.py](src/store_sales/models/chronos2.py))
- **Bộ tối ưu khi học:** thuật toán Adam, giảm tốc độ học dần ([neural_ts.py:188-190](src/store_sales/models/neural_ts.py#L188-L190))

**C. Nhóm trộn kết quả 5 mô hình** ([blend.py](src/store_sales/ensemble/blend.py))

- Đo xem các mô hình hay sai giống nhau tới mức nào ([blend.py:28-54](src/store_sales/ensemble/blend.py#L28-L54))
- Giải công thức toán để tìm tỉ lệ trộn tốt nhất ([blend.py:57-75](src/store_sales/ensemble/blend.py#L57-L75))
- Pha trộn theo tỉ lệ đó rồi đổi số về đơn vị thật, chặn giá trị âm ([blend.py:121-140](src/store_sales/ensemble/blend.py#L121-L140))

---

*Mọi đường link trong báo cáo đều bấm được để mở đúng file và đúng dòng code.*
