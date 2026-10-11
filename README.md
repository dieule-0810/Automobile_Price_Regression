# Dự đoán giá xe ô tô (Regression)

**Dataset:** Automobile Price Dataset.\
**Biến mục tiêu:** `Price` (giá xe, giá trị liên tục → bài toán Regression).
---

## 1. Cấu trúc thư mục

```text
24280080_NguyenDieuLe_Assignment
├── README.md                               # hướng dẫn chạy (file này)
├── report.md                               # báo cáo kết quả & nhận xét
├── requirements.txt                        # thư viện cần cài
├── main.py                                 # điểm chạy chính
├── EDA.ipynb                               # notebook khám phá dữ liệu
├── image_EDA/                              # visualization từ EDA
├── data/
│   ├── train.csv                           # dữ liệu huấn luyện (có cột Price)
│   └── test.csv                            # dữ liệu dự đoán (không có Price)
├── outputs/
│   ├── results.csv                         # bảng so sánh các mô hình
│   ├── feature_importance.csv              # độ quan trọng của feature
│   ├── shap_importance.csv                 # độ quan trọng của feature (có chiều)
│   ├── learning_curves_all_models.csv      # learning curve của tất cả model
│   ├── overfit_underfit_tree_family.csv    # mức độ overfit và underfit của các model họ cây
│   ├── sample_predictions.csv              # mẫu dự đoán
│   ├── test_predictions.csv                # KẾT QUẢ DỰ ĐOÁN cho test.csv
│   └── figures/                            # toàn bộ biểu đồ .png
└── src/
    ├── __init__.py
    └── regression.py                       # toàn bộ logic xử lý/huấn luyện/đánh giá
```

---

## 2. Hướng dẫn chạy các file

## 2.1. Khám phá dữ liệu trên file EDA.ipynb
Chạy file EDA.ipynb trên Colab hoặc VScode.\
**Lưu ý:** Các nhận xét về kiểu dữ liệu trong file EDA.ipynb chỉ tương thích với kết quả khi chạy trên Colab. Tuy nhiên, sự khác biệt không ảnh hưởng đến quá trình EDA quan trọng. Chẳng hạn, có sự khác biệt giữa kiểu object/str,...

## 2.2. Cài đặt và chạy mô hình

```bash
python3 -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
```

Cài đặt thư viện:

```bash
pip install -r requirements.txt
```

Chạy bài thực hành:

```bash
python main.py
```

Sau khi chạy, chương trình sẽ:

1. Đọc `data/train.csv`, làm sạch và tiền xử lý.
2. Chia train.csv thành 80% train / 20% test (nội bộ) để đánh giá.
3. Huấn luyện các mô hình: Linear Regression, KNN, Decision Tree (giới hạn và không giới hạn độ sâu), Random Forest.
4. Đánh giá bằng MAE, MSE, RMSE, R²; in bảng so sánh ra màn hình và lưu `outputs/results.csv`.
5. Vẽ biểu đồ so sánh, learning curve, validation curve, feature importance, actual-vs-predicted, residual.
6. Chọn mô hình tốt nhất theo Test RMSE, huấn luyện lại trên **toàn bộ** train.csv và dự đoán `data/test.csv` → `outputs/test_predictions.csv`.

---

## 3. Bài toán & dữ liệu

Mỗi dòng là một chiếc xe đã qua sử dụng. Các đặc trưng gồm:
- Thông tin kỹ thuật: `Engine volume`, `Cylinders`, `Mileage`, `Fuel type`, `Gear box type`, `Drive wheels`...
- Thương hiệu: `Manufacturer`, `Model`.
- Đặc điểm khác: `Color`, `Leather interior`, `Airbags`, `Levy`...

Mục tiêu là dự đoán `Price`.

### Các bước tiền xử lý (theo EDA)

| Bước | Xử lý |
|---|---|
| Làm sạch cột | `Levy` `"-"`→0 (+ cờ `Levy missing`); tách `Turbo` khỏi `Engine volume`; bỏ `" km"` ở `Mileage`; `Leather interior`→0/1; gộp nhãn `Doors` |
| Biến mới | `Age` = năm hiện tại − `Prod. year`; `Mileage per year` = `Mileage`/(`Prod. year` + 1); `Turbo`; `Levy missing` |
| Bỏ cột | `ID`, `Prod. year` (đã thay bằng `Age`) |
| Outlier (chỉ train) | `Mileage` ≤ 1.000.000; `Engine volume` ≤ 10; `Price` lọc ra với upper_quantile = 0.995 |
| Biến mục tiêu | `Price` (Linear/KNN huấn luyện trên log Price, Decision Tree/ Random Forest huấn luyện trên Price gốc, tất cả đo chỉ số trên Price gốc) |
| Mã hóa | OneHot (gộp nhãn hiếm) cho biến ít nhóm; **Target Encoding** cho `Manufacturer`, `Model`, `Color` |
| Chuẩn hóa | `StandardScaler` cho biến số |

Việc fit scaler/encoder chỉ dựa trên tập train; mọi phép xóa dòng (trùng,
outlier) **không** áp dụng cho `test.csv` để tránh rò rỉ dữ liệu.

### Tiền xử lý riêng theo từng họ mô hình

Mỗi mô hình dùng cách tiền xử lý phù hợp nhất với bản chất của nó:

| Họ mô hình | Scale | Log biến lệch (`Mileage`,`Levy`, `Mileage per year`) | Log biến mục tiêu (Price) |Ghi chú |
|---|---|---|---|---|
| Linear / KNN / SVR | Có | Có | Có | Nhạy với thang đo và độ lệch |
| Decision Tree / Random Forest | Không | Không | Không | Cây bất biến với phép biến đổi đơn điệu của đặc trưng |

### Giải thích mô hình

Ngoài `feature_importance`, chương trình tạo thêm **SHAP** cho mô hình cây tốt
nhất (`outputs/figures/shap_summary.png`, `outputs/shap_importance.csv`), cho
biết mỗi đặc trưng đẩy giá lên/xuống bao nhiêu. Thang của giá trị SHAP phụ thuộc việc target có bị log hay không:\
- Mô hình bọc TransformedTargetRegressor (log) -> SHAP ở thang log-Price.
- Mô hình huấn luyện thẳng trên Price -> SHAP ở thang USD gốc.
---

## 4. Các chỉ số đánh giá

- **MAE**: sai số tuyệt đối trung bình (cùng đơn vị USD, dễ giải thích).
- **MSE**: trung bình bình phương sai số (phạt nặng sai số lớn).
- **RMSE**: căn của MSE (cùng đơn vị USD, nhạy với sai số lớn).
- **R²**: tỉ lệ phương sai của giá được mô hình giải thích (gần 1 là tốt).

Tiêu chí chọn mô hình tốt nhất: **Test RMSE thấp nhất** kết hợp **Test R² cao nhất**.
