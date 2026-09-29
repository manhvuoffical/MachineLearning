# Dự đoán giá nhà bằng Hồi quy tuyến tính

Bài tập môn học: xây dựng mô hình **Linear Regression** để dự đoán giá nhà dựa trên các đặc điểm: diện tích, số phòng ngủ, số tầng, tuổi nhà, khoảng cách đến trung tâm.

## Cấu trúc thư mục

```
house-price-regression/
├── data/
│   ├── generate_data.py     # script tạo dữ liệu mẫu
│   └── house_prices.csv     # dữ liệu giá nhà (300 dòng)
├── linear_regression.py     # script chính: huấn luyện + đánh giá mô hình
├── requirements.txt         # thư viện cần cài
├── .gitignore
└── README.md
```

## Dữ liệu

File `data/house_prices.csv` gồm các cột:

| Cột                      | Ý nghĩa                                      |
| ------------------------ | -------------------------------------------- |
| dien_tich_m2             | Diện tích nhà (m2)                           |
| so_phong_ngu             | Số phòng ngủ                                 |
| so_tang                  | Số tầng                                      |
| tuoi_nha                 | Tuổi nhà (năm)                               |
| khoang_cach_trung_tam_km | Khoảng cách đến trung tâm (km)               |
| gia_nha_ty_vnd           | Giá nhà (tỷ VNĐ) — biến mục tiêu cần dự đoán |

> Dữ liệu này là dữ liệu **giả lập** (sinh ngẫu nhiên có quy luật) để phục vụ mục đích học tập. Nếu bài tập yêu cầu dữ liệu thật, xem mục "Thay bằng dữ liệu thật" bên dưới.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy

```bash
# (tuỳ chọn) tạo lại dữ liệu mới
python3 data/generate_data.py

# chạy mô hình hồi quy
python3 linear_regression.py
```

Kết quả in ra gồm: RMSE, R², hệ số hồi quy của từng biến, và một biểu đồ `ket_qua_du_doan.png` so sánh giá thực tế và giá dự đoán.

## Tác giả

Mạnh Vũ
