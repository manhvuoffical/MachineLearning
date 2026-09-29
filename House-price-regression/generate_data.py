"""
Script tạo dữ liệu giá nhà mẫu (giả lập nhưng có quy luật thực tế)
Chạy 1 lần để tạo file data/house_prices.csv
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 300  # số căn nhà

dien_tich = np.random.normal(90, 30, n).clip(25, 250)          # m2
so_phong_ngu = np.random.randint(1, 6, n)                       # phòng
so_tang = np.random.randint(1, 4, n)                            # tầng
tuoi_nha = np.random.randint(0, 30, n)                          # năm
khoang_cach_trung_tam = np.random.normal(8, 5, n).clip(0.5, 30) # km

# Công thức giả lập giá nhà (tỷ VNĐ), có thêm nhiễu ngẫu nhiên
gia_nha = (
    0.03 * dien_tich
    + 0.25 * so_phong_ngu
    + 0.15 * so_tang
    - 0.02 * tuoi_nha
    - 0.05 * khoang_cach_trung_tam
    + np.random.normal(0, 0.3, n)
    + 1.0
).clip(0.3, None)

df = pd.DataFrame({
    "dien_tich_m2": dien_tich.round(1),
    "so_phong_ngu": so_phong_ngu,
    "so_tang": so_tang,
    "tuoi_nha": tuoi_nha,
    "khoang_cach_trung_tam_km": khoang_cach_trung_tam.round(1),
    "gia_nha_ty_vnd": gia_nha.round(3),
})

df.to_csv("data/house_prices.csv", index=False)
print("Đã tạo file data/house_prices.csv với", len(df), "dòng dữ liệu")
