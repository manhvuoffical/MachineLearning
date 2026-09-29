"""
Bài tập: Dự đoán giá nhà bằng Hồi quy tuyến tính (Linear Regression)
Tác giả: (điền tên bạn)

Cách chạy:
    python3 linear_regression.py
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------------------------
# 1. Đọc dữ liệu
# ----------------------------------------------------------------------
df = pd.read_csv("house_prices.csv")
print("5 dòng đầu tiên của dữ liệu:")
print(df.head())
print("\nThông tin dữ liệu:")
print(df.describe())

# ----------------------------------------------------------------------
# 2. Tách biến đầu vào (X) và biến mục tiêu (y)
# ----------------------------------------------------------------------
X = df[["dien_tich_m2", "so_phong_ngu", "so_tang", "tuoi_nha", "khoang_cach_trung_tam_km"]]
y = df["gia_nha_ty_vnd"]

# ----------------------------------------------------------------------
# 3. Chia dữ liệu: 80% train, 20% test
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------------------------------
# 4. Huấn luyện mô hình hồi quy tuyến tính
# ----------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------------------------------------------------
# 5. Dự đoán trên tập test
# ----------------------------------------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------------------------------------
# 6. Đánh giá mô hình
# ----------------------------------------------------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== KẾT QUẢ MÔ HÌNH =====")
print(f"RMSE (sai số căn bậc hai trung bình): {rmse:.3f} tỷ VNĐ")
print(f"R^2 (độ phù hợp mô hình)             : {r2:.3f}")

print("\nHệ số hồi quy (mức ảnh hưởng của từng biến):")
for ten_bien, he_so in zip(X.columns, model.coef_):
    print(f"  {ten_bien:30s}: {he_so:.4f}")
print(f"  {'Hằng số (intercept)':30s}: {model.intercept_:.4f}")

# ----------------------------------------------------------------------
# 7. Vẽ biểu đồ so sánh giá thực tế và giá dự đoán
# ----------------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", label="Dự đoán hoàn hảo")
plt.xlabel("Giá thực tế (tỷ VNĐ)")
plt.ylabel("Giá dự đoán (tỷ VNĐ)")
plt.title("So sánh giá nhà thực tế và giá dự đoán")
plt.legend()
plt.tight_layout()
plt.savefig("ket_qua_du_doan.png", dpi=150)
print("\nĐã lưu biểu đồ vào file: ket_qua_du_doan.png")

# ----------------------------------------------------------------------
# 8. Ví dụ: dự đoán giá cho 1 căn nhà mới
# ----------------------------------------------------------------------
nha_moi = pd.DataFrame({
    "dien_tich_m2": [100],
    "so_phong_ngu": [3],
    "so_tang": [2],
    "tuoi_nha": [5],
    "khoang_cach_trung_tam_km": [4],
})
gia_du_doan = model.predict(nha_moi)[0]
print(f"\nVí dụ: nhà 100m2, 3 phòng ngủ, 2 tầng, 5 năm tuổi, cách trung tâm 4km")
print(f"=> Giá dự đoán: {gia_du_doan:.3f} tỷ VNĐ")
