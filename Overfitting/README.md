# Overfitting Lab — Gây overfitting rồi khắc phục

Bài tập môn Học máy: huấn luyện một mô hình, **cố ý làm nó overfitting**, sau đó áp dụng
các kỹ thuật chống overfitting đã học và so sánh kết quả bằng số liệu + biểu đồ.

## 1. Dữ liệu

Bộ **`digits`** của scikit-learn: 1797 ảnh chữ số viết tay 8×8 (64 đặc trưng), 10 lớp.
Bộ này đi kèm thư viện nên chạy được offline, không cần tải gì thêm.

| Tập | Số mẫu | Vai trò |
|---|---|---|
| Train | **150** | cố tình lấy rất ít để dễ overfit |
| Validation | 150 | chọn siêu tham số / early stopping |
| Test | 540 | giữ nguyên cho mọi thí nghiệm để so sánh công bằng |

Đặc trưng được chuẩn hoá bằng `StandardScaler` (fit trên tập train để tránh rò rỉ dữ liệu).

## 2. Cách gây overfitting

Công thức gồm 4 yếu tố:

1. **Rất ít dữ liệu** — chỉ 150 mẫu cho bài toán 10 lớp, 64 chiều
2. **Mô hình thừa năng lực** — MLP 2 lớp ẩn 512 nơ-ron → **301.066 tham số**, gấp ~2007 lần số mẫu
3. **Tắt regularization** — `alpha = 0`
4. **Huấn luyện quá lâu** — 300 epoch, không early stopping

### Kết quả phần 1

| Mô hình | Train acc | Test acc | Gap |
|---|---|---|---|
| MLP(512,512), alpha=0, 300 epoch | **1.0000** | 0.8778 | **+0.1222** |
| DecisionTree không giới hạn độ sâu | **1.0000** | 0.6204 | **+0.3796** |

![overfitting](results/01_overfitting_curves.png)

Đọc biểu đồ: sau ~10 epoch, độ chính xác trên tập train đạt 100% và loss về gần 0,
nhưng độ chính xác trên tập test đứng yên ở ~88%. **Khoảng cách giữa hai đường
chính là dấu hiệu overfitting** — mô hình học thuộc lòng 150 mẫu thay vì học quy luật chung.

Cây quyết định còn tệ hơn: học thuộc 100% tập train nhưng chỉ đúng 62% trên test.

## 3. Các kỹ thuật chống overfitting

| # | Kỹ thuật | Train | Test | Gap |
|---|---|---|---|---|
| — | **[GỐC] MLP(512,512) không regularization** | 1.0000 | 0.8778 | +0.1222 |
| 1 | Giảm độ phức tạp: MLP(32) | 1.0000 | 0.8963 | +0.1037 |
| 2 | L2 regularization (alpha = 0.3162) | 1.0000 | 0.9056 | +0.0944 |
| 3 | Early stopping (dừng ở epoch 18) | 0.9800 | 0.8352 | +0.1448 |
| 4 | Thêm dữ liệu (150 → 1257 mẫu) | 1.0000 | 0.9759 | +0.0241 |
| 5 | Data augmentation (150 → 1500 mẫu) | 1.0000 | 0.9130 | +0.0870 |
| 6 | GridSearchCV 5-fold | 1.0000 | 0.9093 | +0.0907 |
| 7 | Ensemble: RandomForest(300 cây) | 1.0000 | 0.9000 | +0.1000 |
| + | DecisionTree sau pruning (`ccp_alpha`) | 0.9133 | 0.6204 | +0.2930 |
| ★ | **KẾT HỢP tất cả** | 0.9992 | **0.9815** | **+0.0177** |

![so sánh](results/04_summary_comparison.png)

### Giải thích từng kỹ thuật

**(1) Giảm độ phức tạp mô hình** — từ 512×512 xuống 32 nơ-ron. Ít tham số hơn thì mô hình
không đủ "chỗ trống" để ghi nhớ từng mẫu, buộc phải học đặc trưng chung. Gap giảm từ 0.122 → 0.104.

**(2) L2 regularization (weight decay)** — thêm `alpha·‖W‖²` vào hàm mất mát để phạt các
trọng số lớn, làm hàm quyết định mượt hơn. `alpha` được chọn bằng validation curve:

![validation curve](results/02_validation_curve_alpha.png)

Đường train và validation tách xa nhau ở alpha nhỏ (overfitting) và cùng tụt xuống khi alpha
quá lớn (underfitting). Điểm tốt nhất ở khoảng **alpha ≈ 0.32** — đây chính là minh hoạ
trực quan cho **bias–variance tradeoff**.

**(3) Early stopping** — tách 20% train làm validation, dừng khi validation không cải thiện
trong 10 epoch liên tiếp → dừng ở epoch 18 thay vì 300.
⚠️ Ở đây early stopping lại **làm test giảm** (0.8352). Lý do: tập train vốn đã chỉ có 150 mẫu,
cắt thêm 20% thì chỉ còn 120 mẫu để học. Đây là bài học đáng chú ý —
*early stopping rất hiệu quả khi dữ liệu đủ nhiều, nhưng phản tác dụng khi dữ liệu quá ít.*
Khi kết hợp với tập dữ liệu đầy đủ (dòng ★) thì nó phát huy tác dụng ngay.

**(4) Thêm dữ liệu** — cách chống overfitting hiệu quả nhất. Dùng đủ 1257 mẫu thay vì 150:
test tăng vọt lên 0.9759, gap còn 0.024. Cùng một mô hình khổng lồ 301K tham số, chỉ khác lượng dữ liệu.

**(5) Data augmentation** — khi không xin thêm được dữ liệu thật thì tự tạo: nhân bản mẫu cũ
và thêm nhiễu Gaussian (với ảnh thật thường dùng xoay/dịch/lật/zoom). 150 → 1500 mẫu,
test tăng từ 0.8778 lên 0.9130.

**(6) Cross-validation + Grid search** — chọn kiến trúc và alpha một cách khách quan bằng
5-fold CV thay vì đoán mò. Cấu hình tốt nhất: `hidden_layer_sizes=(64,)`, `alpha=1.0`.

**(7) Ensemble** — Random Forest trung bình 300 cây, mỗi cây thấy một phần dữ liệu/đặc trưng
khác nhau. Lỗi của từng cây triệt tiêu lẫn nhau: từ 0.6204 (1 cây) lên 0.9000.

**(+) Pruning** — cắt tỉa cây bằng cost-complexity pruning (`ccp_alpha`). Cây rút từ 33 lá
xuống 20 lá, train giảm từ 1.0 → 0.913 (mô hình bớt học thuộc lòng), test giữ nguyên nhưng
mô hình đơn giản hơn nhiều nên dễ giải thích và ổn định hơn.

**(★) Kết hợp** — nhiều dữ liệu + kiến trúc vừa phải (128,64) + L2 + early stopping:
**test 0.9815, gap chỉ 0.0177**. Đây là kết quả tốt nhất, cho thấy các kỹ thuật này
bổ trợ nhau chứ không loại trừ nhau.

## 4. Kết luận

1. Overfitting xảy ra khi **năng lực mô hình vượt xa lượng thông tin trong dữ liệu**.
2. Dấu hiệu nhận biết: train acc rất cao, test acc thấp, **gap lớn**.
3. Xếp hạng hiệu quả trong thí nghiệm này: thêm dữ liệu > augmentation > L2 > giảm độ phức tạp ≈ ensemble.
4. Không có kỹ thuật nào là "thuốc tiên": early stopping còn làm giảm kết quả khi dữ liệu quá ít.
5. Kết hợp nhiều kỹ thuật cho kết quả tốt nhất.

## 5. Cách chạy

```bash
pip install -r requirements.txt

cd src
python train_overfit.py        # Phần 1: gây overfitting
python train_regularized.py    # Phần 2: các kỹ thuật khắc phục
```

Biểu đồ và số liệu được ghi vào thư mục `results/`.

## 6. Cấu trúc thư mục

```
overfitting-lab/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── data.py               # tải, chia, chuẩn hoá, augment dữ liệu
│   ├── utils.py              # huấn luyện theo epoch, vẽ biểu đồ, báo cáo
│   ├── train_overfit.py      # PHẦN 1 — cố ý gây overfitting
│   └── train_regularized.py  # PHẦN 2 — 8 kỹ thuật chống overfitting
└── results/                  # biểu đồ .png + số liệu .json
```

## Môi trường

Python 3.11, scikit-learn 1.8.0, numpy 2.4.4, matplotlib 3.10.8.
Mọi thí nghiệm dùng `random_state=42` nên kết quả tái lập được.
