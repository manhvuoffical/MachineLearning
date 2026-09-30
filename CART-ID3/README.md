# Decision Tree - Bài tập Cây Quyết Định

## Cấu trúc thư mục

```
decision_tree/
├── README.md
├── exercise1/          # Bài 1: Dự đoán buys_computer
│   ├── id3.py          # Thuật toán ID3 (Information Gain)
│   └── cart.py         # Thuật toán CART (Gini Impurity)
└── exercise2/          # Bài 2: Dự đoán Rủi ro tín dụng
    └── id3.py          # Thuật toán ID3 (có xử lý thuộc tính liên tục)
```

## Bài 1 - buys_computer (14 mẫu)

**Thuộc tính:** age, income, student, credit_rating  
**Nhãn:** buys_computer (yes/no)

### ID3
- Dùng **Information Gain** (Entropy) để chọn thuộc tính split
- `python exercise1/id3.py`

### CART
- Dùng **Gini Impurity** để chọn thuộc tính split
- `python exercise1/cart.py`

## Bài 2 - Rủi ro tín dụng (15 mẫu)

**Thuộc tính:**
- Rời rạc: Hôn nhân, Sở hữu BĐS
- Liên tục: Độ tuổi (age), Thu nhập (income)

**Nhãn:** Rủi ro tín dụng (0 = không rủi ro, 1 = rủi ro)

### ID3 mở rộng
- Thuộc tính rời rạc: multi-way split
- Thuộc tính liên tục: binary split với **ngưỡng tốt nhất** (midpoint giữa hai giá trị liên tiếp)
- `python exercise2/id3.py`

## Chạy tất cả

```bash
python exercise1/id3.py
python exercise1/cart.py
python exercise2/id3.py
```

## Yêu cầu

Python >= 3.8 (không cần thư viện ngoài, chỉ dùng `math` và `collections`)
