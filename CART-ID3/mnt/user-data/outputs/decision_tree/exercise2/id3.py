"""
Bài tập 2: Cây quyết định ID3 - Dự đoán Rủi ro tín dụng
Dataset: 15 mẫu với thuộc tính Độ tuổi (số), Hôn nhân, Sở hữu BĐS, Thu nhập (số)
Nhãn: Rủi ro tín dụng (0 = không rủi ro, 1 = rủi ro)

Lưu ý:
- Thuộc tính liên tục (age, income) cần tìm ngưỡng chia tốt nhất (binary split)
- Thuộc tính rời rạc (hon_nhan, so_huu_bds) dùng multi-way split
"""

import math
from collections import Counter


# ===================== DATA =====================
data = [
    {"id": 1,  "age": 25, "hon_nhan": "Độc thân",    "so_huu_bds": "Ở cùng bố mẹ", "income": 7000000,  "rui_ro": 0},
    {"id": 2,  "age": 40, "hon_nhan": "Đã kết hôn",  "so_huu_bds": "Nhà sở hữu",   "income": 18000000, "rui_ro": 0},
    {"id": 3,  "age": 35, "hon_nhan": "Từng ly hôn", "so_huu_bds": "Nhà thuê",      "income": 12000000, "rui_ro": 1},
    {"id": 4,  "age": 27, "hon_nhan": "Đã kết hôn",  "so_huu_bds": "Ở cùng bố mẹ", "income": 9000000,  "rui_ro": 1},
    {"id": 5,  "age": 31, "hon_nhan": "Độc thân",    "so_huu_bds": "Nhà thuê",      "income": 6000000,  "rui_ro": 1},
    {"id": 6,  "age": 36, "hon_nhan": "Đã kết hôn",  "so_huu_bds": "Nhà sở hữu",   "income": 8000000,  "rui_ro": 1},
    {"id": 7,  "age": 48, "hon_nhan": "Độc thân",    "so_huu_bds": "Nhà thuê",      "income": 7000000,  "rui_ro": 0},
    {"id": 8,  "age": 26, "hon_nhan": "Đã kết hôn",  "so_huu_bds": "Nhà thuê",      "income": 8000000,  "rui_ro": 1},
    {"id": 9,  "age": 33, "hon_nhan": "Từng ly hôn", "so_huu_bds": "Ở cùng bố mẹ", "income": 5000000,  "rui_ro": 1},
    {"id": 10, "age": 29, "hon_nhan": "Độc thân",    "so_huu_bds": "Nhà thuê",      "income": 10000000, "rui_ro": 0},
    {"id": 11, "age": 38, "hon_nhan": "Đã kết hôn",  "so_huu_bds": "Nhà sở hữu",   "income": 15000000, "rui_ro": 0},
    {"id": 12, "age": 44, "hon_nhan": "Độc thân",    "so_huu_bds": "Nhà sở hữu",   "income": 14000000, "rui_ro": 1},
    {"id": 13, "age": 42, "hon_nhan": "Đã kết hôn",  "so_huu_bds": "Nhà sở hữu",   "income": 10000000, "rui_ro": 0},
    {"id": 14, "age": 28, "hon_nhan": "Độc thân",    "so_huu_bds": "Nhà thuê",      "income": 7000000,  "rui_ro": 1},
    {"id": 15, "age": 30, "hon_nhan": "Đã kết hôn",  "so_huu_bds": "Ở cùng bố mẹ", "income": 6000000,  "rui_ro": 1},
]

TARGET = "rui_ro"
CATEGORICAL = ["hon_nhan", "so_huu_bds"]
CONTINUOUS   = ["age", "income"]


# ===================== FUNCTIONS =====================
def entropy(records):
    n = len(records)
    if n == 0:
        return 0
    counts = Counter(r[TARGET] for r in records)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def info_gain_categorical(records, feature):
    """Info Gain cho thuộc tính rời rạc (multi-way split)."""
    n = len(records)
    values = set(r[feature] for r in records)
    weighted = sum(
        (len(subset := [r for r in records if r[feature] == v]) / n) * entropy(subset)
        for v in values
    )
    return entropy(records) - weighted, None  # threshold = None


def best_threshold(records, feature):
    """
    Tìm ngưỡng tốt nhất cho thuộc tính liên tục (binary split).
    Thử các midpoint giữa các giá trị liên tiếp đã sắp xếp.
    """
    values = sorted(set(r[feature] for r in records))
    best_gain, best_t = -1, None
    base_ent = entropy(records)
    n = len(records)

    for i in range(len(values) - 1):
        t = (values[i] + values[i + 1]) / 2
        left  = [r for r in records if r[feature] <= t]
        right = [r for r in records if r[feature] > t]
        weighted = (len(left) / n) * entropy(left) + (len(right) / n) * entropy(right)
        gain = base_ent - weighted
        if gain > best_gain:
            best_gain, best_t = gain, t

    return best_gain, best_t


def majority_class(records):
    return Counter(r[TARGET] for r in records).most_common(1)[0][0]


def id3(records, cat_features, cont_features, depth=0):
    """
    ID3 mở rộng hỗ trợ cả thuộc tính rời rạc và liên tục.
    Trả về node dict:
      {'type': 'leaf', 'label': ...}
      {'type': 'categorical', 'feature': ..., 'children': {value: node}}
      {'type': 'continuous', 'feature': ..., 'threshold': ...,
       'left': node,  # <= threshold
       'right': node} # > threshold
    """
    indent = "  " * depth

    labels = set(r[TARGET] for r in records)
    if len(labels) == 1:
        label = labels.pop()
        print(f"{indent}→ Leaf: rui_ro={label}")
        return {"type": "leaf", "label": label}

    if not cat_features and not cont_features:
        label = majority_class(records)
        print(f"{indent}→ Leaf (majority): rui_ro={label}")
        return {"type": "leaf", "label": label}

    # Tính gain cho tất cả features còn lại
    gains = {}
    thresholds = {}
    for f in cat_features:
        g, _ = info_gain_categorical(records, f)
        gains[f] = g
        thresholds[f] = None
    for f in cont_features:
        g, t = best_threshold(records, f)
        gains[f] = g if g > 0 else 0
        thresholds[f] = t

    best = max(gains, key=gains.get)
    print(f"{indent}Split on: {best!r}  (Gain = {gains[best]:.4f}"
          + (f", threshold = {thresholds[best]:,.0f}" if thresholds[best] else "") + ")")
    for f in sorted(gains, key=lambda x: -gains[x]):
        thr_str = f" (thresh={thresholds[f]:,.0f})" if thresholds[f] else ""
        print(f"{indent}  {f}: {gains[f]:.4f}{thr_str}")

    if best in cat_features:
        node = {"type": "categorical", "feature": best, "children": {}}
        remaining_cat = [f for f in cat_features if f != best]
        for value in sorted(set(r[best] for r in records)):
            subset = [r for r in records if r[best] == value]
            print(f"{indent}  [{best} = {value}] -> {len(subset)} records")
            node["children"][value] = id3(subset, remaining_cat, cont_features, depth + 1)
        return node
    else:
        t = thresholds[best]
        left  = [r for r in records if r[best] <= t]
        right = [r for r in records if r[best] > t]
        remaining_cont = [f for f in cont_features if f != best]
        print(f"{indent}  [{best} <= {t:,.0f}] -> {len(left)} records")
        left_node  = id3(left,  cat_features, remaining_cont, depth + 1)
        print(f"{indent}  [{best} >  {t:,.0f}] -> {len(right)} records")
        right_node = id3(right, cat_features, remaining_cont, depth + 1)
        return {"type": "continuous", "feature": best, "threshold": t,
                "left": left_node, "right": right_node}


def predict(node, sample):
    if node["type"] == "leaf":
        return node["label"]
    if node["type"] == "categorical":
        value = sample.get(node["feature"])
        child = node["children"].get(value)
        if child is None:
            return None
        return predict(child, sample)
    # continuous
    value = sample.get(node["feature"])
    if value <= node["threshold"]:
        return predict(node["left"], sample)
    return predict(node["right"], sample)


def print_tree(node, indent=0):
    prefix = "  " * indent
    if node["type"] == "leaf":
        print(f"{prefix}→ rui_ro = {node['label']}")
        return
    if node["type"] == "categorical":
        for value, child in node["children"].items():
            print(f"{prefix}[{node['feature']} = {value}]")
            print_tree(child, indent + 1)
    else:
        t = node["threshold"]
        print(f"{prefix}[{node['feature']} <= {t:,.0f}]")
        print_tree(node["left"], indent + 1)
        print(f"{prefix}[{node['feature']} > {t:,.0f}]")
        print_tree(node["right"], indent + 1)


# ===================== MAIN =====================
if __name__ == "__main__":
    print("=" * 60)
    print("    CÂY QUYẾT ĐỊNH ID3 - Rủi ro tín dụng")
    print("=" * 60)
    print(f"\nEntropy gốc: {entropy(data):.4f}")
    print(f"Phân bố nhãn: {Counter(r[TARGET] for r in data)}")

    print("\n--- Quá trình xây dựng cây ---\n")
    tree = id3(data, CATEGORICAL, CONTINUOUS)

    print("\n--- Cây quyết định ID3 ---\n")
    print_tree(tree)

    correct = sum(predict(tree, r) == r[TARGET] for r in data)
    print(f"\nĐộ chính xác trên tập huấn luyện: {correct}/{len(data)} = {correct/len(data)*100:.1f}%")

    # Ví dụ dự đoán
    sample = {"age": 32, "hon_nhan": "Độc thân", "so_huu_bds": "Nhà thuê", "income": 8000000}
    print(f"\nDự đoán cho: {sample}")
    print(f"  → rui_ro = {predict(tree, sample)}")
