"""
Bài tập 1: Cây quyết định ID3 - Dự đoán buys_computer
Dataset: 14 mẫu với các thuộc tính age, income, student, credit_rating
"""

import math
from collections import Counter


# ===================== DATA =====================
data = [
    {"age": "<=30", "income": "high",   "student": "no",  "credit_rating": "fair",      "buys_computer": "no"},
    {"age": "<=30", "income": "high",   "student": "no",  "credit_rating": "excellent",  "buys_computer": "no"},
    {"age": "31..40","income": "high",  "student": "no",  "credit_rating": "fair",       "buys_computer": "yes"},
    {"age": ">40",  "income": "medium", "student": "no",  "credit_rating": "fair",       "buys_computer": "yes"},
    {"age": ">40",  "income": "low",    "student": "yes", "credit_rating": "fair",       "buys_computer": "yes"},
    {"age": ">40",  "income": "low",    "student": "yes", "credit_rating": "excellent",  "buys_computer": "no"},
    {"age": "31..40","income": "low",   "student": "yes", "credit_rating": "excellent",  "buys_computer": "yes"},
    {"age": "<=30", "income": "medium", "student": "no",  "credit_rating": "fair",       "buys_computer": "no"},
    {"age": "<=30", "income": "low",    "student": "yes", "credit_rating": "fair",       "buys_computer": "yes"},
    {"age": ">40",  "income": "medium", "student": "yes", "credit_rating": "fair",       "buys_computer": "yes"},
    {"age": "<=30", "income": "medium", "student": "yes", "credit_rating": "excellent",  "buys_computer": "yes"},
    {"age": "31..40","income": "medium","student": "no",  "credit_rating": "excellent",  "buys_computer": "yes"},
    {"age": "31..40","income": "high",  "student": "yes", "credit_rating": "fair",       "buys_computer": "yes"},
    {"age": ">40",  "income": "medium", "student": "no",  "credit_rating": "excellent",  "buys_computer": "no"},
]

FEATURES = ["age", "income", "student", "credit_rating"]
TARGET = "buys_computer"


# ===================== FUNCTIONS =====================
def entropy(records):
    """Tính entropy của tập records dựa trên nhãn TARGET."""
    n = len(records)
    if n == 0:
        return 0
    counts = Counter(r[TARGET] for r in records)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def info_gain(records, feature):
    """Tính Information Gain của feature trên tập records."""
    n = len(records)
    base_entropy = entropy(records)
    values = set(r[feature] for r in records)
    weighted_entropy = sum(
        (len(subset := [r for r in records if r[feature] == v]) / n) * entropy(subset)
        for v in values
    )
    return base_entropy - weighted_entropy


def majority_class(records):
    """Trả về nhãn đa số trong records."""
    return Counter(r[TARGET] for r in records).most_common(1)[0][0]


def id3(records, features, depth=0):
    """
    Xây dựng cây quyết định ID3 đệ quy.
    Trả về dict biểu diễn cây hoặc string (leaf node).
    """
    indent = "  " * depth

    # Tất cả cùng nhãn -> leaf
    labels = set(r[TARGET] for r in records)
    if len(labels) == 1:
        label = labels.pop()
        print(f"{indent}→ Leaf: {label}")
        return label

    # Hết features -> majority
    if not features:
        label = majority_class(records)
        print(f"{indent}→ Leaf (majority): {label}")
        return label

    # Chọn feature có Info Gain cao nhất
    gains = {f: info_gain(records, f) for f in features}
    best = max(gains, key=gains.get)

    print(f"{indent}Split on: {best!r}  (Gain = {gains[best]:.4f})")
    for f, g in sorted(gains.items(), key=lambda x: -x[1]):
        print(f"{indent}  {f}: {g:.4f}")

    tree = {best: {}}
    remaining = [f for f in features if f != best]

    for value in sorted(set(r[best] for r in records)):
        subset = [r for r in records if r[best] == value]
        print(f"{indent}  [{best} = {value}] -> {len(subset)} records")
        tree[best][value] = id3(subset, remaining, depth + 1)

    return tree


def predict(tree, sample):
    """Dự đoán nhãn cho một sample từ cây đã xây dựng."""
    if isinstance(tree, str):
        return tree
    feature = next(iter(tree))
    value = sample.get(feature)
    subtree = tree[feature].get(value)
    if subtree is None:
        return "unknown"
    return predict(subtree, sample)


def print_tree(tree, indent=0):
    """In cây dưới dạng dễ đọc."""
    prefix = "  " * indent
    if isinstance(tree, str):
        print(f"{prefix}→ {tree}")
        return
    for feature, branches in tree.items():
        for value, subtree in branches.items():
            print(f"{prefix}[{feature} = {value}]")
            print_tree(subtree, indent + 1)


# ===================== MAIN =====================
if __name__ == "__main__":
    print("=" * 60)
    print("       CÂY QUYẾT ĐỊNH ID3 - buys_computer")
    print("=" * 60)

    print(f"\nEntropy gốc: {entropy(data):.4f}")
    print("\n--- Quá trình xây dựng cây ---\n")
    tree = id3(data, FEATURES)

    print("\n--- Cây quyết định ID3 ---\n")
    print_tree(tree)

    # Kiểm tra độ chính xác trên training set
    correct = sum(predict(tree, r) == r[TARGET] for r in data)
    print(f"\nĐộ chính xác trên tập huấn luyện: {correct}/{len(data)} = {correct/len(data)*100:.1f}%")

    # Ví dụ dự đoán
    sample = {"age": "<=30", "income": "medium", "student": "yes", "credit_rating": "fair"}
    print(f"\nDự đoán cho {sample}:")
    print(f"  → buys_computer = {predict(tree, sample)}")
