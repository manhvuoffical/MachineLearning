"""
Bài tập 1: Cây quyết định CART - Dự đoán buys_computer
CART dùng Gini Impurity thay vì Information Gain (như ID3)
Dataset: 14 mẫu với các thuộc tính age, income, student, credit_rating
"""

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
def gini(records):
    """Tính Gini Impurity của tập records."""
    n = len(records)
    if n == 0:
        return 0
    counts = Counter(r[TARGET] for r in records)
    return 1 - sum((c / n) ** 2 for c in counts.values())


def gini_split(records, feature):
    """
    Tính Gini sau khi split theo feature (multi-way split).
    CART gốc làm binary split, nhưng với categorical data nhiều giá trị
    ta có thể dùng multi-way split để dễ minh họa.
    """
    n = len(records)
    values = set(r[feature] for r in records)
    weighted = sum(
        (len(subset := [r for r in records if r[feature] == v]) / n) * gini(subset)
        for v in values
    )
    return weighted


def gini_gain(records, feature):
    """Gini Gain = Gini cha - Gini sau split (để so sánh, chọn max)."""
    return gini(records) - gini_split(records, feature)


def majority_class(records):
    return Counter(r[TARGET] for r in records).most_common(1)[0][0]


def cart(records, features, depth=0):
    """
    Xây dựng cây quyết định CART đệ quy dùng Gini Impurity.
    """
    indent = "  " * depth

    labels = set(r[TARGET] for r in records)
    if len(labels) == 1:
        label = labels.pop()
        print(f"{indent}→ Leaf: {label}")
        return label

    if not features or gini(records) == 0:
        label = majority_class(records)
        print(f"{indent}→ Leaf (majority): {label}")
        return label

    # Chọn feature có Gini Gain cao nhất (= Gini split thấp nhất)
    gini_scores = {f: gini_split(records, f) for f in features}
    gains       = {f: gini_gain(records, f) for f in features}
    best        = min(gini_scores, key=gini_scores.get)

    print(f"{indent}Split on: {best!r}  (Gini split = {gini_scores[best]:.4f}, Gain = {gains[best]:.4f})")
    for f in sorted(gini_scores, key=gini_scores.get):
        print(f"{indent}  {f}: gini_split={gini_scores[f]:.4f}, gain={gains[f]:.4f}")

    tree = {best: {}}
    remaining = [f for f in features if f != best]

    for value in sorted(set(r[best] for r in records)):
        subset = [r for r in records if r[best] == value]
        print(f"{indent}  [{best} = {value}] -> {len(subset)} records, gini={gini(subset):.4f}")
        tree[best][value] = cart(subset, remaining, depth + 1)

    return tree


def predict(tree, sample):
    if isinstance(tree, str):
        return tree
    feature = next(iter(tree))
    value = sample.get(feature)
    subtree = tree[feature].get(value)
    if subtree is None:
        return "unknown"
    return predict(subtree, sample)


def print_tree(tree, indent=0):
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
    print("       CÂY QUYẾT ĐỊNH CART - buys_computer")
    print("=" * 60)

    print(f"\nGini gốc: {gini(data):.4f}")
    print("\n--- Quá trình xây dựng cây ---\n")
    tree = cart(data, FEATURES)

    print("\n--- Cây quyết định CART ---\n")
    print_tree(tree)

    correct = sum(predict(tree, r) == r[TARGET] for r in data)
    print(f"\nĐộ chính xác trên tập huấn luyện: {correct}/{len(data)} = {correct/len(data)*100:.1f}%")

    sample = {"age": "<=30", "income": "medium", "student": "yes", "credit_rating": "fair"}
    print(f"\nDự đoán cho {sample}:")
    print(f"  → buys_computer = {predict(tree, sample)}")
