"""
PHAN 1 - CO TINH LAM MO HINH BI OVERFITTING
===========================================

Cong thuc gay overfitting:
  1. Tap huan luyen rat nho (150 mau cho bai toan 10 lop, 64 chieu)
  2. Mang no-ron rat lon (2 lop an 512 no-ron) -> thua nang luc
  3. Tat hoan toan regularization (alpha = 0)
  4. Huan luyen that lau (300 epoch), khong early stopping

Ket qua mong doi: accuracy tren train len 100%, accuracy tren test dung yen
hoac giam -> khoang cach (gap) lon = mo hinh hoc thuoc long du lieu.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from data import load_data
from utils import plot_curves, report, save_json, train_with_history

EPOCHS = 300


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(n_train=150)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}\n")

    # --- Mo hinh 1: MLP khong le luat hoa ---------------------------------
    model = MLPClassifier(
        hidden_layer_sizes=(512, 512),  # rat nhieu tham so
        alpha=0.0,                      # KHONG co L2
        learning_rate_init=0.01,
        max_iter=1,                     # dung partial_fit nen de = 1
        warm_start=True,
        random_state=42,
    )
    model, hist = train_with_history(
        model, X_train, y_train, X_test, y_test, epochs=EPOCHS
    )

    print("=== PHAN 1: MO HINH BI OVERFITTING ===")
    res_mlp = report("MLP(512,512), alpha=0, 300 epoch", model,
                     X_train, y_train, X_test, y_test)
    plot_curves(hist, "Overfitting: train tang toi 100%, test dung yen",
                "01_overfitting_curves.png")

    n_params = sum(c.size for c in model.coefs_) + sum(b.size for b in model.intercepts_)
    print(f"So tham so cua mo hinh: {n_params:,}")
    print(f"So mau huan luyen      : {len(X_train)}")
    print(f"=> Tham so gap {n_params // len(X_train):,} lan so mau!\n")

    # --- Mo hinh 2: Cay quyet dinh khong gioi han do sau -------------------
    tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
    res_tree = report("DecisionTree khong gioi han do sau", tree,
                      X_train, y_train, X_test, y_test)
    print(f"Do sau cay: {tree.get_depth()}, so la: {tree.get_n_leaves()}")

    save_json(
        {
            "n_params_mlp": int(n_params),
            "n_train": int(len(X_train)),
            "results": [res_mlp, res_tree],
            "history": {k: list(map(float, v)) for k, v in hist.items() if v},
        },
        "01_overfitting.json",
    )


if __name__ == "__main__":
    main()
