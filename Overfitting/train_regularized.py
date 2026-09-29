"""
PHAN 2 - AP DUNG CAC KY THUAT CHONG OVERFITTING
===============================================

Lan luot ap dung 7 ky thuat da hoc, moi ky thuat so sanh voi mo hinh goc:

  (1) Giam do phuc tap mo hinh      - it no-ron hon
  (2) L2 regularization (weight decay) - phat cac trong so lon
  (3) Early stopping                 - dung khi validation khong cai thien nua
  (4) Them du lieu huan luyen        - nhieu du lieu = kho hoc thuoc long
  (5) Data augmentation              - tao them mau bien the tu du lieu cu
  (6) Cross-validation + Grid search  - chon sieu tham so khach quan
  (7) Ensemble (Bagging / Random Forest) - trung binh nhieu mo hinh
  (+) Pruning cho Decision Tree      - cat tia cay bang ccp_alpha
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, validation_curve
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from data import augment, load_data, load_full_pool
from utils import RESULTS_DIR, plot_curves, report, save_json, train_with_history

import os

EPOCHS = 300
SEED = 42
results = []


def baseline(X_train, y_train, X_test, y_test):
    """Mo hinh goc bi overfitting (de lam moc so sanh)."""
    m = MLPClassifier(hidden_layer_sizes=(512, 512), alpha=0.0,
                      learning_rate_init=0.01, max_iter=1,
                      warm_start=True, random_state=SEED)
    m, _ = train_with_history(m, X_train, y_train, X_test, y_test, epochs=EPOCHS)
    return report("[GOC] MLP(512,512) khong regularization", m,
                  X_train, y_train, X_test, y_test)


def technique_1_simpler(X_train, y_train, X_test, y_test):
    """(1) Giam do phuc tap: 512x512 -> 32 no-ron."""
    m = MLPClassifier(hidden_layer_sizes=(32,), alpha=0.0,
                      learning_rate_init=0.01, max_iter=1,
                      warm_start=True, random_state=SEED)
    m, _ = train_with_history(m, X_train, y_train, X_test, y_test, epochs=EPOCHS)
    return report("(1) Giam do phuc tap: MLP(32)", m,
                  X_train, y_train, X_test, y_test)


def technique_2_l2(X_train, y_train, X_val, y_val, X_test, y_test):
    """(2) L2 regularization: them alpha * ||W||^2 vao ham mat mat."""
    # Ve validation curve de thay alpha anh huong the nao
    alphas = np.logspace(-4, 2, 13)
    train_scores, val_scores = validation_curve(
        MLPClassifier(hidden_layer_sizes=(512, 512), learning_rate_init=0.01,
                      max_iter=300, random_state=SEED),
        X_train, y_train, param_name="alpha", param_range=alphas,
        cv=3, n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogx(alphas, train_scores.mean(axis=1), "o-", label="Train")
    ax.semilogx(alphas, val_scores.mean(axis=1), "o-", label="Validation (CV)")
    ax.set_xlabel("alpha (cuong do L2)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation curve: chon alpha toi uu")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "02_validation_curve_alpha.png"), dpi=130)
    plt.close(fig)

    best_alpha = float(alphas[int(np.argmax(val_scores.mean(axis=1)))])
    print(f"    -> alpha tot nhat theo CV: {best_alpha:.4g}")

    m = MLPClassifier(hidden_layer_sizes=(512, 512), alpha=best_alpha,
                      learning_rate_init=0.01, max_iter=1,
                      warm_start=True, random_state=SEED)
    m, hist = train_with_history(m, X_train, y_train, X_test, y_test,
                                 epochs=EPOCHS, X_val=X_val, y_val=y_val)
    plot_curves(hist, f"Sau khi them L2 (alpha={best_alpha:.4g})",
                "03_l2_curves.png", show_val=True)
    return report(f"(2) L2 regularization alpha={best_alpha:.4g}", m,
                  X_train, y_train, X_test, y_test)


def technique_3_early_stopping(X_train, y_train, X_test, y_test):
    """(3) Early stopping: tach 20% lam validation, dung khi het cai thien."""
    m = MLPClassifier(hidden_layer_sizes=(512, 512), alpha=0.0,
                      learning_rate_init=0.01, max_iter=300,
                      early_stopping=True, validation_fraction=0.2,
                      n_iter_no_change=10, random_state=SEED)
    m.fit(X_train, y_train)
    print(f"    -> dung o epoch {m.n_iter_} (thay vi {EPOCHS})")
    return report("(3) Early stopping", m, X_train, y_train, X_test, y_test)


def technique_4_more_data(X_test, y_test):
    """(4) Them du lieu: 150 mau -> toan bo 1257 mau."""
    X_pool, y_pool, X_test_f, y_test_f = load_full_pool()
    m = MLPClassifier(hidden_layer_sizes=(512, 512), alpha=0.0,
                      learning_rate_init=0.01, max_iter=1,
                      warm_start=True, random_state=SEED)
    m, _ = train_with_history(m, X_pool, y_pool, X_test_f, y_test_f, epochs=EPOCHS)
    print(f"    -> so mau train: {len(X_pool)}")
    return report("(4) Them du lieu (1257 mau)", m,
                  X_pool, y_pool, X_test_f, y_test_f)


def technique_5_augment(X_train, y_train, X_test, y_test):
    """(5) Data augmentation: nhan ban + them nhieu."""
    Xa, ya = augment(X_train, y_train, n_copies=9, noise_std=1.0)
    m = MLPClassifier(hidden_layer_sizes=(512, 512), alpha=0.0,
                      learning_rate_init=0.01, max_iter=1,
                      warm_start=True, random_state=SEED)
    m, _ = train_with_history(m, Xa, ya, X_test, y_test, epochs=EPOCHS)
    print(f"    -> {len(X_train)} mau goc -> {len(Xa)} mau sau augmentation")
    return report("(5) Data augmentation", m, Xa, ya, X_test, y_test)


def technique_6_gridsearch(X_train, y_train, X_test, y_test):
    """(6) Cross-validation + Grid search: chon kien truc & alpha cung luc."""
    grid = {
        "hidden_layer_sizes": [(32,), (64,), (128, 64)],
        "alpha": [0.01, 0.1, 1.0, 10.0],
    }
    gs = GridSearchCV(
        MLPClassifier(learning_rate_init=0.01, max_iter=400, random_state=SEED),
        grid, cv=5, n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    print(f"    -> cau hinh tot nhat: {gs.best_params_} (CV={gs.best_score_:.4f})")
    return report("(6) GridSearchCV (5-fold)", gs.best_estimator_,
                  X_train, y_train, X_test, y_test)


def technique_7_ensemble(X_train, y_train, X_test, y_test):
    """(7) Ensemble: Random Forest thay cho 1 cay quyet dinh don le."""
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    return report("(7) Ensemble: RandomForest(300)", rf,
                  X_train, y_train, X_test, y_test)


def technique_8_pruning(X_train, y_train, X_test, y_test):
    """(+) Pruning cay quyet dinh bang cost-complexity (ccp_alpha)."""
    base = DecisionTreeClassifier(random_state=SEED)
    path = base.cost_complexity_pruning_path(X_train, y_train)
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=SEED),
        {"ccp_alpha": path.ccp_alphas[path.ccp_alphas >= 0][:30]},
        cv=5, n_jobs=-1,
    ).fit(X_train, y_train)
    tree = gs.best_estimator_
    print(f"    -> ccp_alpha={tree.ccp_alpha:.5f}, do sau={tree.get_depth()}, "
          f"so la={tree.get_n_leaves()}")
    return report("(+) DecisionTree sau pruning", tree,
                  X_train, y_train, X_test, y_test)


def technique_9_combined():
    """(*) KET HOP: nhieu du lieu + kien truc vua phai + L2 + early stopping."""
    X_pool, y_pool, X_test_f, y_test_f = load_full_pool()
    m = MLPClassifier(
        hidden_layer_sizes=(128, 64),   # vua du, khong thua nang luc
        alpha=0.3162,                   # L2 chon tu validation curve
        learning_rate_init=0.01,
        max_iter=400,
        early_stopping=True,            # dung khi validation het cai thien
        validation_fraction=0.15,
        n_iter_no_change=15,
        random_state=SEED,
    )
    m.fit(X_pool, y_pool)
    print(f"    -> dung o epoch {m.n_iter_}")
    return report("(*) KET HOP tat ca ky thuat", m,
                  X_pool, y_pool, X_test_f, y_test_f)


def plot_summary(res):
    names = [r["name"] for r in res]
    train = [r["train_acc"] for r in res]
    test = [r["test_acc"] for r in res]
    gaps = [r["gap"] for r in res]

    y = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    ax[0].barh(y - 0.2, train, 0.4, label="Train")
    ax[0].barh(y + 0.2, test, 0.4, label="Test")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(names, fontsize=8)
    ax[0].invert_yaxis()
    ax[0].set_xlim(0, 1.05)
    ax[0].set_xlabel("Accuracy")
    ax[0].set_title("Train vs Test")
    ax[0].legend()
    ax[0].grid(axis="x", alpha=0.3)

    colors = ["crimson" if g > 0.10 else "seagreen" for g in gaps]
    ax[1].barh(y, gaps, color=colors)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([])
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Gap = train_acc - test_acc (cang nho cang tot)")
    ax[1].set_title("Muc do overfitting")
    ax[1].grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "04_summary_comparison.png"), dpi=130)
    plt.close(fig)


def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(n_train=150)
    print("=== PHAN 2: CAC KY THUAT CHONG OVERFITTING ===\n")

    results.append(baseline(X_train, y_train, X_test, y_test))
    results.append(technique_1_simpler(X_train, y_train, X_test, y_test))
    results.append(technique_2_l2(X_train, y_train, X_val, y_val, X_test, y_test))
    results.append(technique_3_early_stopping(X_train, y_train, X_test, y_test))
    results.append(technique_4_more_data(X_test, y_test))
    results.append(technique_5_augment(X_train, y_train, X_test, y_test))
    results.append(technique_6_gridsearch(X_train, y_train, X_test, y_test))
    results.append(technique_7_ensemble(X_train, y_train, X_test, y_test))
    results.append(technique_8_pruning(X_train, y_train, X_test, y_test))
    results.append(technique_9_combined())

    plot_summary(results)
    save_json(results, "02_techniques.json")

    best = max(results[1:], key=lambda r: r["test_acc"])
    print(f"\n>> Tot nhat tren tap test: {best['name']} "
          f"({best['test_acc']:.4f}, gap {best['gap']:+.4f})")


if __name__ == "__main__":
    main()
