"""Ham tien ich: huan luyen theo tung epoch, do do chinh xac, ve bieu do."""

import json
import os

import matplotlib

matplotlib.use("Agg")  # khong can man hinh, chi xuat file anh
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_with_history(model, X_train, y_train, X_test, y_test,
                       epochs=300, X_val=None, y_val=None):
    """Huan luyen tung epoch bang partial_fit de ghi lai duong hoc.

    Tra ve dict lich su: train_acc, test_acc, val_acc, train_loss theo epoch.
    """
    classes = np.unique(y_train)
    hist = {"train_acc": [], "test_acc": [], "val_acc": [], "train_loss": []}

    for _ in range(epochs):
        model.partial_fit(X_train, y_train, classes=classes)
        hist["train_acc"].append(model.score(X_train, y_train))
        hist["test_acc"].append(model.score(X_test, y_test))
        hist["train_loss"].append(model.loss_)
        if X_val is not None:
            hist["val_acc"].append(model.score(X_val, y_val))

    return model, hist


def plot_curves(hist, title, filename, show_val=False):
    """Ve do chinh xac train vs test theo epoch -> nhin thay 'khe ho' overfitting."""
    epochs = range(1, len(hist["train_acc"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))

    ax[0].plot(epochs, hist["train_acc"], label="Train")
    if show_val and hist["val_acc"]:
        ax[0].plot(epochs, hist["val_acc"], label="Validation")
    ax[0].plot(epochs, hist["test_acc"], label="Test")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Do chinh xac")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(epochs, hist["train_loss"], color="crimson")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Training loss")
    ax[1].set_title("Ham mat mat tren tap train")
    ax[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def save_json(obj, filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path


def report(name, model, X_train, y_train, X_test, y_test):
    tr = model.score(X_train, y_train)
    te = model.score(X_test, y_test)
    gap = tr - te
    print(f"{name:<38} train={tr:.4f}  test={te:.4f}  gap={gap:+.4f}")
    return {"name": name, "train_acc": tr, "test_acc": te, "gap": gap}
