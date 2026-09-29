"""
Tai va chuan bi du lieu.

Bo du lieu: sklearn `digits` (1797 anh chu so viet tay 8x8 = 64 features, 10 lop).
Du lieu nay di kem thu vien scikit-learn nen khong can tai ve tu internet.

Y tuong quan trong: de mo hinh DE BI OVERFITTING, ta co tinh chi lay mot phan
rat nho lam tap huan luyen (mac dinh 150 mau). It du lieu + mo hinh phuc tap
= cong thuc kinh dien cua overfitting.
"""

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def load_data(n_train=150, n_val=150, scale=True, random_state=RANDOM_STATE):
    """Tra ve (X_train, y_train, X_val, y_val, X_test, y_test).

    n_train: so mau huan luyen. Cang nho cang de overfit.
    n_val  : tap validation dung cho early stopping / chon sieu tham so.
    """
    digits = load_digits()
    X, y = digits.data, digits.target

    # Tach tap test truoc (giu nguyen cho moi thi nghiem de so sanh cong bang)
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )

    # Lay n_train mau tu pool lam tap train, n_val mau lam tap validation
    X_train, X_rest, y_train, y_rest = train_test_split(
        X_pool, y_pool, train_size=n_train, random_state=random_state, stratify=y_pool
    )
    X_val, _, y_val, _ = train_test_split(
        X_rest, y_rest, train_size=n_val, random_state=random_state, stratify=y_rest
    )

    if scale:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_full_pool(scale=True, random_state=RANDOM_STATE):
    """Tra ve toan bo tap train lon (dung cho ky thuat 'them du lieu')."""
    digits = load_digits()
    X, y = digits.data, digits.target
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    if scale:
        scaler = StandardScaler().fit(X_pool)
        X_pool = scaler.transform(X_pool)
        X_test = scaler.transform(X_test)
    return X_pool, y_pool, X_test, y_test


def augment(X, y, n_copies=3, noise_std=0.35, random_state=RANDOM_STATE):
    """Data augmentation don gian: nhan ban du lieu va them nhieu Gaussian.

    Voi anh thuc te ta se dung xoay/dich/lat; o day dung nhieu de minh hoa
    y tuong: lam giau du lieu -> mo hinh kho hoc thuoc long tung mau.
    """
    rng = np.random.default_rng(random_state)
    Xs, ys = [X], [y]
    for _ in range(n_copies):
        Xs.append(X + rng.normal(0, noise_std, size=X.shape))
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)
