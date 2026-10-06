"""
cv.py — Outer folds and inner validation splits (nested cross-validation).

Used only by p5_make_splits.py. The branches never create folds themselves;
they read preprocessing/splits.json, so audio, text and fusion always see the
same sessions in the same folds.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold, StratifiedKFold

METHODS = ("kfold", "grouped", "louo")


def outer_splits(y: np.ndarray, groups: np.ndarray, method: str,
                 n_folds: int = 5, seed: int = 42):
    """Yield (train_idx, test_idx, fold_name)."""
    z = np.zeros(len(y))
    if method == "kfold":
        # speakers appear on both sides: upper bound only, never the headline
        for i, (tr, te) in enumerate(StratifiedKFold(n_folds, shuffle=True,
                                                     random_state=seed).split(z, y), 1):
            yield tr, te, f"fold{i}"
    elif method == "grouped":
        for i, (tr, te) in enumerate(StratifiedGroupKFold(n_folds, shuffle=True,
                                                          random_state=seed).split(z, y, groups), 1):
            yield tr, te, f"fold{i}"
    elif method == "louo":
        for tr, te in LeaveOneGroupOut().split(z, y, groups):
            yield tr, te, f"user={groups[te][0]}"
    else:
        raise ValueError(method)


def inner_split(y_tr: np.ndarray, g_tr: np.ndarray, method: str,
                val_frac: float = 0.2, seed: int = 42):
    """
    Validation set carved out of TRAINING, for early stopping and for fitting
    fusion weights. For grouped and louo whole speakers go to validation, so it
    is speaker-disjoint exactly like the outer test fold. Returns positions
    within the training split.
    """
    rng = np.random.RandomState(seed)
    n = len(y_tr)
    if method in ("grouped", "louo"):
        users = np.unique(g_tr)
        k = max(1, int(round(len(users) * val_frac)))
        k = min(k, max(1, len(users) - 2))
        val_users = set(rng.choice(users, size=k, replace=False).tolist())
        mask = np.array([g in val_users for g in g_tr])
    else:
        mask = np.zeros(n, dtype=bool)
        for c in np.unique(y_tr):
            idx = np.where(y_tr == c)[0]
            if len(idx) < 2:
                continue
            k = max(1, int(round(len(idx) * val_frac)))
            mask[rng.choice(idx, size=min(k, len(idx) - 1), replace=False)] = True
    if mask.sum() == 0 or (~mask).sum() < 4:
        perm = rng.permutation(n)
        cut = max(2, int(n * (1 - val_frac)))
        mask = np.zeros(n, dtype=bool)
        mask[perm[cut:]] = True
    return np.where(~mask)[0], np.where(mask)[0]
