"""
common.py — Shared core for the audio emotion recognition pipeline.

Replaces the duplicated logic that previously lived across six near-identical
scripts (sc.py, sc3.py, sc_kfold.py, sc3_kfold.py, sc3_groupedkfold.py, ...).

Everything that is not modality-specific lives here:
  - Label definitions and 7-class <-> 3-class mapping
  - Data loading (video_index.json, features.jsonl)
  - Cross-validation splitters (kfold / grouped / louo)
  - The multi-task MLP (7-class head + 3-class head + multi-label head)
  - Training loop and evaluation / reporting

Import from the modality scripts:
    from common import (EMOTION_LABELS, THREE_CLASS_LABELS, load_video_index,
                        load_features, make_splits, MultiTaskMLP, run_cv)
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import (
    StratifiedKFold,
    StratifiedGroupKFold,
    LeaveOneGroupOut,
)

warnings.filterwarnings("ignore")

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# ─────────────────────────────────────────────────────────────────────────────
# Labels
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
N_EMOTIONS = len(EMOTION_LABELS)

THREE_CLASS_LABELS = ["negative", "neutral", "positive"]
N_THREE = len(THREE_CLASS_LABELS)

# Priority for collapsing a multi-label self-report into one dominant label.
# Non-neutral wins over neutral: "Neutral,Happiness" -> "happy".
DOMINANT_PRIORITY = ["angry", "fear", "disgust", "sad", "surprise", "happy", "neutral"]

LABEL_MAP = {
    "happiness": "happy",   "happy": "happy",
    "sadness": "sad",       "sad": "sad",
    "fear": "fear",         "fearful": "fear",
    "surprise": "surprise", "surprised": "surprise",
    "anger": "angry",       "angry": "angry",
    "disgust": "disgust",   "disgusted": "disgust",
    "neutral": "neutral",   "calm": "neutral",
}

# Valence grouping. NOTE: "surprise" is valence-ambiguous in affective science.
# It is grouped as positive here because in this corpus it co-occurs with
# happiness far more often than with negative emotions. Document this choice
# in the report -- it is a defensible but non-universal decision.
EMOTION_TO_3CLASS = {
    "angry": "negative", "disgust": "negative",
    "fear": "negative",  "sad": "negative",
    "neutral": "neutral",
    "happy": "positive", "surprise": "positive",
}

# ─────────────────────────────────────────────────────────────────────────────
# Training hyperparameters (small dataset ~250-300 sessions -> keep model small)
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_DIMS = [256, 128, 64]
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-3
EPOCHS = 200
BATCH_SIZE = 32
PATIENCE = 30
MULTILABEL_THR = 0.35

# Loss weights for the multi-task heads.
W_7CLASS = 1.0
W_3CLASS = 0.4      # auxiliary: helps minority classes without dominating
W_MULTI = 0.3

MINORITY_BOOST = 1.3   # extra weight on classes below median frequency


# ─────────────────────────────────────────────────────────────────────────────
# Label utilities
# ─────────────────────────────────────────────────────────────────────────────
def normalise_emotion(raw: str) -> Optional[str]:
    return LABEL_MAP.get(str(raw).strip().lower())


def parse_reported_emotions(reported: str) -> list[str]:
    """'Neutral,Happiness' -> ['neutral', 'happy']"""
    parts = [p.strip() for p in str(reported).split(",") if p.strip()]
    out = [normalise_emotion(p) for p in parts]
    return [e for e in out if e is not None]


def dominant_emotion(emotions: list[str]) -> str:
    if not emotions:
        return "neutral"
    for prio in DOMINANT_PRIORITY:
        if prio in emotions:
            return prio
    return emotions[0]


def unweighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    UA / UAR: mean per-class recall. Reported alongside WA (= accuracy) in most
    SER papers, so compute it if you want numbers comparable to the literature.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    recalls = [float((y_pred[y_true == c] == c).mean())
               for c in np.unique(y_true) if (y_true == c).sum()]
    return float(np.mean(recalls)) if recalls else 0.0


def emotions_to_multilabel(emotions: list[str]) -> np.ndarray:
    vec = np.zeros(N_EMOTIONS, dtype=np.float32)
    for e in emotions:
        if e in EMOTION_LABELS:
            vec[EMOTION_LABELS.index(e)] = 1.0
    return vec


def emotion7_to_3class(label_7: str) -> str:
    return EMOTION_TO_3CLASS.get(label_7, "neutral")


def idx7_to_idx3(idx_7: int) -> int:
    return THREE_CLASS_LABELS.index(emotion7_to_3class(EMOTION_LABELS[idx_7]))


IDX7_TO_IDX3 = np.array([idx7_to_idx3(i) for i in range(N_EMOTIONS)], dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_video_index(path: str | Path) -> dict:
    """
    Returns {(user_id, session_id, video_type): {...}}
    """
    with open(path) as f:
        entries = json.load(f)

    index = {}
    for e in entries:
        key = (e["user_id"], e["session_id"], e.get("video_type", "").lower())
        reported = parse_reported_emotions(e.get("reported_emotion", ""))
        index[key] = {
            "reported_emotions": reported,
            "dominant": dominant_emotion(reported),
            "video_id": e.get("video_id", ""),
            "local_path": e.get("local_path", ""),
            "timestamp": e.get("timestamp", 0),
        }
    return index


def load_features(jsonl_path: str | Path) -> dict:
    """
    Load the output of 03_feature_extraction.py.
    Returns {stem: record}
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise SystemExit(
            f"[ERROR] Features not found: {path}\n"
            f"        Run 03_feature_extraction.py first."
        )
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[Path(rec["file"]).stem] = rec
    return out


def resolve_stem(info: dict, features: dict) -> dict:
    """Match a video_index entry to its feature record, trying several key forms."""
    vid = info.get("video_id", "")
    candidates = [
        vid.replace("-webm", "").replace(".webm", ""),
        vid,
        vid.replace("-", "_"),
    ]
    lp = info.get("local_path", "")
    if lp:
        candidates.append(Path(lp).stem)
    for key in candidates:
        if key and key in features:
            return features[key]
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation splitters
# ─────────────────────────────────────────────────────────────────────────────
CV_METHODS = {
    "kfold": (
        "Standard StratifiedKFold (NOT grouped by user)",
        "Sessions from the same user appear in both train and test. "
        "Inflates metrics via speaker-identity leakage. Include only as an "
        "upper bound to quantify the leakage gap.",
    ),
    "grouped": (
        "StratifiedGroupKFold (grouped by user_id)",
        "No user appears in both train and test; class balance preserved as "
        "far as the grouping allows. Multiple users per test fold, so more "
        "stable than LOUO while still leakage-free.",
    ),
    "louo": (
        "Leave-One-User-Out",
        "One held-out user per fold. Most conservative and the closest match "
        "to deployment on a new person. High variance with only 9 users.",
    ),
}


def make_splits(
    y: np.ndarray,
    groups: np.ndarray,
    method: str,
    n_folds: int = 5,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray, str]]:
    """
    Yield (train_idx, test_idx, fold_name) for the requested CV method.
    """
    if method == "kfold":
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for i, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
            yield tr, te, f"fold{i}"

    elif method == "grouped":
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for i, (tr, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups), 1):
            held = sorted(set(groups[te]))
            yield tr, te, f"fold{i}({len(held)}u)"

    elif method == "louo":
        logo = LeaveOneGroupOut()
        for tr, te in logo.split(np.zeros(len(y)), y, groups):
            yield tr, te, f"user={groups[te][0]}"

    else:
        raise ValueError(f"Unknown CV method: {method!r}. Use kfold|grouped|louo.")


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class MultiTaskMLP(nn.Module):
    """
    Shared trunk + three heads:
      - head7    : 7-class dominant emotion       (primary)
      - head3    : 3-class valence                (auxiliary, also reported)
      - headmul  : 7-dim multi-label sigmoid      (auxiliary)

    The 3-class head is trained jointly rather than derived post-hoc, so you
    can report BOTH the dedicated-head result and the post-hoc mapped result
    and compare them.
    """

    def __init__(self, in_dim: int, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head7 = nn.Linear(prev, N_EMOTIONS)
        self.head3 = nn.Linear(prev, N_THREE)
        self.headmul = nn.Linear(prev, N_EMOTIONS)

    def forward(self, x):
        z = self.trunk(x)
        return self.head7(z), self.head3(z), self.headmul(z)


# ─────────────────────────────────────────────────────────────────────────────
# Class weighting
# ─────────────────────────────────────────────────────────────────────────────
def compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """
    sqrt inverse-frequency with a modest boost for below-median classes.

    Deliberately conservative. Focal loss and aggressive oversampling were
    tested on this dataset and destabilised training -- with ~250 sessions and
    a heavy skew toward 'happy', hard rebalancing overfits the minority classes
    instead of learning them.
    """
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = np.sqrt(counts.sum() / (n_classes * counts))

    median = np.median(counts)
    w[counts < median] *= MINORITY_BOOST

    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Train one fold
# ─────────────────────────────────────────────────────────────────────────────
def make_inner_split(
    y_tr: np.ndarray,
    groups_tr: np.ndarray,
    method: str,
    val_frac: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carve an inner VALIDATION set out of the training split, for early stopping.

    For grouped and louo, the split is by SPEAKER: whole speakers go to inner
    validation, so the validation set is speaker-disjoint from inner training
    exactly as the outer test fold is. Anything less and early stopping would
    select checkpoints that generalise within-speaker but not across speakers,
    which is the very thing those CV schemes exist to measure.

    Returns (inner_train_idx, inner_val_idx) as positions within the outer
    training split.
    """
    rng = np.random.RandomState(seed)
    n = len(y_tr)

    if method in ("grouped", "louo"):
        users = np.unique(groups_tr)
        n_val = max(1, int(round(len(users) * val_frac)))
        if len(users) - n_val < 2:
            n_val = max(1, len(users) - 2)
        val_users = set(rng.choice(users, size=n_val, replace=False).tolist())
        val_mask = np.array([g in val_users for g in groups_tr])
    else:
        # Standard K-Fold makes no speaker-disjointness claim, so a stratified
        # random split is the matching inner protocol.
        val_mask = np.zeros(n, dtype=bool)
        for c in np.unique(y_tr):
            idx_c = np.where(y_tr == c)[0]
            k = max(1, int(round(len(idx_c) * val_frac)))
            val_mask[rng.choice(idx_c, size=min(k, len(idx_c) - 1) or 1,
                                replace=False)] = True

    inner_val = np.where(val_mask)[0]
    inner_tr = np.where(~val_mask)[0]

    # Degenerate guards: never leave inner training without at least 2 classes.
    if len(inner_val) == 0 or len(inner_tr) < 4 or len(np.unique(y_tr[inner_tr])) < 2:
        cut = max(2, int(n * (1 - val_frac)))
        perm = rng.permutation(n)
        inner_tr, inner_val = perm[:cut], perm[cut:]

    return inner_tr, inner_val


def train_fold_explicit(
    X_tr: np.ndarray, y7_tr: np.ndarray, ym_tr: np.ndarray,
    X_val: np.ndarray, y7_val: np.ndarray, ym_val: np.ndarray,
    device: str = DEVICE,
    epochs: int = EPOCHS,
    verbose: bool = False,
    seed: int = 42,
) -> tuple[MultiTaskMLP, dict]:
    """
    Fit on (X_tr, y7_tr) and early-stop on an EXPLICITLY SUPPLIED validation
    set. The caller is responsible for making that validation set
    speaker-disjoint from training; run_cv does this via make_inner_split.

    The outer test fold is never seen during training or model selection.
    This is nested cross-validation and it is not optional for publication.
    An earlier version of this code early-stopped on the test fold, which
    selects the best checkpoint *for the test data*. That inflates results and
    is a methodological error a reviewer will reject -- not a limitation that
    can be declared away in the text.

    Expect the honest numbers to be LOWER than the leaky ones. That is the
    point of fixing it.
    """
    # Fold-local standardisation: statistics from the training split only.
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    X_tr_n = (X_tr - mu) / sd
    X_val_n = (X_val - mu) / sd

    y3_tr = IDX7_TO_IDX3[y7_tr]

    Xtr = torch.tensor(X_tr_n, dtype=torch.float32, device=device)
    Xval = torch.tensor(X_val_n, dtype=torch.float32, device=device)
    y7tr = torch.tensor(y7_tr, dtype=torch.long, device=device)
    y3tr = torch.tensor(y3_tr, dtype=torch.long, device=device)
    ymtr = torch.tensor(ym_tr, dtype=torch.float32, device=device)
    y7val_np = np.asarray(y7_val)

    torch.manual_seed(seed)
    model = MultiTaskMLP(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    w7 = compute_class_weights(y7_tr, N_EMOTIONS).to(device)
    w3 = compute_class_weights(y3_tr, N_THREE).to(device)

    n = len(Xtr)
    best_f1, best_state, patience, best_epoch = -1.0, None, 0, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            if len(idx) < 2:      # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            o7, o3, om = model(Xtr[idx])
            loss = (
                W_7CLASS * F.cross_entropy(o7, y7tr[idx], weight=w7)
                + W_3CLASS * F.cross_entropy(o3, y3tr[idx], weight=w3)
                + W_MULTI * F.binary_cross_entropy_with_logits(om, ymtr[idx])
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # ── Early stopping on the INNER VALIDATION split ─────────────────────
        model.eval()
        with torch.no_grad():
            o7, _, _ = model(Xval)
            pred = o7.argmax(dim=1).cpu().numpy()
        f1 = f1_score(y7val_np, pred, average="macro", zero_division=0)

        if f1 > best_f1:
            best_f1, best_epoch = f1, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

        if verbose and epoch % 25 == 0:
            print(f"      epoch {epoch:3d}  inner-val macroF1={f1:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        "mu": mu, "sd": sd,
        "inner_val_macro_f1": best_f1,      # NOT a test metric
        "best_epoch": best_epoch,
        "n_inner_train": int(len(y7_tr)),
        "n_inner_val": int(len(y7val_np)),
    }


def train_fold(X_tr, y7_tr, ym_tr, X_te, y7_te, ym_te, device=DEVICE,
               groups_tr=None, method='louo', seed=42, **kw):
    """
    Back-compatible wrapper: carves its own inner validation split from the
    training data and ignores the test arguments entirely for model
    selection. The test set is passed only so existing call sites keep their
    signature; it is NOT used for early stopping.
    """
    if groups_tr is None:
        groups_tr = np.arange(len(y7_tr))
    in_tr, in_val = make_inner_split(y7_tr, groups_tr, method, seed=seed)
    return train_fold_explicit(
        X_tr[in_tr], y7_tr[in_tr], ym_tr[in_tr],
        X_tr[in_val], y7_tr[in_val], ym_tr[in_val],
        device=device, seed=seed, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate one fold
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_fold(
    model: MultiTaskMLP,
    X_te: np.ndarray, y7_te: np.ndarray, ym_te: np.ndarray,
    norm: dict, device: str = DEVICE,
) -> dict:
    model.eval()
    Xte = torch.tensor((X_te - norm["mu"]) / norm["sd"], dtype=torch.float32, device=device)

    o7, o3, om = model(Xte)
    p7 = F.softmax(o7, dim=1).cpu().numpy()
    p3 = F.softmax(o3, dim=1).cpu().numpy()
    pm = torch.sigmoid(om).cpu().numpy()

    pred7 = p7.argmax(axis=1)
    pred3_head = p3.argmax(axis=1)
    pred3_post = IDX7_TO_IDX3[pred7]        # post-hoc mapping from 7-class
    true3 = IDX7_TO_IDX3[y7_te]

    # "Flexible" accuracy: correct if the prediction is any emotion the user
    # actually reported for that session, not only the dominant one.
    flexible = float(np.mean([ym_te[i, pred7[i]] > 0 for i in range(len(pred7))]))

    return {
        "n": int(len(y7_te)),
        "y7_true": y7_te.tolist(),
        "y7_pred": pred7.tolist(),
        "y3_true": true3.tolist(),
        "y3_pred_head": pred3_head.tolist(),
        "y3_pred_post": pred3_post.tolist(),
        "proba7": p7.tolist(),
        "proba3": p3.tolist(),
        "proba_multi": pm.tolist(),
        "acc7": float(accuracy_score(y7_te, pred7)),
        "f1_macro7": float(f1_score(y7_te, pred7, average="macro", zero_division=0)),
        "f1_micro7": float(f1_score(y7_te, pred7, average="micro", zero_division=0)),
        "f1_weighted7": float(f1_score(y7_te, pred7, average="weighted", zero_division=0)),
        "ua7": unweighted_accuracy(y7_te, pred7),
        "acc3_head": float(accuracy_score(true3, pred3_head)),
        "f1_macro3_head": float(f1_score(true3, pred3_head, average="macro", zero_division=0)),
        "acc3_post": float(accuracy_score(true3, pred3_post)),
        "f1_macro3_post": float(f1_score(true3, pred3_post, average="macro", zero_division=0)),
        "flexible_acc": flexible,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full CV run
# ─────────────────────────────────────────────────────────────────────────────
def run_cv(
    X: np.ndarray,
    y7: np.ndarray,
    ym: np.ndarray,
    groups: np.ndarray,
    method: str,
    n_folds: int = 5,
    seed: int = 42,
    tag: str = "model",
    device: str = DEVICE,
    augment_fn=None,
) -> dict:
    """
    Run cross-validation and pool predictions across folds.

    Pooled metrics (computed over the concatenation of all test folds) are the
    headline numbers -- with 9 users, per-fold averages are dominated by the
    folds that happen to contain a rare class.

    augment_fn : optional callable(train_idx) -> (X_extra, y7_extra, ym_extra)
        Called once per fold with the TRAINING indices only. Whatever it
        returns is appended to that fold's training set. Test folds are never
        passed to it, so augmentation cannot leak. Return None to skip.
    """
    desc, caveat = CV_METHODS[method]
    print(f"\n{'=' * 74}")
    print(f"  {tag}  |  CV: {desc}")
    print(f"  {caveat}")
    print(f"{'=' * 74}")

    fold_results, per_fold_summary = [], []

    for tr, te, name in make_splits(y7, groups, method, n_folds, seed):
        if len(np.unique(y7[tr])) < 2:
            print(f"  [SKIP] {name}: fewer than 2 classes in train")
            continue

        # Inner split FIRST, on real sessions only. Augmented copies are then
        # appended to inner-training. Synthetic rows must never reach the
        # validation set: early stopping would be selecting checkpoints that
        # fit the corruption process rather than real speech.
        in_tr_rel, in_val_rel = make_inner_split(
            y7[tr], groups[tr], method, seed=seed)
        real_tr_idx = tr[in_tr_rel]

        X_tr, y_tr, ym_tr = X[real_tr_idx], y7[real_tr_idx], ym[real_tr_idx]

        if augment_fn is not None:
            extra = augment_fn(real_tr_idx)
            if extra is not None:
                Xe, ye, yme = extra
                if len(Xe):
                    X_tr = np.vstack([X_tr, Xe])
                    y_tr = np.concatenate([y_tr, ye])
                    ym_tr = np.vstack([ym_tr, yme])

        model, norm = train_fold_explicit(
            X_tr, y_tr, ym_tr,
            X[tr[in_val_rel]], y7[tr[in_val_rel]], ym[tr[in_val_rel]],
            device=device, seed=seed)
        res = evaluate_fold(model, X[te], y7[te], ym[te], norm, device=device)
        res["fold"] = name
        res["groups"] = groups[te]
        fold_results.append(res)

        print(
            f"  {name:<16} n={res['n']:<4} "
            f"acc7={res['acc7']:.3f}  mF1_7={res['f1_macro7']:.3f}  "
            f"acc3={res['acc3_head']:.3f}  mF1_3={res['f1_macro3_head']:.3f}  "
            f"flex={res['flexible_acc']:.3f}"
        )
        per_fold_summary.append({
            "fold": name, "n": res["n"],
            "acc7": res["acc7"], "f1_macro7": res["f1_macro7"],
            "acc3_head": res["acc3_head"], "f1_macro3_head": res["f1_macro3_head"],
        })

    if not fold_results:
        raise SystemExit("[ERROR] No usable folds.")

    # ── Pool across folds ────────────────────────────────────────────────────
    y7_true = np.concatenate([r["y7_true"] for r in fold_results])
    y7_pred = np.concatenate([r["y7_pred"] for r in fold_results])
    y3_true = np.concatenate([r["y3_true"] for r in fold_results])
    y3_head = np.concatenate([r["y3_pred_head"] for r in fold_results])
    y3_post = np.concatenate([r["y3_pred_post"] for r in fold_results])
    flex = float(np.mean([r["flexible_acc"] for r in fold_results]))

    pooled = {
        "acc7": float(accuracy_score(y7_true, y7_pred)),
        "f1_macro7": float(f1_score(y7_true, y7_pred, average="macro", zero_division=0)),
        "f1_micro7": float(f1_score(y7_true, y7_pred, average="micro", zero_division=0)),
        "f1_weighted7": float(f1_score(y7_true, y7_pred, average="weighted", zero_division=0)),
        "ua7": unweighted_accuracy(y7_true, y7_pred),
        "ua3_head": unweighted_accuracy(y3_true, y3_head),
        "acc3_head": float(accuracy_score(y3_true, y3_head)),
        "f1_macro3_head": float(f1_score(y3_true, y3_head, average="macro", zero_division=0)),
        "acc3_post": float(accuracy_score(y3_true, y3_post)),
        "f1_macro3_post": float(f1_score(y3_true, y3_post, average="macro", zero_division=0)),
        "flexible_acc": flex,
    }

    # Speaker id per pooled prediction, for cluster bootstrap.
    pooled_groups = np.concatenate([r["groups"] for r in fold_results])

    from stats import print_stats_block, cluster_bootstrap_ci
    stats_block = print_stats_block(
        y7_true, y7_pred, pooled_groups,
        title=f"7-class, speaker-clustered uncertainty ({tag} / {method})",
        seed=seed)
    ci3 = cluster_bootstrap_ci(y3_true, y3_head, pooled_groups,
                               metric="macro_f1", seed=seed)
    print(f"  3-class macro F1 (head): {ci3['point']:.3f} "
          f"[{ci3['ci_low']:.3f}, {ci3['ci_high']:.3f}]")

    print(f"\n  {'-' * 70}")
    print(f"  POOLED ({len(y7_true)} sessions)")
    print(f"  {'-' * 70}")
    print(f"  7-class   accuracy      {pooled['acc7']:.4f}")
    print(f"  7-class   macro F1      {pooled['f1_macro7']:.4f}")
    print(f"  7-class   micro F1      {pooled['f1_micro7']:.4f}   (equals accuracy "
          f"in a single-label task)")
    print(f"  7-class   weighted F1   {pooled['f1_weighted7']:.4f}")
    print(f"  7-class   UA / UAR      {pooled['ua7']:.4f}   (mean per-class recall)")
    print(f"  7-class   flexible acc  {pooled['flexible_acc']:.4f}  (any reported emotion)")
    print()
    print(f"  3-class   accuracy      {pooled['acc3_head']:.4f}   (dedicated head)")
    print(f"  3-class   macro F1      {pooled['f1_macro3_head']:.4f}   (dedicated head)")
    print(f"  3-class   accuracy      {pooled['acc3_post']:.4f}   (post-hoc from 7-class)")
    print(f"  3-class   macro F1      {pooled['f1_macro3_post']:.4f}   (post-hoc from 7-class)")

    delta = pooled["f1_macro3_head"] - pooled["f1_macro3_post"]
    print(f"\n  Head vs post-hoc macro F1 gap: {delta:+.4f}")
    print("  Positive gap => the dedicated valence head learns something the")
    print("  7-class head does not. Near zero => post-hoc mapping is sufficient.")

    print(f"\n  7-class report:")
    print(classification_report(
        y7_true, y7_pred, labels=list(range(N_EMOTIONS)),
        target_names=EMOTION_LABELS, zero_division=0, digits=3))

    print(f"  3-class report (dedicated head):")
    print(classification_report(
        y3_true, y3_head, labels=list(range(N_THREE)),
        target_names=THREE_CLASS_LABELS, zero_division=0, digits=3))

    return {
        "tag": tag,
        "cv_method": method,
        "stats": {k: v for k, v in stats_block.items()},
        "stats_3class_macro_f1": ci3,
        "nested_cv": True,
        "pooled_groups": pooled_groups.tolist(),
        "cv_description": desc,
        "cv_caveat": caveat,
        "n_folds": len(fold_results),
        "n_sessions": int(len(y7_true)),
        "feature_dim": int(X.shape[1]),
        "pooled": pooled,
        "per_fold": per_fold_summary,
        "fold_predictions": [
            {
                "fold": r["fold"],
                "y7_true": r["y7_true"], "y7_pred": r["y7_pred"],
                "groups": list(map(str, r["groups"])),
                "proba7": r["proba7"], "proba3": r["proba3"],
            }
            for r in fold_results
        ],
    }


def save_results(results: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results -> {out_path}")
