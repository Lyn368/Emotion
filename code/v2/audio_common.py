"""
audio_common.py - Pieces shared by every audio-branch script.

  - data access: dataset JSONs, splits.json, audio loading, windowing
  - SessionModel: siamese head used by BOTH the frozen and the fine-tuned runs
        freestyle segments -> layer weights -> attention pooling -> f
        passage segments   -> same weights  -> same pooling       -> p
        session vector = [f, f - p] (+ acoustic features) -> MLP -> 7 emotions
  - training loop with early stopping on the inner validation split
  - saving out-of-fold probabilities for the fusion stage
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score

from config import SAMPLE_RATE, V2_DIR, eval_name, eval_paths, outputs
from labels import EMOTIONS, VALENCE, TO_VALENCE

N7 = len(EMOTIONS)
TO_V = np.array([VALENCE.index(TO_VALENCE[e]) for e in EMOTIONS])

WINDOW_S = 4.0            # segment length fed to the encoders
HOP_S = 2.0               # 50% overlap
MAX_SEGMENTS = 40         # per clip; evenly spaced if the clip is longer


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_records(name: str | None = None) -> dict[str, dict]:
    """Records of a dataset JSON; default = the active evaluation set (see config.set_eval)."""
    d = read_json(outputs()["prep"] / (name or eval_paths()["records"]))
    recs = d["records"] if isinstance(d, dict) and "records" in d else d
    return {r.get("clip_id") or r.get("text_id"): r for r in recs}     # text-only sets use text_id


def abs_path(p: str | None) -> Path | None:
    """JSON paths are relative to code\\v2 so the folder can be moved."""
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else V2_DIR / q


def load_evaluation():
    """Sessions of the active evaluation set and its shared folds (p5 for own, p6 for public)."""
    p = eval_paths()["splits"]
    if not p.exists():
        maker = "p5_make_splits.py" if eval_name() == "own" else f"p6_public_sessions.py --datasets {eval_name()}"
        raise SystemExit(f"[ERROR] {p} not found. Run {maker} first.")
    d = read_json(p)
    return d["evaluation_set"], d["splits"]


def load_wav(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"{path.name}: {sr} Hz, expected {SAMPLE_RATE}. Re-run p2.")
    return y


def windows(n_samples: int, window_s=WINDOW_S, hop_s=HOP_S, max_seg=MAX_SEGMENTS):
    """(start, end) sample spans. Short clips give one segment; long ones are subsampled evenly."""
    w, h = int(window_s * SAMPLE_RATE), int(hop_s * SAMPLE_RATE)
    if n_samples <= w:
        return [(0, n_samples)]
    spans = [(s, s + w) for s in range(0, n_samples - w + 1, h)]
    if n_samples - spans[-1][1] >= SAMPLE_RATE:          # keep a tail of >= 1 s
        spans.append((n_samples - w, n_samples))
    if len(spans) > max_seg:
        idx = np.linspace(0, len(spans) - 1, max_seg).astype(int)
        spans = [spans[i] for i in idx]
    return spans


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SegmentPool(nn.Module):
    """Learned weighted sum over encoder layers, then attention pooling over segments."""

    def __init__(self, n_layers: int, dim: int, proj: int = 256, dropout: float = 0.2):
        super().__init__()
        self.layer_w = nn.Parameter(torch.zeros(n_layers))       # softmax -> equal weights at start
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, proj), nn.GELU(), nn.Dropout(dropout))
        self.att = nn.Linear(proj, 1)

    def forward(self, x, mask):                                   # x (B,S,L,D), mask (B,S)
        x = x.float()                                             # stored as float16 to save memory
        w = torch.softmax(self.layer_w, 0)
        h = self.proj((x * w[None, None, :, None]).sum(2))        # (B,S,proj)
        a = self.att(h).squeeze(-1).float().masked_fill(~mask, -1e4)
        a = torch.softmax(a, 1).to(h.dtype)
        return (a.unsqueeze(-1) * h).sum(1)                       # (B,proj)


class SessionModel(nn.Module):
    """
    Siamese session classifier. The SAME pooling is applied to freestyle and
    passage, so f - p compares the two clips in one learned space: the
    within-session neutral reference.
    """

    def __init__(self, n_layers=1, dim=768, acoustic_dim=0, use_encoder=True,
                 use_delta=True, proj=256, hidden=128, dropout=0.3):
        super().__init__()
        self.use_encoder, self.use_delta = use_encoder, use_delta
        self.use_acoustic = acoustic_dim > 0
        size = 0
        if use_encoder:
            self.pool = SegmentPool(n_layers, dim, proj)
            size += proj * (2 if use_delta else 1)
        if self.use_acoustic:
            self.ac = nn.Sequential(nn.Linear(acoustic_dim, hidden), nn.GELU(), nn.Dropout(dropout))
            size += hidden
        self.head = nn.Sequential(nn.LayerNorm(size), nn.Linear(size, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, N7))

    def forward(self, free=None, free_mask=None, pas=None, pas_mask=None, has_pas=None, ac=None):
        parts = []
        if self.use_encoder:
            f = self.pool(free, free_mask)
            parts.append(f)
            if self.use_delta:
                p = self.pool(pas, pas_mask)
                p = torch.where(has_pas[:, None], p, f)           # no passage -> zero delta
                parts.append(f - p)
        if self.use_acoustic:
            parts.append(self.ac(ac))
        return self.head(torch.cat(parts, -1))


def class_weights(y: np.ndarray) -> torch.Tensor:
    """sqrt inverse frequency, mean 1: helps rare classes without letting them dominate."""
    c = np.bincount(y, minlength=N7).astype(float)
    w = np.where(c > 0, np.sqrt(c.sum() / np.maximum(c, 1)), 0.0)
    return torch.tensor(w / w[w > 0].mean(), dtype=torch.float32)


def macro_f1(y, p) -> float:
    return float(f1_score(y, p, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Out-of-fold bookkeeping (what the fusion stage reads)
# ---------------------------------------------------------------------------
class OOF:
    """
    Collects, for one protocol and seed:
      test probabilities for every session (each session is tested exactly once),
      validation probabilities per fold (used to fit fusion weights without the test).
    """

    def __init__(self, keys: list[str]):
        self.keys = keys
        self.pos = {k: i for i, k in enumerate(keys)}
        self.test = np.full((len(keys), N7), np.nan, dtype=np.float32)
        self.fold_of = np.full(len(keys), -1, dtype=int)
        self.val = []

    def add(self, fold_i, fold_name, test_keys, test_p, val_keys, val_p):
        for k, p in zip(test_keys, test_p):
            self.test[self.pos[k]] = p
            self.fold_of[self.pos[k]] = fold_i
        self.val.append({"fold": fold_name, "keys": list(val_keys),
                         "proba": np.asarray(val_p, float).round(5).tolist()})

    def save(self, path: Path, y7: np.ndarray):
        done = ~np.isnan(self.test).any(1)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"keys": self.keys, "y7": y7.tolist(), "fold_of": self.fold_of.tolist(),
                       "test_proba": np.where(np.isnan(self.test), 0, self.test).round(5).tolist(),
                       "tested": done.tolist(), "val": self.val}, f)
        return done


def summarise(y7, pred) -> dict:
    return {"acc7": float(accuracy_score(y7, pred)), "f1_7": macro_f1(y7, pred),
            "acc3": float(accuracy_score(TO_V[y7], TO_V[pred])), "f1_3": macro_f1(TO_V[y7], TO_V[pred])}
