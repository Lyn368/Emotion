"""
08_train_encoder.py
═══════════════════════════════════════════════════════════════════════════════
Train YOUR OWN speech-emotion encoder on the public corpora (RAVDESS +
CREMA-D + IEMOCAP), with a 7-class head and a 3-class valence head, on the
speaker-disjoint TRAIN split only. The 20% holdout defined in splits.json is
never loaded here.

    # 1. cheap sanity run: frozen backbone, only the head trains
    python 08_train_encoder.py --utterances data_public/utterances.jsonl \
        --backbone microsoft/wavlm-base-plus --freeze_backbone \
        --epochs 6 --out encoders/wavlm_bp_frozen

    # 2. the real run: fine-tune the top transformer layers
    python 08_train_encoder.py --utterances data_public/utterances.jsonl \
        --backbone microsoft/wavlm-base-plus --unfreeze_top 6 \
        --epochs 12 --batch 8 --out encoders/wavlm_bp_ft

    # 3. your supervisor's "spectrogram image" suggestion, as a baseline
    python 08_train_encoder.py --utterances data_public/utterances.jsonl \
        --backbone melcnn --epochs 30 --batch 32 --out encoders/melcnn

    # 4. fuse the handcrafted librosa bank into the encoder
    python 08_train_encoder.py ... --hc_features data_public/features.jsonl \
        --hc_blocks mfcc,dmfcc,ddmfcc,spectral,lsf,prosody

ARCHITECTURE
─────────────────────────────────────────────────────────────────────────────
  waveform ─► SSL backbone (all hidden layers)
                 │
                 ├─ learned softmax weights over layers  (SUPERB-style)
                 │  Different layers encode different things: lower layers
                 │  carry speaker and channel, middle layers carry prosody and
                 │  phonetics. Letting the model weight them beats picking the
                 │  last layer, which is the most ASR-specialised one.
                 │
                 ▼
          attentive statistics pooling  ──►  mean ⊕ std  (2×H)
                 │                            (+ handcrafted branch, optional)
                 ▼
          projection ──► EMBEDDING (this is what 09 exports)
                 ├─► head7  : 7 emotions
                 └─► head3  : negative / neutral / positive

WHY WavLM
─────────────────────────────────────────────────────────────────────────────
WavLM's pre-training adds simulated-overlap denoising on top of masked
prediction, which forces it to model *who is speaking and how* rather than
only *what is said*. That is the paralinguistic content SER needs, and it is
why WavLM leads the SUPERB emotion task. It also ties directly to the paper
already in your folder — robust emotion features emerging from speaker
recognition is the same claim from the other direction.

Practical order: wavlm-base-plus frozen (minutes) → wavlm-base-plus with the
top 6 layers unfrozen (hours on MPS) → wavlm-large if you get a CUDA GPU.
Do not start with large on a laptop.

MASKED LOSS ACROSS CORPORA
─────────────────────────────────────────────────────────────────────────────
CREMA-D has no 'surprise'. Training a 7-way softmax on it would teach the
model that CREMA-D audio is evidence against surprise, which is an artefact of
corpus design, not of speech. Logits for classes a sample's corpus does not
contain are therefore masked out before the loss. splits.json records which
labels each corpus actually has.

NO LLM IS INVOLVED HERE. This stage is a discriminative audio encoder; the
only generative model in the whole pipeline is at session fusion (06).
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from common import (
    EMOTION_LABELS, N_EMOTIONS, THREE_CLASS_LABELS, N_THREE, IDX7_TO_IDX3,
    DEVICE, compute_class_weights,
)

TARGET_SR = 16_000


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
def load_utterances(path: Path, split: str | None = "train") -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if split and r.get("split") != split:
                continue
            if r["emotion"] not in EMOTION_LABELS:
                continue
            rows.append(r)
    return rows


def read_wav(path: str, sr: int = TARGET_SR) -> np.ndarray:
    import soundfile as sf
    y, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != sr:
        import librosa
        y = librosa.resample(y.astype(np.float32), orig_sr=file_sr, target_sr=sr)
    return np.nan_to_num(y.astype(np.float32))


class UtteranceDataset(Dataset):
    """
    Fixed-length crops so every batch is rectangular: random crop while
    training (a free augmentation), centre crop at evaluation time.
    """

    def __init__(self, rows: list[dict], crop_s: float, train: bool,
                 hc: dict | None = None, hc_keys: list[str] | None = None,
                 hc_norm: tuple | None = None, allowed: dict | None = None,
                 gain_jitter_db: float = 3.0):
        self.rows = rows
        self.crop = int(crop_s * TARGET_SR)
        self.train = train
        self.hc, self.hc_keys, self.hc_norm = hc, hc_keys, hc_norm
        self.allowed = allowed or {}
        self.gain_jitter_db = gain_jitter_db

    def __len__(self):
        return len(self.rows)

    def _crop(self, y: np.ndarray, rng: random.Random) -> tuple[np.ndarray, int]:
        n = len(y)
        if n >= self.crop:
            start = rng.randint(0, n - self.crop) if self.train else (n - self.crop) // 2
            return y[start:start + self.crop], self.crop
        out = np.zeros(self.crop, dtype=np.float32)
        off = 0 if not self.train else rng.randint(0, self.crop - n)
        out[off:off + n] = y
        return out, n

    def _hc_vec(self, stem: str) -> np.ndarray:
        if not self.hc_keys:
            return np.zeros(0, dtype=np.float32)
        rec = self.hc.get(stem, {}) if self.hc else {}
        v = np.array([float(rec.get(k, 0.0)) for k in self.hc_keys], dtype=np.float32)
        if self.hc_norm is not None:
            mu, sd = self.hc_norm
            v = (v - mu) / sd
        return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def __getitem__(self, i):
        r = self.rows[i]
        rng = random.Random(hash((r["stem"], i, self.train)) & 0xFFFFFFFF)
        try:
            y = read_wav(r["path"])
        except Exception:
            y = np.zeros(self.crop, dtype=np.float32)
        y, n_valid = self._crop(y, rng)

        if self.train and self.gain_jitter_db > 0:
            g = 10 ** (rng.uniform(-self.gain_jitter_db, self.gain_jitter_db) / 20.0)
            y = np.clip(y * g, -1.0, 1.0)

        y7 = EMOTION_LABELS.index(r["emotion"])
        allowed = self.allowed.get(r["corpus"], np.ones(N_EMOTIONS, dtype=bool))
        return {
            "wav": torch.from_numpy(y),
            "n_valid": n_valid,
            "y7": y7,
            "y3": int(IDX7_TO_IDX3[y7]),
            "allowed": torch.from_numpy(allowed.copy()),
            "hc": torch.from_numpy(self._hc_vec(r["stem"])),
            "idx": i,
        }


def collate(batch):
    return {
        "wav": torch.stack([b["wav"] for b in batch]),
        "n_valid": torch.tensor([b["n_valid"] for b in batch]),
        "y7": torch.tensor([b["y7"] for b in batch]),
        "y3": torch.tensor([b["y3"] for b in batch]),
        "allowed": torch.stack([b["allowed"] for b in batch]),
        "hc": torch.stack([b["hc"] for b in batch]),
        "idx": torch.tensor([b["idx"] for b in batch]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class AttentiveStatsPooling(nn.Module):
    """
    Weighted mean and std over time, with the weights predicted from the
    frames themselves. Plain average pooling dilutes short emotional bursts
    across the whole utterance; attention lets the model concentrate on the
    frames that actually carry the affect.
    """

    def __init__(self, dim: int, bottleneck: int = 128):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv1d(dim, bottleneck, 1), nn.Tanh(), nn.Conv1d(bottleneck, dim, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)   mask: (B, T) bool
        xt = x.transpose(1, 2)                       # (B, D, T)
        w = self.att(xt)
        w = w.masked_fill(~mask.unsqueeze(1), -1e4)
        w = torch.softmax(w, dim=2)
        mu = (w * xt).sum(dim=2)
        var = (w * xt * xt).sum(dim=2) - mu ** 2
        return torch.cat([mu, torch.sqrt(var.clamp(min=1e-6))], dim=1)


class MelCNN(nn.Module):
    """
    Log-mel spectrogram + small CNN. This is the "spectrogram image" baseline;
    it needs no pretrained weights and trains in minutes, which makes it the
    right thing to put in the table as the floor the SSL encoder has to beat.
    """

    def __init__(self, n_mels: int = 64, width: int = 64):
        super().__init__()
        import torchaudio
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SR, n_fft=1024, hop_length=256,
            win_length=400, n_mels=n_mels, f_min=20, f_max=TARGET_SR // 2)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.spec_aug = nn.Sequential(
            torchaudio.transforms.FrequencyMasking(freq_mask_param=12),
            torchaudio.transforms.TimeMasking(time_mask_param=30),
        )
        c = width
        self.net = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(c, c * 2, 3, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(c * 2, c * 4, 3, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(c * 4, c * 4, 3, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(),
        )
        self.out_dim = c * 4 * (n_mels // 8)
        self.frame_ratio = 256 * 4        # hop * time pooling factor

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        x = self.db(self.mel(wav))                      # (B, M, T)
        if self.training:
            x = self.spec_aug(x)
        x = (x - x.mean(dim=(1, 2), keepdim=True)) / (x.std(dim=(1, 2), keepdim=True) + 1e-5)
        h = self.net(x.unsqueeze(1))                    # (B, C, M', T')
        b, c, m, t = h.shape
        return h.permute(0, 3, 1, 2).reshape(b, t, c * m)


class EmotionEncoder(nn.Module):
    def __init__(self, backbone: str, emb_dim: int = 256, hc_dim: int = 0,
                 dropout: float = 0.2, freeze_backbone: bool = False,
                 unfreeze_top: int = 0, spec_aug: float = 0.05):
        super().__init__()
        self.backbone_name = backbone
        self.is_ssl = backbone != "melcnn"

        if self.is_ssl:
            from transformers import AutoConfig, AutoModel
            cfg = AutoConfig.from_pretrained(backbone)
            cfg.output_hidden_states = True
            cfg.apply_spec_augment = spec_aug > 0
            cfg.mask_time_prob = spec_aug
            cfg.mask_feature_prob = spec_aug

            # LayerDrop OFF. Two reasons, and the second one is the real one.
            #
            # 1. In recent transformers, a dropped layer also drops its entry
            #    from hidden_states, so the tuple length varies BETWEEN BATCHES
            #    and only while training. That is the 12-vs-13 shape error: the
            #    count looks stable in eval mode and is not stable in training.
            # 2. Even if the shapes lined up, weighted layer pooling and
            #    LayerDrop are incompatible by design. Weight i is supposed to
            #    mean "how much layer i matters". If layers are skipped at
            #    random, weight i lands on whatever layer happens to be in that
            #    position, so the weights are learning against a moving target.
            #
            # SpecAugment stays on, so the model is still regularised.
            for attr in ("layerdrop", "encoder_layerdrop"):
                if getattr(cfg, attr, 0.0):
                    print(f"[INFO] Disabling {attr} "
                          f"(was {getattr(cfg, attr)}) — it is incompatible "
                          f"with weighted layer pooling")
                    setattr(cfg, attr, 0.0)
            self.backbone = AutoModel.from_pretrained(backbone, config=cfg)
            if hasattr(self.backbone, "freeze_feature_encoder"):
                self.backbone.freeze_feature_encoder()
            hidden = cfg.hidden_size
            # How many hidden states the backbone actually returns is measured,
            # not assumed. num_hidden_layers + 1 is the usual answer, but it is
            # not guaranteed across transformers versions, and guessing wrong
            # only surfaces as a shape mismatch mid-training.
            self.layer_w = nn.Parameter(torch.zeros(self._probe_n_hidden_states(cfg)))
            self._set_trainable(freeze_backbone, unfreeze_top, cfg.num_hidden_layers)
        else:
            self.backbone = MelCNN()
            hidden = self.backbone.out_dim
            self.layer_w = None

        self.pool = AttentiveStatsPooling(hidden)
        pooled = hidden * 2

        self.hc_dim = hc_dim
        if hc_dim > 0:
            self.hc_branch = nn.Sequential(
                nn.LayerNorm(hc_dim), nn.Linear(hc_dim, 128), nn.ReLU(),
                nn.Dropout(dropout))
            pooled += 128

        self.proj = nn.Sequential(
            nn.Linear(pooled, emb_dim), nn.BatchNorm1d(emb_dim), nn.ReLU(),
            nn.Dropout(dropout))
        self.head7 = nn.Linear(emb_dim, N_EMOTIONS)
        self.head3 = nn.Linear(emb_dim, N_THREE)
        self.emb_dim = emb_dim

    def _probe_n_hidden_states(self, cfg) -> int:
        """One throwaway forward pass to count the returned hidden states."""
        was_training = self.backbone.training
        n = cfg.num_hidden_layers + 1
        counts = set()
        try:
            with torch.no_grad():
                for training in (False, True):
                    self.backbone.train(training)
                    # Several passes in training mode: anything stochastic in
                    # the layer stack shows up as a varying count.
                    for _ in range(1 if not training else 5):
                        out = self.backbone(torch.zeros(1, TARGET_SR),
                                            output_hidden_states=True)
                        if getattr(out, "hidden_states", None):
                            counts.add(len(out.hidden_states))
            if counts:
                n = max(counts)
        except Exception as exc:
            print(f"[WARN] Could not probe hidden states ({exc}); assuming {n}.")
        finally:
            self.backbone.train(was_training)

        if len(counts) > 1:
            raise RuntimeError(
                f"The backbone returned a varying number of hidden states "
                f"across forward passes: {sorted(counts)}.\nThat means layer "
                f"dropping is still active, which weighted layer pooling "
                f"cannot work with.\nCheck that config.layerdrop and "
                f"config.encoder_layerdrop are both 0.0.")

        print(f"[INFO] Backbone returns {n} hidden states "
              f"(config declares {cfg.num_hidden_layers} layers)")
        return n

    def _set_trainable(self, freeze_backbone: bool, unfreeze_top: int, n_layers: int):
        if not freeze_backbone and unfreeze_top <= 0:
            return                                   # full fine-tune
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_top > 0 and hasattr(self.backbone, "encoder"):
            layers = self.backbone.encoder.layers
            for layer in layers[-unfreeze_top:]:
                for p in layer.parameters():
                    p.requires_grad = True

    def frames(self, wav: torch.Tensor, n_valid: torch.Tensor):
        if self.is_ssl:
            out = self.backbone(wav, output_hidden_states=True)
            hs = torch.stack(out.hidden_states, dim=0)          # (L, B, T, H)
            if hs.shape[0] != self.layer_w.shape[0]:
                raise RuntimeError(
                    f"Backbone returned {hs.shape[0]} hidden states but the "
                    f"layer weights are sized {self.layer_w.shape[0]}.\n"
                    f"If the count changes between batches, LayerDrop is still "
                    f"active: check that config.layerdrop reached 0.0.\n"
                    f"If it is stable but wrong, the checkpoint was trained on "
                    f"a different transformers version — retrain it.")
            w = torch.softmax(self.layer_w, dim=0).view(-1, 1, 1, 1)
            h = (hs * w).sum(dim=0)                             # (B, T, H)
        else:
            h = self.backbone(wav)                              # (B, T, D)

        # Frame-level validity mask, derived from the ratio of valid samples to
        # total samples. Avoids depending on private length-conversion helpers.
        B, T, _ = h.shape
        total = wav.shape[1]
        frac = (n_valid.to(h.device).float() / float(total)).clamp(min=1e-3, max=1.0)
        lens = (frac * T).ceil().long().clamp(min=1, max=T)
        ar = torch.arange(T, device=h.device).unsqueeze(0)
        mask = ar < lens.unsqueeze(1)
        return h, mask

    def embed(self, wav, n_valid, hc=None):
        h, mask = self.frames(wav, n_valid)
        z = self.pool(h, mask)
        if self.hc_dim > 0 and hc is not None and hc.numel():
            z = torch.cat([z, self.hc_branch(hc)], dim=1)
        return self.proj(z)

    def forward(self, wav, n_valid, hc=None):
        e = self.embed(wav, n_valid, hc)
        return self.head7(e), self.head3(e), e


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────
def masked_ce(logits, target, allowed, weight=None, smoothing: float = 0.05):
    """Cross entropy with classes absent from the sample's corpus masked out."""
    logits = logits.masked_fill(~allowed, -1e4)
    return F.cross_entropy(logits, target, weight=weight, label_smoothing=smoothing)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def all_metrics(y_true, y_pred, labels) -> dict:
    ua = []
    for c in np.unique(y_true):
        m = y_true == c
        if m.sum():
            ua.append(float((y_pred[m] == c).mean()))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "ua": float(np.mean(ua)) if ua else 0.0,
        "n": int(len(y_true)),
    }


@torch.no_grad()
def evaluate(model, loader, rows, device) -> dict:
    model.eval()
    P7, P3, Y7, Y3, IDX = [], [], [], [], []
    for b in loader:
        o7, o3, _ = model(b["wav"].to(device), b["n_valid"],
                          b["hc"].to(device) if b["hc"].numel() else None)
        o7 = o7.masked_fill(~b["allowed"].to(device), -1e4)
        P7.append(o7.argmax(1).cpu().numpy())
        P3.append(o3.argmax(1).cpu().numpy())
        Y7.append(b["y7"].numpy())
        Y3.append(b["y3"].numpy())
        IDX.append(b["idx"].numpy())

    y7 = np.concatenate(Y7); p7 = np.concatenate(P7)
    y3 = np.concatenate(Y3); p3 = np.concatenate(P3)
    idx = np.concatenate(IDX)

    out = {"overall7": all_metrics(y7, p7, EMOTION_LABELS),
           "overall3": all_metrics(y3, p3, THREE_CLASS_LABELS),
           "per_corpus": {}}

    corpora = np.array([rows[i]["corpus"] for i in idx])
    for c in sorted(set(corpora)):
        m = corpora == c
        out["per_corpus"][c] = {
            "7class": all_metrics(y7[m], p7[m], EMOTION_LABELS),
            "3class": all_metrics(y3[m], p3[m], THREE_CLASS_LABELS),
        }
    out["_y7"], out["_p7"] = y7, p7
    return out


def print_eval(res: dict, title: str) -> None:
    o7, o3 = res["overall7"], res["overall3"]
    print(f"  {title}")
    print(f"    7-class  acc={o7['accuracy']:.4f}  UA={o7['ua']:.4f}  "
          f"mF1={o7['f1_macro']:.4f}  microF1={o7['f1_micro']:.4f}  "
          f"wF1={o7['f1_weighted']:.4f}")
    print(f"    3-class  acc={o3['accuracy']:.4f}  UA={o3['ua']:.4f}  "
          f"mF1={o3['f1_macro']:.4f}  wF1={o3['f1_weighted']:.4f}")
    for c, m in sorted(res["per_corpus"].items()):
        print(f"      {c:<9} 7cls acc={m['7class']['accuracy']:.4f} "
              f"mF1={m['7class']['f1_macro']:.4f}  |  "
              f"3cls acc={m['3class']['accuracy']:.4f}  (n={m['7class']['n']})")


# ─────────────────────────────────────────────────────────────────────────────
# Speaker-disjoint inner validation split
# ─────────────────────────────────────────────────────────────────────────────
def speaker_split(rows: list[dict], val_frac: float, seed: int):
    by_corpus = defaultdict(list)
    for r in rows:
        by_corpus[r["corpus"]].append(r["speaker"])
    rng = random.Random(seed)
    val_speakers = set()
    for corpus, spks in by_corpus.items():
        uniq = sorted(set(spks))
        k = max(1, int(round(len(uniq) * val_frac)))
        rng.shuffle(uniq)
        val_speakers.update(uniq[:k])
    tr = [r for r in rows if r["speaker"] not in val_speakers]
    va = [r for r in rows if r["speaker"] in val_speakers]
    return tr, va, sorted(val_speakers)


# ─────────────────────────────────────────────────────────────────────────────
# Handcrafted features (optional branch)
# ─────────────────────────────────────────────────────────────────────────────
def load_hc(path: Path, blocks: list[str] | None):
    """Read features.jsonl and return ({stem: {key: val}}, ordered key list)."""
    from features_audio import block_of
    table, keyset = {}, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            feats = rec.get("handcrafted", {}).get("features", {})
            if not feats:
                continue
            stem = rec.get("stem") or Path(rec["file"]).stem
            table[stem] = feats
            if keyset is None:
                keyset = set(feats.keys())
            else:
                keyset &= set(feats.keys())
    if not table:
        raise SystemExit(f"[ERROR] No handcrafted features in {path}")
    keys = sorted(keyset)
    if blocks:
        wanted = set(blocks)
        keys = [k for k in keys if block_of(k) in wanted]
    return table, keys


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Train a custom SER encoder on public corpora")
    p.add_argument("--utterances", type=Path, required=True)
    p.add_argument("--splits", type=Path, default=None,
                   help="splits.json; used for the per-corpus label mask")
    p.add_argument("--out", type=Path, required=True)

    p.add_argument("--backbone", default="microsoft/wavlm-base-plus",
                   help="HF id, or 'melcnn' for the log-mel CNN baseline")
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--unfreeze_top", type=int, default=0,
                   help="Fine-tune only the top N transformer layers")
    p.add_argument("--spec_aug", type=float, default=0.05)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=2, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=1e-3, help="LR for heads/pooling")
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--crop", type=float, default=4.0, help="Training crop, seconds")
    p.add_argument("--eval_crop", type=float, default=6.0)
    p.add_argument("--w3", type=float, default=0.4, help="Weight of the 3-class loss")
    p.add_argument("--sampler", default="class", choices=["none", "class", "corpus_class"])
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="Debug: cap training rows")

    p.add_argument("--hc_features", type=Path, default=None,
                   help="features.jsonl with the handcrafted bank, to fuse in")
    p.add_argument("--hc_blocks", default="mfcc,dmfcc,ddmfcc,spectral,lsf,prosody")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = DEVICE
    print(f"[INFO] Device: {device}   Backbone: {args.backbone}")

    rows = load_utterances(args.utterances, split="train")
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise SystemExit("[ERROR] No training rows. Did 00 write utterances.jsonl?")

    # Per-corpus label mask
    allowed: dict[str, np.ndarray] = {}
    labels_per_corpus = None
    splits_path = args.splits or args.utterances.parent / "splits.json"
    if splits_path.exists():
        labels_per_corpus = json.loads(splits_path.read_text()).get("labels_per_corpus")
    if not labels_per_corpus:
        labels_per_corpus = defaultdict(set)
        for r in rows:
            labels_per_corpus[r["corpus"]].add(r["emotion"])
        labels_per_corpus = {c: sorted(v) for c, v in labels_per_corpus.items()}
    for c, labs in labels_per_corpus.items():
        m = np.zeros(N_EMOTIONS, dtype=bool)
        for l in labs:
            if l in EMOTION_LABELS:
                m[EMOTION_LABELS.index(l)] = True
        allowed[c] = m
        print(f"  label mask {c:<9} {[EMOTION_LABELS[i] for i in np.where(m)[0]]}")

    tr_rows, va_rows, val_spk = speaker_split(rows, args.val_frac, args.seed)
    print(f"[INFO] {len(tr_rows)} train / {len(va_rows)} val utterances "
          f"({len(val_spk)} validation speakers, speaker-disjoint)")
    print("  class counts:", dict(Counter(r['emotion'] for r in tr_rows)))

    # Handcrafted branch
    hc_table, hc_keys, hc_norm = None, None, None
    if args.hc_features:
        hc_table, hc_keys = load_hc(args.hc_features,
                                    [b.strip() for b in args.hc_blocks.split(",") if b.strip()])
        M = np.array([[float(hc_table.get(r["stem"], {}).get(k, 0.0)) for k in hc_keys]
                      for r in tr_rows], dtype=np.float64)
        mu, sd = M.mean(0), M.std(0)
        sd[sd < 1e-8] = 1.0
        hc_norm = (mu.astype(np.float32), sd.astype(np.float32))
        print(f"[INFO] Handcrafted branch: {len(hc_keys)} features "
              f"(normalised on training rows only)")

    ds_tr = UtteranceDataset(tr_rows, args.crop, True, hc_table, hc_keys, hc_norm, allowed)
    ds_va = UtteranceDataset(va_rows, args.eval_crop, False, hc_table, hc_keys, hc_norm, allowed)

    sampler, shuffle = None, True
    if args.sampler != "none":
        if args.sampler == "class":
            key = [r["emotion"] for r in tr_rows]
        else:
            key = [f"{r['corpus']}|{r['emotion']}" for r in tr_rows]
        cnt = Counter(key)
        w = np.array([1.0 / cnt[k] for k in key], dtype=np.float64)
        sampler = WeightedRandomSampler(torch.from_numpy(w), len(w), replacement=True)
        shuffle = False

    dl_tr = DataLoader(ds_tr, batch_size=args.batch, sampler=sampler, shuffle=shuffle,
                       num_workers=args.workers, collate_fn=collate, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=max(2, args.batch), shuffle=False,
                       num_workers=args.workers, collate_fn=collate)

    model = EmotionEncoder(
        args.backbone, emb_dim=args.emb_dim,
        hc_dim=len(hc_keys) if hc_keys else 0,
        freeze_backbone=args.freeze_backbone, unfreeze_top=args.unfreeze_top,
        spec_aug=args.spec_aug).to(device)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Trainable {n_train / 1e6:.1f}M / {n_total / 1e6:.1f}M parameters")

    bb_ids = {id(p) for p in model.backbone.parameters()}
    # melcnn is trained from scratch, so it takes the head learning rate; a
    # pretrained SSL backbone takes the small one or it forgets what it knows.
    bb_lr = args.lr if args.backbone == "melcnn" else args.lr_backbone
    groups = [
        {"params": [p for p in model.parameters()
                    if p.requires_grad and id(p) in bb_ids], "lr": bb_lr},
        {"params": [p for p in model.parameters()
                    if p.requires_grad and id(p) not in bb_ids], "lr": args.lr},
    ]
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    steps = max(1, len(dl_tr) // args.accum) * args.epochs
    warm = max(1, int(0.1 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    y_tr = np.array([EMOTION_LABELS.index(r["emotion"]) for r in tr_rows])
    w7 = compute_class_weights(y_tr, N_EMOTIONS).to(device)
    w3 = compute_class_weights(IDX7_TO_IDX3[y_tr], N_THREE).to(device)

    args.out.mkdir(parents=True, exist_ok=True)
    best, bad, log = -1.0, 0, []

    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot, seen = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)

        for step, b in enumerate(dl_tr, 1):
            wav = b["wav"].to(device)
            hc = b["hc"].to(device) if b["hc"].numel() else None
            o7, o3, _ = model(wav, b["n_valid"], hc)
            loss = (masked_ce(o7, b["y7"].to(device), b["allowed"].to(device), w7)
                    + args.w3 * F.cross_entropy(o3, b["y3"].to(device), weight=w3,
                                                label_smoothing=0.05))
            (loss / args.accum).backward()

            if step % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 5.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)

            tot += float(loss) * len(b["y7"])
            seen += len(b["y7"])
            if step % 50 == 0:
                print(f"    ep{ep} step {step}/{len(dl_tr)}  loss={tot / max(seen, 1):.4f}",
                      flush=True)

        res = evaluate(model, dl_va, va_rows, device)
        score = res["overall7"]["f1_macro"]
        print(f"\n  [epoch {ep}] train loss {tot / max(seen, 1):.4f}  "
              f"({time.time() - t0:.0f}s)")
        print_eval(res, "validation (speaker-disjoint)")

        log.append({"epoch": ep, "train_loss": tot / max(seen, 1),
                    "val": {k: v for k, v in res.items() if not k.startswith("_")}})

        if score > best:
            best, bad = score, 0
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "backbone": args.backbone, "emb_dim": args.emb_dim,
                    "hc_dim": len(hc_keys) if hc_keys else 0,
                    "hc_keys": hc_keys, "crop": args.crop, "eval_crop": args.eval_crop,
                    "hc_mu": hc_norm[0].tolist() if hc_norm else None,
                    "hc_sd": hc_norm[1].tolist() if hc_norm else None,
                    "spec_aug": args.spec_aug,
                },
                "labels": EMOTION_LABELS,
                "labels3": THREE_CLASS_LABELS,
                "val_macro_f1": best,
                "val_speakers": val_spk,
                "epoch": ep,
            }, args.out / "encoder.pt")
            print(f"    saved -> {args.out / 'encoder.pt'}  (val macro F1 {best:.4f})")
        else:
            bad += 1
            print(f"    no improvement ({bad}/{args.patience})")
            if bad >= args.patience:
                print("  early stop")
                break

    (args.out / "training_log.json").write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()},
        "best_val_macro_f1": best, "history": log,
        "val_speakers": val_spk,
    }, indent=2))

    print(f"\n[INFO] Best validation macro F1: {best:.4f}")
    print(f"[INFO] Checkpoint: {args.out / 'encoder.pt'}")
    print("[INFO] Next: 09_extract_embeddings.py to apply it to your own audio,")
    print("             10_benchmark_public.py for the held-out public numbers.")


if __name__ == "__main__":
    main()