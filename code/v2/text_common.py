"""
text_common.py - Shared utilities for the v2 text branch (adapted from v1).

Used by:
    t1_text_pretrain.py   stage-1 fine-tuning on GoEmotions (+ IEMOCAP improvised)
    t2_text_models.py     encoders x (direct / frozen / finetune) x (full / sentence)
    t3_text_llm.py        LLM zero-shot baseline

Contents
  - Canonical label handling for external models and corpora
  - Session loading (freestyle transcripts only; the passage script is constant)
  - Sentence splitting
  - NRC Emotion Lexicon + NRC VAD features, with negated words counted SEPARATELY
  - Hugging Face encoder loading, frozen embedding, fine-tuning loop
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from config import get_device
from labels import EMOTIONS as EMOTION_LABELS
from audio_common import class_weights as _cw, load_evaluation, load_records

N_EMOTIONS = len(EMOTION_LABELS)
DEVICE = get_device(verbose=False)


def compute_class_weights(y, n_classes=None):
    return _cw(np.asarray(y))

# ─────────────────────────────────────────────────────────────────────────────
# Encoders
# ─────────────────────────────────────────────────────────────────────────────
ENCODERS = {
    # current baseline: small general-purpose sentence encoder, no emotion training
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    # emotion-tuned, 7 classes (anger disgust fear joy neutral sadness surprise)
    "emo_distilroberta": "j-hartmann/emotion-english-distilroberta-base",
    # RoBERTa-base trained on GoEmotions (28 fine-grained labels)
    "goemo_roberta": "SamLowe/roberta-base-go_emotions",
}


def resolve_encoder(spec: str) -> tuple[str, str]:
    """Alias or raw HF id / local path -> (alias, model_id)."""
    if spec in ENCODERS:
        return spec, ENCODERS[spec]
    alias = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(spec).name or spec)
    return alias, spec


# ─────────────────────────────────────────────────────────────────────────────
# Label mapping
# ─────────────────────────────────────────────────────────────────────────────
# Names used by external models -> canonical label
NAME_MAP = {
    "anger": "angry", "angry": "angry",
    "disgust": "disgust", "disgusted": "disgust",
    "fear": "fear", "fearful": "fear",
    "joy": "happy", "happiness": "happy", "happy": "happy",
    "neutral": "neutral",
    "sadness": "sad", "sad": "sad",
    "surprise": "surprise", "surprised": "surprise",
}

# Official GoEmotions -> Ekman grouping (plus neutral), collapsed onto our names.
GOEMO_TO_CANON = {}
for canon, fine in {
    "angry": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "happy": ["joy", "amusement", "approval", "excitement", "gratitude", "love",
              "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sad": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}.items():
    for f in fine:
        GOEMO_TO_CANON[f] = canon


def label_to_canon(name: str) -> Optional[str]:
    n = str(name).strip().lower()
    return NAME_MAP.get(n) or GOEMO_TO_CANON.get(n)


def canonical_projection(id2label: dict) -> Optional[np.ndarray]:
    """
    Matrix M (n_model_labels x 7) such that probs_canon = probs_model @ M.
    Works for 7-label emotion heads (identity/permutation) and for GoEmotions'
    28 labels (probabilities summed within each Ekman group).
    Returns None if any label cannot be mapped.
    """
    n = len(id2label)
    M = np.zeros((n, N_EMOTIONS), dtype=np.float32)
    for i in range(n):
        name = id2label.get(i, id2label.get(str(i)))
        c = label_to_canon(name)
        if c is None:
            return None
        M[i, EMOTION_LABELS.index(c)] = 1.0
    if M.sum(axis=0).min() == 0:
        return None          # some canonical class never produced
    return M


# ─────────────────────────────────────────────────────────────────────────────
# In-house sessions
# ─────────────────────────────────────────────────────────────────────────────
def load_sessions() -> list[dict]:
    """
    Evaluation sessions (same order as splits.json) with their FREESTYLE transcript.
    The passage transcript is not used: the script is identical in every session.
    """
    ev, _ = load_evaluation()
    recs = load_records()                       # active evaluation set (own or iemocap)
    out, empty = [], 0
    for s in ev:
        r = recs.get(s["freestyle_clip"], {})
        text = (r.get("transcript") or "").strip()
        if not text:
            empty += 1
            text = "."                        # kept so every session stays aligned with the splits
        out.append({"key": s["key"], "user_id": s["speaker_id"], "text": text, "y7": s["y7"],
                    "asr_logprob": float(r.get("asr_avg_logprob") or -1.0),
                    "asr_nospeech": float(r.get("asr_no_speech_prob") or 0.5)})
    if empty:
        print(f"[WARN] {empty} session(s) have an empty freestyle transcript (kept as '.')")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitting
# ─────────────────────────────────────────────────────────────────────────────
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str, min_words: int = 3, max_words: int = 40,
                    max_sentences: int = 40) -> list[str]:
    """
    Split on sentence punctuation. Very short fragments are merged into the
    previous sentence ("Yeah." carries little on its own). Runs without
    punctuation — Whisper sometimes produces these — are chunked by word count
    so a single 'sentence' can never be the whole transcript.
    """
    parts = [p.strip() for p in _SENT_RE.split(text.strip()) if p.strip()]
    merged: list[str] = []
    for p in parts:
        if merged and len(p.split()) < min_words:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    # a short opening fragment has no previous sentence: merge it forward
    if len(merged) > 1 and len(merged[0].split()) < min_words:
        merged[1] = merged[0] + " " + merged[1]
        merged = merged[1:]

    out: list[str] = []
    for s in merged:
        w = s.split()
        if len(w) <= max_words:
            out.append(s)
        else:
            for i in range(0, len(w), max_words):
                chunk = " ".join(w[i:i + max_words])
                if out and len(chunk.split()) < min_words:
                    out[-1] += " " + chunk
                else:
                    out.append(chunk)
    if not out:
        out = [text.strip()]
    if len(out) > max_sentences:
        idx = np.linspace(0, len(out) - 1, max_sentences).astype(int)
        out = [out[i] for i in idx]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# NRC lexicons
# ─────────────────────────────────────────────────────────────────────────────
EMOLEX_CATS = ["anger", "anticipation", "disgust", "fear", "joy",
               "negative", "positive", "sadness", "surprise", "trust"]

NEGATORS = {"not", "no", "never", "nothing", "nobody", "none", "neither",
            "nor", "cannot", "without", "hardly", "barely"}
NEG_SCOPE = 3
_TOK_RE = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,!?;:]")


class Lexicon:
    """
    NRC Emotion Lexicon (EmoLex) + NRC VAD Lexicon.

    Both are free for research use and must be downloaded by the user from
    saifmohammad.com (they are not redistributed here).

    NEGATION: a word inside the scope of a negator (the next 3 tokens, reset at
    punctuation) is counted in a SEPARATE 'negated' bucket rather than flipped.
    "not happy" is not the same as "sad", so flipping would invent information;
    counting it separately lets the classifier learn what negation means.
    """

    def __init__(self, emolex_path: Optional[Path], vad_path: Optional[Path]):
        self.emolex: dict[str, set] = defaultdict(set)
        self.vad: dict[str, tuple] = {}
        if emolex_path:
            self._load_emolex(Path(emolex_path))
        if vad_path:
            self._load_vad(Path(vad_path))
        if not self.emolex and not self.vad:
            raise SystemExit(
                "[ERROR] No lexicon loaded. Download the NRC Emotion Lexicon and\n"
                "        NRC VAD Lexicon from saifmohammad.com and pass --emolex\n"
                "        and --vad, or run without --lexicon.")

    def _load_emolex(self, p: Path):
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 3:
                    continue
                word, cat, val = parts
                if cat in EMOLEX_CATS and val.strip() == "1":
                    self.emolex[word.lower()].add(cat)
        print(f"[INFO] EmoLex: {len(self.emolex)} words with at least one association")

    def _load_vad(self, p: Path):
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                try:
                    v, a, d = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    continue            # header
                self.vad[parts[0].lower()] = (v, a, d)
        print(f"[INFO] NRC VAD: {len(self.vad)} words")

    @staticmethod
    def feature_names() -> list[str]:
        names = [f"emolex_{c}" for c in EMOLEX_CATS]
        names += [f"emolex_neg_{c}" for c in EMOLEX_CATS]
        names += ["vad_v_mean", "vad_a_mean", "vad_d_mean", "vad_v_std",
                  "vad_coverage", "vad_v_mean_negated", "log_n_words"]
        return names

    def features(self, text: str) -> np.ndarray:
        toks = _TOK_RE.findall(text.lower())
        words = [t for t in toks if t[0].isalpha()]
        n = max(len(words), 1)

        pos = np.zeros(len(EMOLEX_CATS), dtype=np.float32)
        neg = np.zeros(len(EMOLEX_CATS), dtype=np.float32)
        vad_hits, vad_neg_v = [], []

        scope = 0
        for t in toks:
            if not t[0].isalpha():
                scope = 0
                continue
            if t in NEGATORS or t.endswith("n't"):
                scope = NEG_SCOPE
                continue
            negated = scope > 0
            if scope > 0:
                scope -= 1
            for c in self.emolex.get(t, ()):
                (neg if negated else pos)[EMOLEX_CATS.index(c)] += 1
            if t in self.vad:
                if negated:
                    vad_neg_v.append(self.vad[t][0])
                else:
                    vad_hits.append(self.vad[t])

        if vad_hits:
            arr = np.array(vad_hits, dtype=np.float32)
            v_mean, a_mean, d_mean = arr.mean(axis=0)
            v_std = float(arr[:, 0].std())
        else:
            v_mean = a_mean = d_mean = 0.5
            v_std = 0.0
        v_neg = float(np.mean(vad_neg_v)) if vad_neg_v else 0.5

        return np.concatenate([
            pos / n, neg / n,
            np.array([v_mean, a_mean, d_mean, v_std, len(vad_hits) / n,
                      v_neg, math.log1p(len(words))], dtype=np.float32),
        ]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Hugging Face models
# ─────────────────────────────────────────────────────────────────────────────
def load_tokenizer(model_id: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_id)


def _last_linear(model, out_features: int):
    last = None
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == out_features:
            last = m
    return last


def load_classifier(model_id: str):
    """
    Load a sequence classifier with a 7-way head in CANONICAL order.

    If the source model already predicts the same seven emotions (e.g. the
    emotion distilroberta), its final layer is PERMUTED into canonical order
    rather than re-initialised, so its emotion knowledge is kept. Otherwise the
    head is replaced with a fresh 7-way layer.
    """
    from transformers import AutoConfig, AutoModelForSequenceClassification

    id2label_c = {i: l for i, l in enumerate(EMOTION_LABELS)}
    label2id_c = {l: i for i, l in id2label_c.items()}

    cfg = AutoConfig.from_pretrained(model_id)
    orig = getattr(cfg, "id2label", None) or {}
    orig = {int(k): v for k, v in orig.items()}
    M = canonical_projection(orig) if len(orig) == N_EMOTIONS else None
    is_perm = M is not None and np.allclose(M.sum(axis=0), 1) and np.allclose(M.sum(axis=1), 1)

    if is_perm:
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        lin = _last_linear(model, N_EMOTIONS)
        order = [int(np.argmax(M[:, c])) for c in range(N_EMOTIONS)]
        with torch.no_grad():
            lin.weight.copy_(lin.weight[order].clone())
            if lin.bias is not None:
                lin.bias.copy_(lin.bias[order].clone())
        model.config.id2label, model.config.label2id = id2label_c, label2id_c
        head = "kept (permuted to canonical order)"
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=N_EMOTIONS, id2label=id2label_c,
            label2id=label2id_c, ignore_mismatched_sizes=True)
        head = "new 7-way head"
    model.config.problem_type = "single_label_classification"
    return model, head


def free(model=None):
    if model is not None:
        del model
    if DEVICE == "mps":
        torch.mps.empty_cache()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def embed_texts(model_id: str, texts: list[str], max_len: int,
                batch: int = 32) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    FROZEN encoder features: mean-pooled last hidden state.
    If the model has an emotion head that maps onto the seven canonical
    classes, its canonical probabilities are returned as well.
    """
    from transformers import AutoModel, AutoConfig, AutoModelForSequenceClassification
    tok = load_tokenizer(model_id)
    cfg = AutoConfig.from_pretrained(model_id)
    orig = {int(k): v for k, v in (getattr(cfg, "id2label", None) or {}).items()}
    M = canonical_projection(orig) if orig else None

    if M is not None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, output_hidden_states=True).to(DEVICE).eval()
    else:
        model = AutoModel.from_pretrained(model_id).to(DEVICE).eval()

    embs, probs = [], []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt").to(DEVICE)
        out = model(**enc)
        hidden = out.hidden_states[-1] if M is not None else out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        embs.append(pooled.cpu().numpy())
        if M is not None:
            p = F.softmax(out.logits, dim=-1).cpu().numpy()
            if getattr(cfg, "problem_type", "") == "multi_label_classification":
                p = torch.sigmoid(out.logits).cpu().numpy()
                p = p / np.clip(p.sum(1, keepdims=True), 1e-6, None)
            probs.append(p @ M)
    free(model)
    return (np.vstack(embs).astype(np.float32),
            np.vstack(probs).astype(np.float32) if probs else None)


@torch.no_grad()
def direct_probs(model_id: str, texts: list[str], max_len: int,
                 batch: int = 32) -> np.ndarray:
    """Zero-training baseline: the external model's own emotion head, mapped to canonical."""
    _, p = embed_texts(model_id, texts, max_len, batch)
    if p is None:
        raise ValueError(f"{model_id} has no head that maps onto the 7 classes")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
def _batches(n, bs, shuffle, gen=None):
    idx = torch.randperm(n, generator=gen).tolist() if shuffle else list(range(n))
    for i in range(0, n, bs):
        yield idx[i:i + bs]


@torch.no_grad()
def predict_probs(model, tok, texts: list[str], max_len: int, batch: int = 32) -> np.ndarray:
    model.eval()
    out = []
    for b in _batches(len(texts), batch, False):
        enc = tok([texts[j] for j in b], padding=True, truncation=True,
                  max_length=max_len, return_tensors="pt").to(DEVICE)
        out.append(F.softmax(model(**enc).logits, dim=-1).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, N_EMOTIONS), np.float32)


def finetune(model, tok, train_texts: list[str], train_y: np.ndarray,
             val_fn=None, epochs: int = 4, batch: int = 16, max_len: int = 256,
             lr_encoder: float = 2e-5, lr_head: float = 1e-4,
             weight_decay: float = 0.01, patience: int = 2, seed: int = 42,
             verbose: bool = True):
    """
    Fine-tune a sequence classifier on (texts, canonical labels).

    Encoder and head get different learning rates: the encoder moves slowly
    (it already knows language), the head moves fast (it may be fresh).
    Class weights are the same sqrt-inverse-frequency scheme used everywhere
    else in the project.

    val_fn(model) -> score : called after every epoch for early stopping. It
    must evaluate on data carved out of TRAINING (never the test fold).
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model.to(DEVICE)

    head_params, enc_params = [], []
    for n, p in model.named_parameters():
        (head_params if ("classifier" in n or "score" in n) else enc_params).append(p)
    opt = torch.optim.AdamW([
        {"params": enc_params, "lr": lr_encoder},
        {"params": head_params, "lr": lr_head},
    ], weight_decay=weight_decay)

    w = compute_class_weights(np.asarray(train_y), N_EMOTIONS).to(DEVICE)
    y_t = torch.tensor(np.asarray(train_y), dtype=torch.long)

    steps = epochs * math.ceil(len(train_texts) / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(0.1 * steps))) *
        max(0.0, 1 - s / max(1, steps)))

    best, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for b in _batches(len(train_texts), batch, True, gen):
            enc = tok([train_texts[j] for j in b], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(DEVICE)
            logits = model(**enc).logits
            loss = F.cross_entropy(logits, y_t[b].to(DEVICE), weight=w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
            nb += 1
        score = val_fn(model) if val_fn else -tot
        if verbose:
            print(f"      epoch {ep + 1}/{epochs}  loss={tot / max(nb, 1):.4f}  "
                  f"val={score:.4f}")
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def aggregate_by_owner(probs: np.ndarray, owner: np.ndarray, n_sessions: int) -> np.ndarray:
    """Mean unit probabilities per session. Rows for sessions with no units stay zero."""
    out = np.zeros((n_sessions, N_EMOTIONS), dtype=np.float32)
    cnt = np.zeros(n_sessions, dtype=np.float32)
    np.add.at(out, owner, probs)
    np.add.at(cnt, owner, 1.0)
    nz = cnt > 0
    out[nz] /= cnt[nz, None]
    return out
