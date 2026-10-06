"""
labels.py — One place for every label decision.

Preprocessing stores RAW labels ("Neutral,Happiness", "fru", "03"). The
branches call these functions to turn raw labels into model targets, so a
modelling decision (e.g. the dominant-label priority, or whether RAVDESS
'calm' counts as neutral) can be changed here without re-running preprocessing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]  # alphabetical
N_EMOTIONS = len(EMOTIONS)
VALENCE = ["negative", "neutral", "positive"]

# "Surprise" is valence-ambiguous in general; grouped as positive because in
# this corpus it co-occurs with happiness far more often than with negative
# emotions. Stated in the paper as a choice, not a standard.
TO_VALENCE = {"angry": "negative", "disgust": "negative", "fear": "negative",
              "sad": "negative", "neutral": "neutral",
              "happy": "positive", "surprise": "positive"}
IDX7_TO_IDX3 = np.array([VALENCE.index(TO_VALENCE[e]) for e in EMOTIONS])

# Multi-label self-report -> one dominant label. Non-neutral beats neutral:
# neutral is the default answer in self-report, so when someone ticks neutral
# AND something else, the something else is the informative part.
DOMINANT_PRIORITY = ["angry", "fear", "disgust", "sad", "surprise", "happy", "neutral"]

# ── own dataset (self-report strings from the app) ──────────────────────────
OWN_MAP = {"happiness": "happy", "happy": "happy", "sadness": "sad", "sad": "sad",
           "fear": "fear", "fearful": "fear", "surprise": "surprise",
           "surprised": "surprise", "anger": "angry", "angry": "angry",
           "disgust": "disgust", "disgusted": "disgust",
           "neutral": "neutral", "calm": "neutral"}

# ── IEMOCAP major-vote codes. fru/oth/xxx have no slot -> None (dropped) ────
IEMOCAP_MAP = {"ang": "angry", "hap": "happy", "exc": "happy", "sad": "sad",
               "neu": "neutral", "fea": "fear", "dis": "disgust", "sur": "surprise"}

# ── RAVDESS emotion codes ────────────────────────────────────────────────────
RAVDESS_CODES = {"01": "neutral", "02": "calm", "03": "happy", "04": "sad",
                 "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"}
RAVDESS_MAP = {"neutral": "neutral", "calm": "neutral", "happy": "happy", "sad": "sad",
               "angry": "angry", "fearful": "fear", "disgust": "disgust",
               "surprised": "surprise"}

# ── CREMA-D ──────────────────────────────────────────────────────────────────
CREMAD_MAP = {"ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy",
              "NEU": "neutral", "SAD": "sad"}

# ── GoEmotions -> Ekman groups (the authors' own mapping) ───────────────────
GOEMO_TO_CANON = {}
for _c, _fine in {
    "angry": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "happy": ["joy", "amusement", "approval", "excitement", "gratitude", "love",
              "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sad": ["sadness", "disappointment", "embarrassment", "grief", "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"],
}.items():
    for _f in _fine:
        GOEMO_TO_CANON[_f] = _c


def parse_own(raw: str) -> list[str]:
    """'Neutral,Happiness' -> ['neutral', 'happy']"""
    out = [OWN_MAP.get(p.strip().lower()) for p in str(raw).split(",")]
    return [e for e in out if e]


def dominant(emotions: list[str]) -> Optional[str]:
    for p in DOMINANT_PRIORITY:
        if p in emotions:
            return p
    return None


def multihot(emotions: list[str]) -> np.ndarray:
    v = np.zeros(N_EMOTIONS, dtype=np.float32)
    for e in emotions:
        if e in EMOTIONS:
            v[EMOTIONS.index(e)] = 1.0
    return v


def resolve(dataset: str, raw) -> dict:
    """
    Raw label -> {'labels': [...], 'dominant': str|None, 'y7': int|None, 'y3': int|None}.
    Returns y7 = None when the raw label has no place in the seven classes.
    """
    if dataset == "own":
        labs = parse_own(raw)
    elif dataset == "iemocap":
        labs = [IEMOCAP_MAP[raw]] if raw in IEMOCAP_MAP else []
    elif dataset == "ravdess":
        labs = [RAVDESS_MAP[raw]] if raw in RAVDESS_MAP else []
    elif dataset == "cremad":
        labs = [CREMAD_MAP[raw]] if raw in CREMAD_MAP else []
    elif dataset == "goemotions":
        groups = {GOEMO_TO_CANON.get(x) for x in (raw or [])} - {None}
        labs = list(groups) if len(groups) == 1 else []   # mixed groups = ambiguous
    else:
        raise ValueError(dataset)
    dom = dominant(labs)
    y7 = EMOTIONS.index(dom) if dom else None
    return {"labels": labs, "dominant": dom, "y7": y7,
            "y3": int(IDX7_TO_IDX3[y7]) if y7 is not None else None}