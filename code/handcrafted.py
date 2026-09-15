"""
handcrafted.py — Compact handcrafted acoustic features.

WHY THIS EXISTS
─────────────────────────────────────────────────────────────────────────────
The audio branch currently feeds a 2,336-dimensional vector into 393 training
sessions (5.94 dims per sample) and performs at chance. Adding more feature
dimensions to that vector makes the problem worse, not better.

The useful move is the opposite: REPLACE the high-dimensional emotion2vec
blocks with a compact, interpretable set. That is what this module provides.

eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set, Eyben et al.,
IEEE TAFFC 2016) is the field-standard handcrafted set for affective
computing: 88 features covering F0, jitter, shimmer, HNR, formants, MFCCs,
spectral slope and loudness dynamics. Use it rather than hand-rolling a
feature list, because it is standardised, citable, and reviewers already know
what it contains — nobody will ask why you picked these 88 and not others.

    pip install opensmile

If openSMILE is unavailable, `librosa_features()` provides a 74-dimensional
fallback covering roughly the same ground. It is a reasonable substitute but
NOT equivalent: it lacks jitter, shimmer and HNR, which are the voice-quality
measures that matter most for neutral-toned speech, where the emotional signal
sits in phonation quality rather than in pitch excursions.

USAGE
─────────────────────────────────────────────────────────────────────────────
Called from 03_feature_extraction.py per clip. Emits a flat dict of named
scalars, so every feature stays traceable to a name in the final analysis
(unlike embedding dimension 431).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

TARGET_SR = 16_000

_smile = None


# ─────────────────────────────────────────────────────────────────────────────
# eGeMAPS via openSMILE (preferred)
# ─────────────────────────────────────────────────────────────────────────────
def get_smile():
    """Lazy-load the openSMILE eGeMAPSv02 functionals extractor."""
    global _smile
    if _smile is None:
        try:
            import opensmile
            _smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            print("[INFO] openSMILE eGeMAPSv02 loaded (88 features)")
        except ImportError:
            _smile = "unavailable"
    return _smile


def egemaps_features(wav_path: Path) -> Optional[dict]:
    """
    88 eGeMAPS functionals. Returns None if openSMILE is not installed.
    Keys are prefixed 'ege_' so they never collide with the librosa set.
    """
    smile = get_smile()
    if smile == "unavailable":
        return None
    try:
        df = smile.process_file(str(wav_path))
    except Exception:
        return None
    row = df.iloc[0]
    return {f"ege_{name}": float(val) for name, val in row.items()}


# ─────────────────────────────────────────────────────────────────────────────
# librosa fallback (74 features)
# ─────────────────────────────────────────────────────────────────────────────
def _stats(x: np.ndarray, prefix: str, higher_moments: bool = False) -> dict:
    """
    Summarise a frame-level contour into scalars.

    Only mean and std by default. Skewness and kurtosis are available but off
    by default: they double the feature count for a modest information gain,
    and on 393 samples that trade is not obviously worth taking. Turn them on
    as an explicit ablation rather than by default.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        out = {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0}
        if higher_moments:
            out.update({f"{prefix}_skew": 0.0, f"{prefix}_kurt": 0.0})
        return out

    mu = float(x.mean())
    sd = float(x.std())
    out = {f"{prefix}_mean": mu, f"{prefix}_std": sd}

    if higher_moments:
        if sd < 1e-9:
            out.update({f"{prefix}_skew": 0.0, f"{prefix}_kurt": 0.0})
        else:
            z = (x - mu) / sd
            out[f"{prefix}_skew"] = float((z ** 3).mean())
            out[f"{prefix}_kurt"] = float((z ** 4).mean() - 3.0)
    return out


def librosa_features(
    wav_path: Path,
    n_mfcc: int = 13,
    higher_moments: bool = False,
    include_deltas: bool = True,
) -> dict:
    """
    Compact librosa feature set. 74 dims with the defaults.

    Blocks:
      MFCC 1-13                    mean + std          26
      MFCC delta                   mean + std          26
      spectral centroid            mean + std           2
      spectral bandwidth           mean + std           2
      spectral rolloff             mean + std           2
      spectral flatness            mean + std           2
      spectral contrast (7 bands)  mean                 7
      zero-crossing rate           mean + std           2
      RMS energy                   mean + std           2
      F0 (pyin)                    mean, std, range     3
      voiced fraction                                   1
    """
    import librosa

    y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    if y.size < 512:
        return {}

    feats: dict = {}

    # ── MFCCs and their first derivative ─────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    for i in range(n_mfcc):
        feats.update(_stats(mfcc[i], f"mfcc{i + 1}", higher_moments))

    if include_deltas:
        # width must be odd and <= number of frames
        w = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 == 1 else mfcc.shape[1] - 1)
        if w >= 3:
            d = librosa.feature.delta(mfcc, width=w)
            for i in range(n_mfcc):
                feats.update(_stats(d[i], f"dmfcc{i + 1}", higher_moments))

    # ── Spectral shape ───────────────────────────────────────────────────────
    feats.update(_stats(librosa.feature.spectral_centroid(y=y, sr=sr),
                        "centroid", higher_moments))
    feats.update(_stats(librosa.feature.spectral_bandwidth(y=y, sr=sr),
                        "bandwidth", higher_moments))
    feats.update(_stats(librosa.feature.spectral_rolloff(y=y, sr=sr),
                        "rolloff", higher_moments))
    feats.update(_stats(librosa.feature.spectral_flatness(y=y),
                        "flatness", higher_moments))

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(contrast.shape[0]):
        feats[f"contrast{i + 1}_mean"] = float(np.nanmean(contrast[i]))

    feats.update(_stats(librosa.feature.zero_crossing_rate(y),
                        "zcr", higher_moments))
    feats.update(_stats(librosa.feature.rms(y=y), "rms", higher_moments))

    # ── F0 contour ───────────────────────────────────────────────────────────
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=550, sr=sr, frame_length=1024)
        voiced = f0[np.isfinite(f0)]
        if voiced.size > 0:
            feats["f0_mean"] = float(voiced.mean())
            feats["f0_std"] = float(voiced.std())
            feats["f0_range"] = float(np.percentile(voiced, 95)
                                      - np.percentile(voiced, 5))
        else:
            feats.update({"f0_mean": 0.0, "f0_std": 0.0, "f0_range": 0.0})
        feats["voiced_fraction"] = float(np.mean(voiced_flag)) if voiced_flag.size else 0.0
    except Exception:
        feats.update({"f0_mean": 0.0, "f0_std": 0.0,
                      "f0_range": 0.0, "voiced_fraction": 0.0})

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Entry point used by 03_feature_extraction.py
# ─────────────────────────────────────────────────────────────────────────────
def extract_handcrafted(wav_path: Path, prefer_egemaps: bool = True) -> dict:
    """
    Return {"set": "egemaps"|"librosa", "features": {...}, "dim": int}.

    eGeMAPS is preferred; librosa is the fallback. The set name is recorded so
    the paper can state which was used, and so a mixed corpus can never go
    unnoticed.
    """
    if prefer_egemaps:
        ege = egemaps_features(wav_path)
        if ege:
            return {"set": "egemaps", "features": ege, "dim": len(ege)}
    try:
        lib = librosa_features(wav_path)
        return {"set": "librosa", "features": lib, "dim": len(lib)}
    except ImportError:
        return {"set": "none", "features": {}, "dim": 0}
    except Exception:
        return {"set": "none", "features": {}, "dim": 0}


def feature_vector(rec: dict, keys: Optional[list[str]] = None) -> tuple[np.ndarray, list[str]]:
    """
    Turn a stored handcrafted dict into a fixed-order vector.

    `keys` must be supplied (and identical) across every clip so columns line
    up. 03 writes the key list once to the manifest for exactly this reason.
    """
    hc = rec.get("handcrafted", {}).get("features", {})
    if keys is None:
        keys = sorted(hc.keys())
    vec = np.array([float(hc.get(k, 0.0)) for k in keys], dtype=np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0), keys
