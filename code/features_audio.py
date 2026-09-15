"""
features_audio.py — Handcrafted acoustic feature bank (replaces handcrafted.py)
═══════════════════════════════════════════════════════════════════════════════
Implements the feature families your supervisor asked for, organised into NAMED
BLOCKS so 11_feature_selection.py can switch each one on and off and measure
what it is worth. Every feature keeps a human-readable name, so a result can be
reported as "MFCC-3 std" rather than "embedding dimension 431".

BLOCKS
─────────────────────────────────────────────────────────────────────────────
  mfcc      MFCC 1..n_mfcc                     mean/std(/skew/kurt)
  dmfcc     MFCC Δ (first derivative)          mean/std(/skew/kurt)
  ddmfcc    MFCC ΔΔ (second derivative)        mean/std(/skew/kurt)
  logmel    log-mel band energies (n_mels)     mean/std
  spectral  TRUE spectral moments per frame (centroid, spread, skewness,
            kurtosis, slope, decrease, entropy, flux, rolloff, flatness,
            contrast, zcr, rms)                mean/std(/skew/kurt)
  lsf       Line Spectral Frequencies from LPC mean/std
  prosody   F0 statistics, voiced fraction, jitter, shimmer, HNR
  egemaps   openSMILE eGeMAPSv02 (88), optional

TWO KINDS OF "SKEWNESS"
─────────────────────────────────────────────────────────────────────────────
Both are computed, because they are different things and the literature uses
the same word for both:
  * SPECTRAL skewness — the 3rd moment of the energy distribution ACROSS
    FREQUENCY within a frame. Asymmetry of the spectral shape.
  * TEMPORAL skewness — the 3rd moment of a feature's contour OVER TIME.
    Whether a contour has occasional large excursions.
Spectral moments are named `sm_*`; temporal moments are the `_skew` / `_kurt`
suffixes on any contour.

COST
─────────────────────────────────────────────────────────────────────────────
pyin (F0) dominates the runtime, roughly 1-3 s per 10 s clip. `--fast`
downsamples the LSF frame grid and skips higher moments. On ~20k public-corpus
clips, run the extraction once and cache it; do not re-extract per experiment.

    pip install librosa
    pip install opensmile        # optional, for the egemaps block
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np

TARGET_SR = 16_000
N_FFT = 1024
HOP = 256          # 16 ms at 16 kHz
WIN = 400          # 25 ms

DEFAULT_BLOCKS = ("mfcc", "dmfcc", "ddmfcc", "logmel", "spectral", "lsf", "prosody")
ALL_BLOCKS = DEFAULT_BLOCKS + ("egemaps",)

# Prefix -> block. Used by feature selection to group columns.
_PREFIX_BLOCK = [
    ("ddmfcc", "ddmfcc"), ("dmfcc", "dmfcc"), ("mfcc", "mfcc"),
    ("mel", "logmel"),
    ("lsf", "lsf"),
    ("f0", "prosody"), ("voiced", "prosody"), ("jitter", "prosody"),
    ("shimmer", "prosody"), ("hnr", "prosody"),
    ("ege_", "egemaps"),
]

_smile = None


def block_of(key: str) -> str:
    """Map a feature name back to its block. Anything unmatched is 'spectral'."""
    for prefix, block in _PREFIX_BLOCK:
        if key.startswith(prefix):
            return block
    return "spectral"


# ─────────────────────────────────────────────────────────────────────────────
# Contour summarisation
# ─────────────────────────────────────────────────────────────────────────────
def _stats(x: np.ndarray, prefix: str, higher_moments: bool = True) -> dict:
    """Summarise a frame-level contour into scalars (temporal moments)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        out = {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0}
        if higher_moments:
            out.update({f"{prefix}_skew": 0.0, f"{prefix}_kurt": 0.0})
        return out

    mu, sd = float(x.mean()), float(x.std())
    out = {f"{prefix}_mean": mu, f"{prefix}_std": sd}
    if higher_moments:
        if sd < 1e-9:
            out.update({f"{prefix}_skew": 0.0, f"{prefix}_kurt": 0.0})
        else:
            z = (x - mu) / sd
            out[f"{prefix}_skew"] = float((z ** 3).mean())
            out[f"{prefix}_kurt"] = float((z ** 4).mean() - 3.0)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# True spectral moments (across frequency, per frame)
# ─────────────────────────────────────────────────────────────────────────────
def spectral_moments(S: np.ndarray, freqs: np.ndarray) -> dict:
    """
    S     : magnitude spectrogram, (n_bins, n_frames)
    freqs : bin centre frequencies, (n_bins,)

    Returns per-frame contours for centroid / spread / skewness / kurtosis /
    slope / decrease / entropy, which are then summarised over time.
    """
    p = S / np.maximum(S.sum(axis=0, keepdims=True), 1e-12)      # per-frame pmf
    f = freqs[:, None]

    centroid = (f * p).sum(axis=0)
    dev = f - centroid[None, :]
    spread = np.sqrt(np.maximum((dev ** 2 * p).sum(axis=0), 1e-12))
    skew = (dev ** 3 * p).sum(axis=0) / (spread ** 3 + 1e-12)
    kurt = (dev ** 4 * p).sum(axis=0) / (spread ** 4 + 1e-12) - 3.0

    # Spectral slope: least-squares fit of magnitude on frequency, per frame.
    fm = freqs.mean()
    fc = freqs - fm
    denom = float((fc ** 2).sum()) + 1e-12
    slope = (fc[:, None] * (S - S.mean(axis=0, keepdims=True))).sum(axis=0) / denom

    # Spectral decrease: how fast energy falls off above the first bin.
    k = np.arange(1, S.shape[0])[:, None]
    decrease = ((S[1:] - S[0:1]) / k).sum(axis=0) / np.maximum(S[1:].sum(axis=0), 1e-12)

    entropy = -(p * np.log(p + 1e-12)).sum(axis=0) / np.log(S.shape[0])

    return {
        "sm_centroid": centroid, "sm_spread": spread,
        "sm_skewness": skew, "sm_kurtosis": kurt,
        "sm_slope": slope, "sm_decrease": decrease, "sm_entropy": entropy,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Line Spectral Frequencies
# ─────────────────────────────────────────────────────────────────────────────
def lpc_to_lsf(a: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert LPC coefficients [1, a1..ap] to p Line Spectral Frequencies in
    radians, ascending.

    LSFs are the angular positions of the roots of the symmetric and
    antisymmetric extensions of A(z). They are bounded, ordered and far more
    stable to quantisation and averaging than raw LPC coefficients, which is
    exactly why they are used as features rather than the LPCs themselves.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 1 or a.size < 3 or not np.all(np.isfinite(a)):
        return None
    p = a.size - 1

    A = np.concatenate([a, [0.0]])
    B = A[::-1]
    P = A - B          # antisymmetric, root at z = +1
    Q = A + B          # symmetric,     root at z = -1

    try:
        if p % 2 == 0:
            P = np.polydiv(P, np.array([1.0, -1.0]))[0]
            Q = np.polydiv(Q, np.array([1.0, 1.0]))[0]
        else:
            P = np.polydiv(P, np.array([1.0, 0.0, -1.0]))[0]
        rp, rq = np.roots(P), np.roots(Q)
    except Exception:
        return None

    ang = np.concatenate([
        np.angle(rp[np.imag(rp) >= 0]),
        np.angle(rq[np.imag(rq) >= 0]),
    ])
    lsf = np.sort(ang[(ang > 1e-6) & (ang < np.pi - 1e-6)])
    return lsf if lsf.size == p else None


def lsf_features(y: np.ndarray, order: int = 16, frame_step: int = 4,
                 higher_moments: bool = False) -> dict:
    """
    Frame-wise LSFs, summarised over time. `frame_step` subsamples the frame
    grid: LPC on every frame is slow and successive frames are near-identical.
    """
    import librosa

    frames = librosa.util.frame(y, frame_length=WIN, hop_length=HOP)
    if frames.shape[1] < 2:
        return {f"lsf{i + 1}_{s}": 0.0
                for i in range(order) for s in ("mean", "std")}

    window = np.hanning(WIN)[:, None]
    # Fixed-seed dither: LPC on a silent or perfectly periodic frame is
    # singular. Seeding keeps extraction reproducible run to run.
    rs = np.random.RandomState(0)
    collected = []
    for j in range(0, frames.shape[1], max(1, frame_step)):
        seg = frames[:, j] * window[:, 0]
        if np.abs(seg).max() < 1e-5:
            continue
        try:
            a = librosa.lpc(seg + 1e-9 * rs.randn(seg.size), order=order)
        except Exception:
            continue
        lsf = lpc_to_lsf(a)
        if lsf is not None:
            collected.append(lsf)

    out: dict = {}
    if not collected:
        for i in range(order):
            out[f"lsf{i + 1}_mean"] = 0.0
            out[f"lsf{i + 1}_std"] = 0.0
            if higher_moments:
                out[f"lsf{i + 1}_skew"] = 0.0
                out[f"lsf{i + 1}_kurt"] = 0.0
        return out

    M = np.stack(collected)                       # (n_frames, order)
    for i in range(order):
        out.update(_stats(M[:, i], f"lsf{i + 1}", higher_moments))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Voice quality: jitter, shimmer, HNR
# ─────────────────────────────────────────────────────────────────────────────
def _jitter_shimmer_hnr(y: np.ndarray, sr: int, f0: np.ndarray,
                        voiced: np.ndarray) -> dict:
    """
    Frame-level approximations of the Praat measures.

    These are APPROXIMATIONS and must be described as such in the write-up.
    Praat computes jitter and shimmer from individual glottal cycles; here
    jitter comes from the frame-to-frame period contour and shimmer from the
    frame-to-frame RMS contour on voiced frames. HNR uses the normalised
    autocorrelation peak at the pitch lag, which is Praat's method.

    If voice quality turns out to matter, install openSMILE and use the
    eGeMAPS block instead — those are the properly cited implementations.
    """
    out = {"jitter_local": 0.0, "jitter_ppq5": 0.0,
           "shimmer_local": 0.0, "shimmer_db": 0.0,
           "hnr_mean": 0.0, "hnr_std": 0.0}

    f0v = f0[voiced] if voiced.size == f0.size else f0[np.isfinite(f0)]
    f0v = f0v[np.isfinite(f0v) & (f0v > 0)]
    if f0v.size >= 3:
        T = 1.0 / f0v
        d = np.abs(np.diff(T))
        out["jitter_local"] = float(d.mean() / (T.mean() + 1e-12))
        if T.size >= 5:
            k = 5
            smooth = np.convolve(T, np.ones(k) / k, mode="valid")
            out["jitter_ppq5"] = float(
                np.abs(T[k // 2: k // 2 + smooth.size] - smooth).mean()
                / (T.mean() + 1e-12))

    # Shimmer from the RMS contour on voiced frames
    import librosa
    rms = librosa.feature.rms(y=y, frame_length=WIN, hop_length=HOP)[0]
    n = min(rms.size, voiced.size)
    amp = rms[:n][voiced[:n]] if voiced.size else rms
    amp = amp[amp > 1e-8]
    if amp.size >= 3:
        out["shimmer_local"] = float(np.abs(np.diff(amp)).mean() / (amp.mean() + 1e-12))
        db = 20 * np.log10(amp + 1e-12)
        out["shimmer_db"] = float(np.abs(np.diff(db)).mean())

    # HNR: normalised autocorrelation peak within the plausible pitch range
    frames = librosa.util.frame(y, frame_length=WIN, hop_length=HOP)
    lo, hi = int(sr / 500), int(sr / 50)
    hnrs = []
    for j in range(0, frames.shape[1], 2):
        seg = frames[:, j] - frames[:, j].mean()
        e = float(seg @ seg)
        if e < 1e-9:
            continue
        ac = np.correlate(seg, seg, mode="full")[seg.size - 1:]
        if ac.size <= hi or ac[0] <= 0:
            continue
        r = float(ac[lo:hi].max() / ac[0])
        r = min(max(r, 1e-4), 1 - 1e-4)
        hnrs.append(10.0 * np.log10(r / (1.0 - r)))
    if hnrs:
        out["hnr_mean"] = float(np.mean(hnrs))
        out["hnr_std"] = float(np.std(hnrs))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────
def librosa_features(
    wav_path: Path,
    blocks: Iterable[str] = DEFAULT_BLOCKS,
    n_mfcc: int = 20,
    n_mels: int = 40,
    lpc_order: int = 16,
    higher_moments: bool = True,
    fast: bool = False,
) -> dict:
    """Flat dict of named scalars. Blocks not requested are simply absent."""
    import librosa

    blocks = set(blocks)
    if fast:
        higher_moments = False

    y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    if y.size < N_FFT:
        return {}
    y = np.nan_to_num(y.astype(np.float64))

    feats: dict = {}
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, win_length=WIN))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    # ── MFCC, Δ, ΔΔ ──────────────────────────────────────────────────────────
    if {"mfcc", "dmfcc", "ddmfcc"} & blocks:
        mel = librosa.feature.melspectrogram(
            S=S ** 2, sr=sr, n_mels=n_mels, fmin=20, fmax=sr // 2)
        logmel = librosa.power_to_db(mel, ref=np.max)
        mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=n_mfcc)

        if "mfcc" in blocks:
            for i in range(n_mfcc):
                feats.update(_stats(mfcc[i], f"mfcc{i + 1}", higher_moments))

        n_frames = mfcc.shape[1]
        w = min(9, n_frames if n_frames % 2 == 1 else n_frames - 1)
        if w >= 3:
            if "dmfcc" in blocks:
                d1 = librosa.feature.delta(mfcc, width=w)
                for i in range(n_mfcc):
                    feats.update(_stats(d1[i], f"dmfcc{i + 1}", higher_moments))
            if "ddmfcc" in blocks:
                d2 = librosa.feature.delta(mfcc, width=w, order=2)
                for i in range(n_mfcc):
                    feats.update(_stats(d2[i], f"ddmfcc{i + 1}", higher_moments))
        else:
            for name in ("dmfcc", "ddmfcc"):
                if name in blocks:
                    for i in range(n_mfcc):
                        feats.update(_stats(np.zeros(1), f"{name}{i + 1}",
                                            higher_moments))

        if "logmel" in blocks:
            for i in range(logmel.shape[0]):
                feats.update(_stats(logmel[i], f"mel{i + 1}", False))

    elif "logmel" in blocks:
        mel = librosa.feature.melspectrogram(
            S=S ** 2, sr=sr, n_mels=n_mels, fmin=20, fmax=sr // 2)
        logmel = librosa.power_to_db(mel, ref=np.max)
        for i in range(logmel.shape[0]):
            feats.update(_stats(logmel[i], f"mel{i + 1}", False))

    # ── Spectral shape and moments ───────────────────────────────────────────
    if "spectral" in blocks:
        for name, contour in spectral_moments(S, freqs).items():
            feats.update(_stats(contour, name, higher_moments))

        feats.update(_stats(librosa.feature.spectral_rolloff(
            S=S, sr=sr, roll_percent=0.85), "rolloff85", higher_moments))
        feats.update(_stats(librosa.feature.spectral_rolloff(
            S=S, sr=sr, roll_percent=0.95), "rolloff95", higher_moments))
        feats.update(_stats(librosa.feature.spectral_bandwidth(S=S, sr=sr),
                            "bandwidth", higher_moments))
        feats.update(_stats(librosa.feature.spectral_flatness(S=S),
                            "flatness", higher_moments))

        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        for i in range(contrast.shape[0]):
            feats.update(_stats(contrast[i], f"contrast{i + 1}", False))

        flux = np.sqrt(((np.diff(S, axis=1)) ** 2).sum(axis=0))
        feats.update(_stats(flux, "flux", higher_moments))

        feats.update(_stats(librosa.feature.zero_crossing_rate(
            y, frame_length=WIN, hop_length=HOP), "zcr", higher_moments))
        rms = librosa.feature.rms(y=y, frame_length=WIN, hop_length=HOP)[0]
        feats.update(_stats(rms, "rms", higher_moments))
        feats.update(_stats(20 * np.log10(rms + 1e-10), "rmsdb", higher_moments))

    # ── LSF ──────────────────────────────────────────────────────────────────
    if "lsf" in blocks:
        feats.update(lsf_features(y, order=lpc_order,
                                  frame_step=8 if fast else 4,
                                  higher_moments=False))

    # ── Prosody and voice quality ────────────────────────────────────────────
    if "prosody" in blocks:
        try:
            f0, voiced_flag, voiced_prob = librosa.pyin(
                y, fmin=50, fmax=500, sr=sr,
                frame_length=N_FFT, hop_length=HOP)
            voiced_flag = np.nan_to_num(voiced_flag).astype(bool)
            v = f0[np.isfinite(f0)]
            if v.size:
                feats["f0_mean"] = float(v.mean())
                feats["f0_std"] = float(v.std())
                feats["f0_median"] = float(np.median(v))
                feats["f0_p5"] = float(np.percentile(v, 5))
                feats["f0_p95"] = float(np.percentile(v, 95))
                feats["f0_range"] = feats["f0_p95"] - feats["f0_p5"]
                feats["f0_iqr"] = float(np.percentile(v, 75) - np.percentile(v, 25))
                lv = np.log(v + 1e-6)
                feats["f0_log_mean"] = float(lv.mean())
                feats["f0_log_std"] = float(lv.std())
                d = np.diff(v)
                feats["f0_delta_abs_mean"] = float(np.abs(d).mean()) if d.size else 0.0
                feats["f0_rising_frac"] = float((d > 0).mean()) if d.size else 0.0
            else:
                for k in ("f0_mean", "f0_std", "f0_median", "f0_p5", "f0_p95",
                          "f0_range", "f0_iqr", "f0_log_mean", "f0_log_std",
                          "f0_delta_abs_mean", "f0_rising_frac"):
                    feats[k] = 0.0
            feats["voiced_fraction"] = float(voiced_flag.mean()) if voiced_flag.size else 0.0
            feats["voiced_prob_mean"] = float(np.nanmean(voiced_prob)) if voiced_prob.size else 0.0
            feats.update(_jitter_shimmer_hnr(y, sr, f0, voiced_flag))
        except Exception:
            for k in ("f0_mean", "f0_std", "f0_median", "f0_p5", "f0_p95",
                      "f0_range", "f0_iqr", "f0_log_mean", "f0_log_std",
                      "f0_delta_abs_mean", "f0_rising_frac", "voiced_fraction",
                      "voiced_prob_mean", "jitter_local", "jitter_ppq5",
                      "shimmer_local", "shimmer_db", "hnr_mean", "hnr_std"):
                feats[k] = 0.0

    return {k: float(np.nan_to_num(v)) for k, v in feats.items()}


# ─────────────────────────────────────────────────────────────────────────────
# eGeMAPS via openSMILE (optional block)
# ─────────────────────────────────────────────────────────────────────────────
def get_smile():
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
    smile = get_smile()
    if smile == "unavailable":
        return None
    try:
        df = smile.process_file(str(wav_path))
    except Exception:
        return None
    return {f"ege_{name}": float(val) for name, val in df.iloc[0].items()}


# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────
def extract_features(
    wav_path: Path,
    blocks: Iterable[str] = DEFAULT_BLOCKS,
    n_mfcc: int = 20,
    n_mels: int = 40,
    lpc_order: int = 16,
    higher_moments: bool = True,
    fast: bool = False,
) -> dict:
    """
    Returns {"set", "features", "dim", "blocks": {block: [keys]}}.

    `blocks` is recorded in the output so a mixed corpus can never go
    unnoticed, and so the paper can state exactly which families were used.
    """
    blocks = set(blocks)
    feats = librosa_features(wav_path, blocks=blocks, n_mfcc=n_mfcc,
                             n_mels=n_mels, lpc_order=lpc_order,
                             higher_moments=higher_moments, fast=fast)
    if "egemaps" in blocks:
        ege = egemaps_features(wav_path)
        if ege:
            feats.update(ege)

    by_block: dict[str, list[str]] = {}
    for k in feats:
        by_block.setdefault(block_of(k), []).append(k)

    return {
        "set": "librosa_bank_v2" + ("+egemaps" if "egemaps" in by_block else ""),
        "features": feats,
        "dim": len(feats),
        "blocks": {b: sorted(v) for b, v in by_block.items()},
    }


def extract_handcrafted(wav_path: Path, prefer_egemaps: bool = True) -> dict:
    """Back-compatible entry point for 03_feature_extraction.py."""
    blocks = list(DEFAULT_BLOCKS) + (["egemaps"] if prefer_egemaps else [])
    return extract_features(wav_path, blocks=blocks)


def feature_vector(rec: dict, keys: Optional[list[str]] = None
                   ) -> tuple[np.ndarray, list[str]]:
    """
    Turn a stored handcrafted dict into a fixed-order vector.

    `keys` must be identical across every clip so columns line up. 03 writes the
    key list once to the manifest for exactly this reason.
    """
    hc = rec.get("handcrafted", {}).get("features", {})
    if keys is None:
        keys = sorted(hc.keys())
    vec = np.array([float(hc.get(k, 0.0)) for k in keys], dtype=np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0), keys


def keys_for_blocks(keys: list[str], blocks: Iterable[str]) -> list[str]:
    """Subset a key list to the requested blocks, preserving order."""
    wanted = set(blocks)
    return [k for k in keys if block_of(k) in wanted]


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Inspect the feature bank on one clip")
    p.add_argument("wav", type=Path)
    p.add_argument("--blocks", default=",".join(DEFAULT_BLOCKS))
    args = p.parse_args()

    out = extract_features(args.wav, blocks=args.blocks.split(","))
    print(f"set={out['set']}  dim={out['dim']}")
    for b, ks in sorted(out["blocks"].items()):
        print(f"  {b:<10} {len(ks):4d} features   e.g. {ks[:3]}")
