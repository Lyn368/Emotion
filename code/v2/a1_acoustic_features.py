"""
a1_acoustic_features.py - Continuous acoustic features for every clip.

Computed on the MEASUREMENT audio (no denoising, no normalisation), because both
alter pitch, jitter, shimmer and loudness. Nothing is discretised.

Per clip (~100 values):
  eGeMAPS v02 functionals (88)   pitch, jitter, shimmer, HNR, formants, MFCC,
                                 spectral slope, loudness ... (openSMILE)
  F0 and loudness skew/kurtosis  shape of the pitch and loudness contours
  pauses (from the VAD times)    speech ratio, pauses per minute, mean / max /
                                 std pause length, mean speech-run length
  speaking rate                  words per second of speech (Whisper transcript)

Output: preprocessing/features/acoustic_<dataset>.json  {clip_id: {name: value}}

    python a1_acoustic_features.py                 # in-house data
    python a1_acoustic_features.py --dataset iemocap
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.stats import kurtosis, skew

from audio_common import abs_path, load_records
from config import ensure_dirs


def pause_features(segments, duration_s) -> dict:
    """Pause statistics from the VAD speech segments [[start, end], ...] in seconds."""
    segs = sorted((float(a), float(b)) for a, b in (segments or []) if b > a)
    speech = sum(b - a for a, b in segs)
    gaps = [b2 - a1 for (_, a1), (b2, _) in zip(segs[:-1], segs[1:]) if b2 > a1]
    gaps = [g for g in gaps if g >= 0.15]                      # shorter gaps are not pauses
    minutes = max(duration_s, 1e-6) / 60
    return {
        "pause_speech_ratio": speech / max(duration_s, 1e-6),
        "pause_per_min": len(gaps) / minutes,
        "pause_mean_s": float(np.mean(gaps)) if gaps else 0.0,
        "pause_max_s": float(np.max(gaps)) if gaps else 0.0,
        "pause_std_s": float(np.std(gaps)) if gaps else 0.0,
        "speech_run_mean_s": speech / len(segs) if segs else 0.0,
        "speech_total_s": speech,
    }


def contour_moments(lld) -> dict:
    """Skewness and kurtosis of the F0 (voiced frames only) and loudness contours."""
    out = {}
    cols = {c.lower(): c for c in lld.columns}
    f0c = next((cols[c] for c in cols if c.startswith("f0semitone")), None)
    loc = next((cols[c] for c in cols if c.startswith("loudness")), None)
    for name, col, voiced in (("f0", f0c, True), ("loudness", loc, False)):
        x = lld[col].to_numpy(float) if col else np.array([])
        if voiced:
            x = x[x > 0]
        ok = x.size >= 10 and np.std(x) > 1e-8
        out[f"{name}_skew"] = float(skew(x)) if ok else 0.0
        out[f"{name}_kurt"] = float(kurtosis(x)) if ok else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description="Continuous acoustic features")
    ap.add_argument("--dataset", default="own", help="own | iemocap | ravdess | cremad")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import opensmile
    func = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                           feature_level=opensmile.FeatureLevel.Functionals)
    lld = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                          feature_level=opensmile.FeatureLevel.LowLevelDescriptors)

    name = {"own": "own_dataset.json"}.get(args.dataset, f"{args.dataset}.json")
    recs = list(load_records(name).values())
    if args.limit:
        recs = recs[: args.limit]
    out_dir = ensure_dirs()["prep"] / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"acoustic_{args.dataset}.json"

    feats, errors, t0 = {}, 0, time.time()
    for i, r in enumerate(recs, 1):
        wav = abs_path(r.get("measure_path"))
        if wav is None or not wav.exists():
            errors += 1
            continue
        try:
            f = {k: float(v) for k, v in func.process_file(str(wav)).iloc[0].items()}
            f.update(contour_moments(lld.process_file(str(wav))))
        except Exception as exc:
            print(f"  [ERROR] {r['clip_id']}: {exc}")
            errors += 1
            continue
        dur = float(r.get("duration_raw_s") or 0.0)
        f.update(pause_features(r.get("speech_segments"), dur))
        words = len((r.get("transcript") or "").split())
        f["speaking_rate_wps"] = words / f["speech_total_s"] if f["speech_total_s"] >= 2.0 else 0.0  # too little speech -> unreliable
        feats[r["clip_id"]] = {k: (0.0 if not np.isfinite(v) else round(v, 6)) for k, v in f.items()}
        if i % 50 == 0 or i == len(recs):
            rate = i / max(time.time() - t0, 1e-6)
            print(f"  [{i}/{len(recs)}] {rate:.1f} clips/s  ETA {(len(recs) - i) / max(rate, 1e-6) / 60:.1f} min")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(feats, f)
    n = len(next(iter(feats.values()))) if feats else 0
    print(f"[INFO] {len(feats)} clips x {n} features -> {out_path}  (errors: {errors})")


if __name__ == "__main__":
    main()
