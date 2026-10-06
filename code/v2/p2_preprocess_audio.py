"""
p2_preprocess_audio.py — Two versions of every clip + VAD timestamps.
═══════════════════════════════════════════════════════════════════════════════
    python p2_preprocess_audio.py
    python p2_preprocess_audio.py --datasets own          # just your recordings
    python p2_preprocess_audio.py --limit 20              # quick timing test

Output:
    preprocessing/audio/measurement/<clip_id>.wav
    preprocessing/audio/encoder/<clip_id>.wav
    preprocessing/audio_manifest.jsonl

WHY TWO VERSIONS
─────────────────────────────────────────────────────────────────────────────
measurement   DC offset removed, resampled to 16 kHz, silence trimmed at the
              START and END only. No filtering, no denoising, no normalisation.
              Pitch, jitter, shimmer, HNR and loudness are measured on this,
              because denoising and normalisation change exactly those values.
encoder       Bandpass 80 Hz-7.2 kHz, gentle spectral gating (0.25), internal
              silences removed by VAD, loudness normalised to -23 LUFS.
              Whisper, emotion2vec and WavLM read this one.

WHY THE TIMESTAMPS
─────────────────────────────────────────────────────────────────────────────
The old pipeline cut the silences out and threw away where they were. How
often and how long someone pauses is one of the stronger cues for low mood,
so every speech segment is kept as [start, end] in seconds of the ORIGINAL
recording. Pause features are computed from these later.

Windows notes: audio is read and written with soundfile (no torchaudio
backends); resampling and filtering use scipy. Compressed formats (.webm,
.m4a, .mp3) are decoded with ffmpeg, which must be on PATH.
Resumable: clips already in the manifest are skipped.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, istft, resample_poly, sosfiltfilt, stft

from config import SAMPLE_RATE, ensure_dirs

SR = SAMPLE_RATE
DENOISE_STRENGTH = 0.25
TARGET_LUFS = -23.0
EDGE_PAD_S = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_audio(path: Path) -> np.ndarray:
    """Mono float32 at 16 kHz."""
    try:
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
    except Exception:
        if not shutil.which("ffmpeg"):
            raise RuntimeError(f"cannot decode {path.name}; install ffmpeg and add it to PATH")
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le",
             "-ac", "1", "-ar", str(SR), "pipe:1"],
            capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype=np.float32).copy()
    if sr != SR:
        g = gcd(sr, SR)
        y = resample_poly(y, SR // g, sr // g).astype(np.float32)
    return y


def save(path: Path, y: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(y, -1.0, 1.0), SR, subtype="PCM_16")


# ─────────────────────────────────────────────────────────────────────────────
# Signal processing
# ─────────────────────────────────────────────────────────────────────────────
_SOS = butter(4, [80, 7200], btype="bandpass", fs=SR, output="sos")


def bandpass(y):
    return sosfiltfilt(_SOS, y).astype(np.float32) if len(y) > 64 else y


def spectral_gate(y, strength=DENOISE_STRENGTH, nfft=512, hop=128):
    if strength <= 0 or len(y) < nfft * 2:
        return y
    _, _, Z = stft(y, fs=SR, nperseg=nfft, noverlap=nfft - hop)
    mag, ph = np.abs(Z), np.angle(Z)
    frame_e = mag.mean(axis=0)
    k = max(1, int(0.1 * len(frame_e)))
    noise = mag[:, np.argsort(frame_e)[:k]].mean(axis=1, keepdims=True)  # quietest 10%
    mag = np.maximum(mag - strength * noise, 0.0)
    _, out = istft(mag * np.exp(1j * ph), fs=SR, nperseg=nfft, noverlap=nfft - hop)
    return out[: len(y)].astype(np.float32)


def lufs(y) -> float | None:
    try:
        import pyloudnorm as pyln
        if len(y) < int(0.4 * SR):
            return None
        v = pyln.Meter(SR).integrated_loudness(y.astype(np.float64))
        return None if not np.isfinite(v) else float(v)
    except ImportError:
        return None


def normalise(y, measured):
    if measured is None:
        rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
        return y * min(10.0, 0.06 / rms)
    return y * float(10 ** ((TARGET_LUFS - measured) / 20))


def snr_db(y) -> float:
    fr = int(0.03 * SR)
    n = len(y) // fr
    if n < 4:
        return 0.0
    e = np.maximum((y[: n * fr].reshape(n, fr) ** 2).mean(1), 1e-12)
    return float(10 * np.log10(np.percentile(e, 90) / np.percentile(e, 10)))


# ─────────────────────────────────────────────────────────────────────────────
# VAD
# ─────────────────────────────────────────────────────────────────────────────
_silero = None


def vad(y) -> tuple[list[tuple[int, int]], str]:
    """Speech segments as (start_sample, end_sample). Silero if installed."""
    global _silero
    if _silero is None:
        try:
            import torch  # noqa
            from silero_vad import get_speech_timestamps, load_silero_vad
            _silero = (load_silero_vad(), get_speech_timestamps)
        except Exception:
            _silero = False
            print("[WARN] silero-vad not available -> energy VAD (pip install silero-vad)")
    if _silero:
        import torch
        model, fn = _silero
        ts = fn(torch.from_numpy(y.astype(np.float32)), model, sampling_rate=SR,
                threshold=0.45, min_speech_duration_ms=250,
                min_silence_duration_ms=100, speech_pad_ms=60)
        return [(t["start"], t["end"]) for t in ts], "silero"
    return energy_vad(y), "energy"


def energy_vad(y, frame_s=0.03, min_speech_s=0.25, min_gap_s=0.10):
    fr = int(frame_s * SR)
    n = len(y) // fr
    if n == 0:
        return []
    e = 10 * np.log10(np.maximum((y[: n * fr].reshape(n, fr) ** 2).mean(1), 1e-12))
    thr = max(np.percentile(e, 95) - 30.0, -60.0)
    on = e > thr
    segs, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append([start * fr, i * fr]); start = None
    if start is not None:
        segs.append([start * fr, n * fr])
    merged = []
    for s in segs:                                      # close short gaps
        if merged and s[0] - merged[-1][1] < min_gap_s * SR:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return [tuple(s) for s in merged if s[1] - s[0] >= min_speech_s * SR]


# ─────────────────────────────────────────────────────────────────────────────
def process(rec: dict, dirs: dict) -> dict:
    y = load_audio(Path(rec["audio_path"]))
    y = (y - y.mean()).astype(np.float32)
    dur = len(y) / SR

    segs, method = vad(bandpass(y))
    fallback = not segs
    if fallback:
        segs = [(0, len(y))]

    # measurement: raw signal, trimmed at the edges only
    pad = int(EDGE_PAD_S * SR)
    a, b = max(0, segs[0][0] - pad), min(len(y), segs[-1][1] + pad)
    meas = y[a:b]
    meas_lufs = lufs(meas)

    # encoder: filtered, gently denoised, internal silences removed, normalised
    clean = spectral_gate(bandpass(y))
    speech = np.concatenate([clean[s:e] for s, e in segs]) if segs else clean
    enc = normalise(speech, lufs(speech))

    m_path = dirs["audio_measure"] / f"{rec['clip_id']}.wav"
    e_path = dirs["audio_encoder"] / f"{rec['clip_id']}.wav"
    save(m_path, meas)
    save(e_path, enc)

    return {
        "clip_id": rec["clip_id"], "dataset": rec["dataset"],
        "measure_path": str(m_path), "encoder_path": str(e_path),
        "duration_raw_s": round(dur, 3),
        "duration_speech_s": round(sum(e - s for s, e in segs) / SR, 3),
        "speech_segments": [[round(s / SR, 3), round(e / SR, 3)] for s, e in segs],
        "measured_lufs": None if meas_lufs is None else round(meas_lufs, 2),
        "snr_db": round(snr_db(y), 2), "vad": method, "vad_fallback": fallback,
    }


def main():
    ap = argparse.ArgumentParser(description="Audio preprocessing v2")
    ap.add_argument("--datasets", default=None, help="e.g. own,iemocap (default: all indexed)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--redo", action="store_true", help="Reprocess clips already done")
    args = ap.parse_args()

    dirs = ensure_dirs()
    idx_path = dirs["prep"] / "index.jsonl"
    if not idx_path.exists():
        raise SystemExit("[ERROR] Run p1_index_datasets.py first.")
    recs = [json.loads(l) for l in idx_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.datasets:
        keep = {d.strip() for d in args.datasets.split(",")}
        recs = [r for r in recs if r["dataset"] in keep]

    man_path = dirs["prep"] / "audio_manifest.jsonl"
    existing = {}
    if man_path.exists():
        for l in man_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                existing[r["clip_id"]] = r
    selected = {r["clip_id"] for r in recs}
    # --redo only discards the SELECTED clips; other datasets stay untouched
    keep = {k: v for k, v in existing.items() if not (args.redo and k in selected)}
    todo = [r for r in recs if r["clip_id"] not in keep]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[INFO] {len(recs)} clips selected | {len(selected & keep.keys())} already done "
          f"| {len(todo)} to process")
    if not shutil.which("ffmpeg"):
        print("[INFO] ffmpeg not on PATH: fine for .wav, required for .webm/.m4a/.mp3")

    t0, errors = time.time(), 0
    with open(man_path, "w", encoding="utf-8") as f:
        for r in keep.values():
            f.write(json.dumps(r) + "\n")
        f.flush()
        for i, rec in enumerate(todo, 1):
            try:
                out = process(rec, dirs)
                f.write(json.dumps(out) + "\n")
                f.flush()
            except Exception as exc:
                errors += 1
                print(f"  [ERROR] {rec['clip_id']}: {exc}")
            if i % 50 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-6)
                print(f"  [{i}/{len(todo)}] {rate:.1f} clips/s  "
                      f"ETA {(len(todo) - i) / max(rate, 1e-6) / 60:.1f} min")

    all_done = [json.loads(l) for l in man_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    fb = sum(r["vad_fallback"] for r in all_done)
    low = sum(r["snr_db"] < 15 for r in all_done)
    print(f"\n[INFO] Manifest: {len(all_done)} clips | errors this run: {errors}")
    print(f"       VAD found no speech (whole clip kept): {fb}")
    print(f"       SNR < 15 dB: {low}")


if __name__ == "__main__":
    main()
