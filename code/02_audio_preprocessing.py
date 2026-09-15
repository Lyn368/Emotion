"""
02_audio_preprocessing.py
═══════════════════════════════════════════════════════════════════════════════
Load -> mono 16 kHz
  1. DC offset removal
  2. Bandpass filter (80 Hz - 7.2 kHz)      keeps the speech band
  3. Spectral gating denoising               GENTLE by default (see below)
  4. Silero VAD v5                           removes silence, keeps speech
  5. Loudness normalisation (EBU R128)       for ASR / emotion2vec consistency

WHAT CHANGED FROM v2 AND WHY
─────────────────────────────────────────────────────────────────────────────
(a) denoise_strength lowered 0.60 -> 0.25.
    Spectral gating at 0.6 subtracts a large constant from every frequency bin.
    On quiet, neutral-toned speech that removes low-amplitude harmonics, which
    is exactly where pitch and voice-quality information lives. The papers that
    motivate this pipeline rely on pitch / energy / speaking-rate as features,
    so over-denoising destroys the signal we are trying to measure. Raise it
    only for files where the manifest reports a genuinely low SNR.

(b) Loudness is MEASURED BEFORE normalisation and written to the manifest.
    v2 normalised every clip to -23 LUFS and the downstream code then read
    "energy" back off those normalised files -- so every clip reported ~-23
    LUFS, sigma ~ 0, and the energy feature was constant and useless.
    The normalised audio is still what Whisper and emotion2vec consume (they
    prefer consistent levels); the TRUE loudness now lives in the manifest and
    is what the feature extractor reads.

(c) A per-file manifest (preprocess_manifest.jsonl) records original_lufs,
    gain_db, SNR estimate, VAD kept ratio and durations. Downstream stages
    read this instead of re-measuring from processed audio.

Output:
    PreprocessedAudio/<stem>.wav        normalised, denoised, VAD-trimmed
    PreprocessedAudio/preprocess_manifest.jsonl
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
TARGET_SR = 16_000

ROOT = Path("/Users/ytinglin/Downloads/ProjectCE903/25-26_CE903-SP_team01/MLmodel/data")


@dataclass
class Config:
    audio_root: Path = ROOT / "audios"
    output_dir: Path = ROOT / "PreprocessedAudio"
    manifest_path: Path = ROOT / "PreprocessedAudio" / "preprocess_manifest.jsonl"
    exts: tuple = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm")

    skip_existing: bool = True

    # Bandpass
    hp_cutoff_hz: float = 80.0
    lp_cutoff_hz: float = 7_200.0

    # Spectral gating -- GENTLE. See header note (a).
    denoise_strength: float = 0.25
    denoise_n_fft: int = 512
    denoise_hop: int = 128

    # Silero VAD v5
    vad_threshold: float = 0.45
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 100
    vad_speech_pad_ms: int = 60
    vad_min_keep_seconds: float = 1.0

    # Loudness normalisation
    target_lufs: float = -23.0
    lufs_fallback_rms: float = 0.06

    max_seconds: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────
def iter_audio_files(root: Path, exts: tuple) -> list[Path]:
    files: list[Path] = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def load_wav_mono_16k(path: Path, max_seconds: Optional[float] = None) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    wav = wav.squeeze(0)
    if max_seconds is not None:
        wav = wav[: int(TARGET_SR * max_seconds)]
    return wav


def save_wav_16k(wav: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = wav.detach().cpu().float().clamp(-1.0, 1.0)
    torchaudio.save(str(path), x.unsqueeze(0), sample_rate=TARGET_SR,
                    encoding="PCM_S", bits_per_sample=16)


def rms_db(wav: torch.Tensor) -> float:
    rms = float(torch.sqrt((wav ** 2).mean() + 1e-12).item())
    return 20.0 * float(np.log10(max(rms, 1e-9)))


def measure_lufs(x: torch.Tensor) -> Optional[float]:
    """Integrated loudness (EBU R128). None if pyloudnorm missing or signal silent."""
    try:
        import pyloudnorm as pyln
    except ImportError:
        return None
    wav_np = x.detach().cpu().numpy().astype(np.float64)
    # pyloudnorm needs at least ~0.4 s for the integrated gate
    if len(wav_np) < int(TARGET_SR * 0.4):
        return None
    try:
        val = pyln.Meter(TARGET_SR).integrated_loudness(wav_np)
    except Exception:
        return None
    if val == float("-inf") or np.isnan(val):
        return None
    return float(val)


def estimate_snr_db(x: torch.Tensor) -> float:
    """
    Crude SNR: ratio of the 90th-percentile frame energy (speech) to the
    10th-percentile frame energy (noise floor). Only used to decide whether
    a file actually needs denoising -- not used as a model feature.
    """
    frame = int(TARGET_SR * 0.03)
    if x.numel() < frame * 4:
        return 0.0
    n_frames = x.numel() // frame
    frames = x[: n_frames * frame].reshape(n_frames, frame)
    energies = (frames ** 2).mean(dim=1).cpu().numpy()
    energies = np.maximum(energies, 1e-12)
    speech = float(np.percentile(energies, 90))
    noise = float(np.percentile(energies, 10))
    return float(10.0 * np.log10(speech / max(noise, 1e-12)))


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — DC offset removal
# ─────────────────────────────────────────────────────────────────────────────
def dc_remove(x: torch.Tensor) -> torch.Tensor:
    return x - x.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Bandpass filter
# ─────────────────────────────────────────────────────────────────────────────
def bandpass_filter(x: torch.Tensor, hp_hz: float, lp_hz: float) -> torch.Tensor:
    x = torchaudio.functional.highpass_biquad(x, TARGET_SR, hp_hz)
    x = torchaudio.functional.lowpass_biquad(x, TARGET_SR, lp_hz)
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Spectral gating denoising
# ─────────────────────────────────────────────────────────────────────────────
def spectral_gate_denoise(x: torch.Tensor, strength: float = 0.25,
                          n_fft: int = 512, hop: int = 128) -> torch.Tensor:
    if strength <= 0.0 or x.numel() < n_fft:
        return x

    win = torch.hann_window(n_fft, device=x.device)
    X = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=win, return_complex=True)
    mag, phase = X.abs(), torch.angle(X)

    # Quietest 0.5 s window -> noise profile
    seg_len = int(TARGET_SR * 0.5)
    if x.numel() <= seg_len:
        noise_ref = x
    else:
        step = max(seg_len // 2, 1)
        best_start, best_energy = 0, float("inf")
        for s in range(0, x.numel() - seg_len, step):
            e = float((x[s: s + seg_len] ** 2).mean().item())
            if e < best_energy:
                best_energy, best_start = e, s
        noise_ref = x[best_start: best_start + seg_len]

    N = torch.stft(noise_ref, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                   window=win, return_complex=True).abs()
    noise_profile = N.mean(dim=1, keepdim=True)

    new_mag = (mag - strength * noise_profile).clamp_min(0.0)
    Y = new_mag * torch.exp(1j * phase)
    return torch.istft(Y, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                       window=win, length=x.numel())


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Silero VAD v5
# ─────────────────────────────────────────────────────────────────────────────
_silero_model = None
_silero_utils = None


def load_silero_vad():
    global _silero_model, _silero_utils
    if _silero_model is None:
        try:
            from silero_vad import load_silero_vad as _load, get_speech_timestamps
            _silero_model = _load()
            _silero_utils = get_speech_timestamps
            print("[INFO] Silero VAD v5 loaded")
        except ImportError:
            print("[WARN] silero-vad not installed -> energy VAD fallback")
            print("       pip install silero-vad")
            _silero_model = "energy"
    return _silero_model, _silero_utils


def vad_silero(x: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, dict]:
    model, get_timestamps = load_silero_vad()
    if model == "energy":
        return _vad_energy_fallback(x, cfg)

    timestamps = get_timestamps(
        x.float(), model,
        threshold=cfg.vad_threshold,
        sampling_rate=TARGET_SR,
        min_speech_duration_ms=cfg.vad_min_speech_ms,
        min_silence_duration_ms=cfg.vad_min_silence_ms,
        speech_pad_ms=cfg.vad_speech_pad_ms,
    )
    if not timestamps:
        return x, {"vad_method": "silero_v5", "speech_segments": 0,
                   "kept_ratio": 1.0, "fallback": True}

    trimmed = torch.cat([x[s["start"]:s["end"]] for s in timestamps])
    kept = trimmed.numel() / max(x.numel(), 1)

    if trimmed.numel() < int(TARGET_SR * cfg.vad_min_keep_seconds):
        return x, {"vad_method": "silero_v5", "speech_segments": len(timestamps),
                   "kept_ratio": round(kept, 3), "fallback": True,
                   "note": "too short after VAD, kept original"}

    # speech_seconds is the real duration of voiced speech -- this is what the
    # feature extractor must divide by to get an honest speaking rate.
    return trimmed, {
        "vad_method": "silero_v5",
        "speech_segments": len(timestamps),
        "kept_ratio": round(kept, 3),
        "fallback": False,
    }


def _vad_energy_fallback(x: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, dict]:
    T = x.numel()
    frame = max(64, int(TARGET_SR * 0.030))
    hop = max(32, int(TARGET_SR * 0.010))
    thresh, eps = -35.0, 1e-12

    loud = []
    for start in range(0, T - frame, hop):
        seg = x[start: start + frame]
        db = 20.0 * float(torch.log10(torch.sqrt((seg ** 2).mean() + eps) + eps).item())
        if db > thresh:
            loud.append((start, start + frame))

    if not loud:
        return x, {"vad_method": "energy_fallback", "kept_ratio": 1.0, "fallback": True}

    s, e = min(a for a, _ in loud), max(b for _, b in loud)
    trimmed = x[s:e]
    ratio = trimmed.numel() / max(T, 1)
    if trimmed.numel() < int(TARGET_SR * cfg.vad_min_keep_seconds):
        return x, {"vad_method": "energy_fallback", "kept_ratio": round(ratio, 3),
                   "fallback": True}
    return trimmed, {"vad_method": "energy_fallback",
                     "kept_ratio": round(ratio, 3), "fallback": False}


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Loudness normalisation
# ─────────────────────────────────────────────────────────────────────────────
def loudness_normalise(x: torch.Tensor, measured_lufs: Optional[float],
                       target_lufs: float, rms_fallback: float) -> tuple[torch.Tensor, dict]:
    """
    Normalise to target LUFS. `measured_lufs` is passed in so the caller keeps
    the pre-normalisation value -- this is the whole point of the v3 change.
    """
    if measured_lufs is None:
        return _rms_normalise(x, rms_fallback), {
            "method": "rms_fallback", "measured_lufs": None, "gain_db": None}

    gain_db = target_lufs - measured_lufs
    gain = float(10.0 ** (gain_db / 20.0))
    out = (x * gain).clamp(-1.0, 1.0)
    return out, {
        "method": "lufs_ebu_r128",
        "measured_lufs": round(measured_lufs, 2),
        "target_lufs": target_lufs,
        "gain_db": round(gain_db, 2),
    }


def _rms_normalise(x: torch.Tensor, target_rms: float = 0.06) -> torch.Tensor:
    rms = float(torch.sqrt((x ** 2).mean() + 1e-12).item())
    if rms < 1e-6:
        return x
    gain = float(np.clip(target_rms / rms, 0.1, 10.0))
    return (x * gain).clamp(-1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Full chain
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_audio(wav: torch.Tensor, cfg: Config) -> tuple[torch.Tensor, dict]:
    meta: dict = {}
    meta["input_duration_s"] = round(wav.numel() / TARGET_SR, 3)
    meta["input_rms_db"] = round(rms_db(wav), 2)
    meta["snr_db_estimate"] = round(estimate_snr_db(wav), 2)

    x = dc_remove(wav)
    x = bandpass_filter(x, cfg.hp_cutoff_hz, cfg.lp_cutoff_hz)

    x = x.detach().cpu()
    x = spectral_gate_denoise(x, cfg.denoise_strength, cfg.denoise_n_fft, cfg.denoise_hop)
    meta["denoise_strength"] = cfg.denoise_strength

    x, vad_info = vad_silero(x, cfg)
    meta["vad"] = vad_info

    # ── CRITICAL: measure loudness on the VAD-trimmed but NOT yet normalised
    # signal. This is the value the feature extractor uses as "energy".
    original_lufs = measure_lufs(x)
    meta["original_lufs"] = round(original_lufs, 2) if original_lufs is not None else None

    x, lufs_info = loudness_normalise(x, original_lufs, cfg.target_lufs, cfg.lufs_fallback_rms)
    meta["loudness"] = lufs_info

    meta["output_duration_s"] = round(x.numel() / TARGET_SR, 3)
    meta["speech_seconds"] = meta["output_duration_s"]   # post-VAD == voiced speech
    meta["output_rms_db"] = round(rms_db(x), 2)
    meta["duration_change_s"] = round(
        meta["output_duration_s"] - meta["input_duration_s"], 3)
    return x, meta


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(cfg: Config) -> None:
    audio_root = cfg.audio_root.expanduser().resolve()
    output_dir = cfg.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = iter_audio_files(audio_root, cfg.exts)
    if not wav_files:
        raise SystemExit(f"[ERROR] No audio files in: {audio_root}")

    print(f"[INFO] {len(wav_files)} file(s) | device={DEVICE}")
    print(f"[INFO] denoise_strength={cfg.denoise_strength}  target_lufs={cfg.target_lufs}")
    print(f"[INFO] Output -> {output_dir}\n")

    load_silero_vad()

    # Resume support: keep manifest entries for files we skip.
    existing: dict[str, dict] = {}
    if cfg.manifest_path.exists():
        with open(cfg.manifest_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        existing[rec["file"]] = rec
                    except Exception:
                        pass

    records: dict[str, dict] = dict(existing)
    processed = skipped = errors = 0

    for i, wav_path in enumerate(wav_files, 1):
        stem = wav_path.stem
        out_wav = output_dir / f"{stem}.wav"

        if cfg.skip_existing and out_wav.exists() and f"{stem}.wav" in records:
            skipped += 1
            continue

        t0 = time.time()
        try:
            wav = load_wav_mono_16k(wav_path, cfg.max_seconds)
            processed_wav, meta = preprocess_audio(wav, cfg)
            save_wav_16k(processed_wav, out_wav)

            records[f"{stem}.wav"] = {
                "file": f"{stem}.wav",
                "source": str(wav_path),
                "processed_path": str(out_wav),
                **meta,
            }
            processed += 1

            lufs_str = (f"{meta['original_lufs']:.1f}"
                        if meta["original_lufs"] is not None else "  n/a")
            print(
                f"  [{i:3d}/{len(wav_files)}] {wav_path.name:<44} "
                f"{meta['input_duration_s']:5.1f}s -> {meta['output_duration_s']:5.1f}s  "
                f"VAD {meta['vad'].get('kept_ratio', 1.0):.0%}  "
                f"LUFS {lufs_str}  SNR {meta['snr_db_estimate']:5.1f}dB  "
                f"{time.time() - t0:.1f}s"
            )
        except Exception as exc:
            print(f"  [{i:3d}/{len(wav_files)}] ERROR {wav_path.name}: {exc}")
            errors += 1

    # ── Write manifest ───────────────────────────────────────────────────────
    cfg.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.manifest_path, "w") as f:
        for rec in records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ── Low-SNR report: which files might actually need heavier denoising ────
    low_snr = [r for r in records.values() if r.get("snr_db_estimate", 99) < 15.0]

    print(f"\n{'-' * 74}")
    print(f"  Processed: {processed} | Skipped: {skipped} | Errors: {errors}")
    print(f"  Manifest -> {cfg.manifest_path}")
    if low_snr:
        print(f"\n  [!] {len(low_snr)} file(s) with SNR < 15 dB. Consider re-running")
        print(f"      just these with a higher --denoise value:")
        for r in sorted(low_snr, key=lambda r: r["snr_db_estimate"])[:10]:
            print(f"        {r['file']:<48} {r['snr_db_estimate']:5.1f} dB")
    else:
        print(f"\n  All files above 15 dB SNR -- gentle denoising is appropriate.")
    print(f"{'-' * 74}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Audio preprocessing v3")
    p.add_argument("--audio_root", type=Path, default=None)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--denoise", type=float, default=None,
                   help="Spectral gating strength 0.0-1.0 (default 0.25)")
    p.add_argument("--target_lufs", type=float, default=None)
    p.add_argument("--no_skip", action="store_true", help="Reprocess existing files")
    args = p.parse_args()

    cfg = Config()
    if args.audio_root:
        cfg.audio_root = args.audio_root
    if args.output_dir:
        cfg.output_dir = args.output_dir
        cfg.manifest_path = args.output_dir / "preprocess_manifest.jsonl"
    if args.denoise is not None:
        cfg.denoise_strength = args.denoise
    if args.target_lufs is not None:
        cfg.target_lufs = args.target_lufs
    if args.no_skip:
        cfg.skip_existing = False

    main(cfg)
