"""
03_feature_extraction.py
═══════════════════════════════════════════════════════════════════════════════
For every preprocessed clip, produce:
  - emotion2vec embedding (768-d) + its own label prediction
  - Whisper transcript          (BOTH passage and freestyle -- see note below)
  - acoustic attributes: pitch / energy / speaking rate, raw AND discretised
  - passage-baseline DELTA features for freestyle clips

WHAT CHANGED AND WHY
─────────────────────────────────────────────────────────────────────────────
(1) PITCH IS ACTUALLY MEASURED.
    The old compute_corpus_stats() did `pitches.append(150.0)` -- a constant.
    That gave mu=150, sigma=0, so _discretise() collapsed to a hard threshold
    at 150 Hz, which separates typical male from typical female voices. The
    "pitch_level" feature was a speaker-sex detector, not an emotion feature,
    and it leaked speaker identity straight into the model. Now every clip is
    measured with PENN (or an autocorrelation fallback).

(2) ENERGY COMES FROM THE MANIFEST, NOT THE NORMALISED FILE.
    02_audio_preprocessing.py records original_lufs before normalising.
    Reading LUFS back off the normalised .wav (as the old code did) returns
    ~-23 for every clip, sigma ~ 0.

(3) STATISTICS ARE PER-USER.
    C2SER discretises pitch/energy/rate into Low/Medium/High at mu +/- sigma.
    With 9 speakers of differing baselines, GLOBAL mu/sigma means "High pitch"
    just identifies who is talking. Per-user mu/sigma makes it mean "high FOR
    THIS PERSON", which is what carries emotional information -- especially
    for a corpus of neutral-toned, non-acted speech. Users with too few clips
    fall back to global stats (flagged in the output).

(4) THE PASSAGE IS TRANSCRIBED TOO.
    Not for text features -- the passage script is identical every session, so
    its lexical content carries no session-specific information, and it stays
    out of the text model. But a transcript is needed to compute reading rate,
    and reading rate on FIXED text is a clean, lexically-controlled measure of
    speaking tempo. Previously passage clips had transcript="" so their rate
    was 0, and those zeros contaminated the corpus mean and sigma.

(5) DELTA FEATURES (freestyle - passage, within the same session).
    Because the passage is a fixed script recorded in the same session on the
    same microphone, the difference between freestyle and passage cancels both
    the speaker's baseline voice and that session's recording conditions.
    This follows the logic of Aldeneh & Mower Provost (2023): emotion is
    measured as a departure from the speaker's own neutral reference rather
    than in absolute terms.
    This is NOT leakage: the passage is available at inference time for the
    same session, it never crosses users, and it never touches the labels.

Output: features.jsonl (one record per clip)
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

from features_audio import extract_features, DEFAULT_BLOCKS, block_of

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
TARGET_SR = 16_000
ROOT = Path(r"C:\Users\2022g\Downloads\RAVDESS_IEMOCAP_CREMAD\code\data")

EMB_DIM = 768
MIN_CLIPS_FOR_USER_STATS = 8      # below this, fall back to global stats


@dataclass
class Config:
    processed_audio_dir: Path = ROOT / "PreprocessedAudio"
    manifest_path: Path = ROOT / "PreprocessedAudio" / "preprocess_manifest.jsonl"
    video_index_path: Path = Path(r"C:\Users\2022g\Downloads\RAVDESS_IEMOCAP_CREMAD\firebase_data\video_index.json")
    output_jsonl: Path = ROOT / "features.jsonl"

    skip_existing: bool = True

    whisper_model_size: str = "medium"

    # Handcrafted acoustic features (eGeMAPS via openSMILE, librosa fallback).
    # A compact named feature set, extracted alongside the embeddings so
    # 07_feature_ablation.py can test them AGAINST the embeddings rather than
    # piling both into one oversized vector.
    extract_handcrafted_feats: bool = True
    prefer_egemaps: bool = True
    emotion2vec_model_id: str = "iic/emotion2vec_plus_large"

    # Which handcrafted families to compute. See features_audio.py.
    hc_blocks: tuple = DEFAULT_BLOCKS
    hc_fast: bool = False          # skip temporal skew/kurt, coarser LSF grid
    hc_workers: int = 4            # parallel processes for the handcrafted pass

    # Stages that can be skipped. On 20k public-corpus clips, Whisper-medium
    # takes days and buys nothing: RAVDESS and CREMA-D read fixed sentences, so
    # their transcripts carry no session-specific information. Turn ASR off
    # there and leave it on for your own spontaneous recordings.
    do_asr: bool = True
    do_e2v: bool = True

    # Reuse clips already present in an existing features.jsonl instead of
    # recomputing them. Matters when a corpus is ADDED to a folder that was
    # already extracted: emotion2vec on CPU is the slowest stage by far, and
    # there is no reason to pay it twice for clips that have not changed.
    resume: bool = False

    # Path to clip_sources.jsonl. The public-corpus layout points many stems at
    # the SAME source wav, because one neutral reference is shared by every
    # emotional utterance of that speaker. Roughly half of all stems are such
    # duplicates. With this set, the per-clip work runs once per unique source
    # and the result is fanned out, which halves the emotion2vec pass — the
    # slowest stage by far on CPU.
    clip_sources: Path | None = None

    VALID_EMOTIONS: tuple = ("angry", "disgust", "fear", "happy",
                             "neutral", "sad", "surprise")
    LABEL_MAP: dict = field(default_factory=lambda: {
        "happiness": "happy", "happy": "happy",
        "sadness": "sad", "sad": "sad",
        "fear": "fear", "fearful": "fear",
        "surprise": "surprise", "surprised": "surprise",
        "anger": "angry", "angry": "angry",
        "disgust": "disgust", "disgusted": "disgust",
        "neutral": "neutral", "calm": "neutral",
    })


def normalise_label(raw: str, cfg: Config) -> str:
    return cfg.LABEL_MAP.get(str(raw).strip().lower(), "neutral")


def load_wav(path: Path) -> torch.Tensor:
    """
    Mono waveform at TARGET_SR.

    torchaudio.load first, soundfile as a fallback. Recent torchaudio routes
    load() through torchcodec and raises ImportError when it is absent, which
    would otherwise take down the whole run over a decoder dependency.
    """
    try:
        wav, sr = torchaudio.load(str(path))
    except Exception:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T.copy())

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────────
def load_video_index(path: Path) -> dict[str, dict]:
    """{stem: {user_id, session_id, video_type, reported_emotion}}"""
    with open(path) as f:
        entries = json.load(f)
    lookup = {}
    for e in entries:
        info = {
            "user_id": e.get("user_id", ""),
            "session_id": e.get("session_id", ""),
            "video_type": e.get("video_type", "").lower(),
            "reported_emotion": e.get("reported_emotion", ""),
        }
        stem = e.get("video_id", "").replace("-webm", "").replace(".webm", "")
        if stem:
            lookup[stem] = info
        lp = Path(e.get("local_path", "")).stem
        if lp and lp not in lookup:
            lookup[lp] = info
    return lookup


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        print(f"[WARN] Manifest not found: {path}")
        print("       Energy will fall back to measuring the normalised file,")
        print("       which makes it near-constant. Re-run 02_audio_preprocessing.py.")
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[Path(rec["file"]).stem] = rec
    return out


def is_passage(meta: dict) -> bool:
    return meta.get("video_type", "").lower() == "passage"


# ─────────────────────────────────────────────────────────────────────────────
# emotion2vec
# ─────────────────────────────────────────────────────────────────────────────
def load_emotion2vec(model_id: str):
    try:
        from funasr import AutoModel
    except ImportError:
        raise ImportError("pip install funasr")
    model = AutoModel(model=model_id, disable_update=True)
    print("[INFO] emotion2vec loaded")
    return model


def run_emotion2vec(wav_path: Path, model, cfg: Config) -> tuple[str, list[float]]:
    rec = model.generate(str(wav_path), output_dir=None,
                         granularity="utterance", extract_embedding=True)[0]
    label = normalise_label(rec.get("labels", ["neutral"])[0], cfg)
    feats = rec.get("feats", None)
    emb = feats.squeeze().tolist() if feats is not None else [0.0] * EMB_DIM
    return label, emb


# ─────────────────────────────────────────────────────────────────────────────
# Whisper
# ─────────────────────────────────────────────────────────────────────────────
def load_whisper(size: str = "medium"):
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    import whisper
    model = whisper.load_model(size)
    print(f"[INFO] Whisper-{size} loaded")
    return model


def transcribe(wav_path: Path, model) -> dict:
    """
    Return the transcript plus Whisper's own confidence signals.

    avg_logprob and no_speech_prob are cheap proxies for transcription quality.
    MASKSER's central finding is that ASR noise is what limits text-based
    emotion recognition; keeping these lets the text model down-weight
    unreliable transcripts instead of trusting them equally.
    """
    res = model.transcribe(str(wav_path), fp16=(DEVICE == "cuda"), language="en")
    segs = res.get("segments", []) or []
    return {
        "transcript": res.get("text", "").strip(),
        "avg_logprob": float(np.mean([s.get("avg_logprob", -1.0) for s in segs])) if segs else -1.0,
        "no_speech_prob": float(np.mean([s.get("no_speech_prob", 0.5) for s in segs])) if segs else 0.5,
        "n_segments": len(segs),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Acoustic attributes
# ─────────────────────────────────────────────────────────────────────────────
def extract_pitch(wav_path: Path) -> float:
    """Mean F0 over voiced frames, in Hz. 0.0 if nothing voiced."""
    try:
        import os
        import tempfile
        import penn
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp_path = tmp.name
        penn.from_file(wav_path, hopsize=0.01, fmin=50.0, fmax=550.0,
                       checkpoint=penn.DEFAULT_CHECKPOINT,
                       gpu=(0 if DEVICE == "cuda" else None),
                       output_path=tmp_path)
        data = torch.load(tmp_path)
        os.unlink(tmp_path)
        f0 = data["frequency"]
        voiced = f0[f0 > 0]
        return float(voiced.mean().item()) if len(voiced) > 0 else 0.0
    except ImportError:
        return _pitch_autocorr(load_wav(wav_path).numpy())
    except Exception:
        return _pitch_autocorr(load_wav(wav_path).numpy())


def _pitch_autocorr(wav_np: np.ndarray, min_hz: float = 50.0,
                    max_hz: float = 550.0) -> float:
    frame_len = int(TARGET_SR * 0.04)
    hop = int(TARGET_SR * 0.01)
    min_lag = int(TARGET_SR / max_hz)
    max_lag = int(TARGET_SR / min_hz)
    pitches = []
    for start in range(0, max(len(wav_np) - frame_len, 0), hop):
        frame = wav_np[start: start + frame_len]
        frame = frame - frame.mean()
        if np.abs(frame).max() < 1e-4:
            continue
        corr = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
        if max_lag >= len(corr):
            continue
        peak = min_lag + int(np.argmax(corr[min_lag:max_lag]))
        if peak > 0:
            pitches.append(TARGET_SR / peak)
    return float(np.mean(pitches)) if pitches else 0.0


def extract_energy(stem: str, manifest: dict, wav_path: Path) -> float:
    """
    True loudness in LUFS, taken from the preprocessing manifest.
    Falls back to measuring the file only if the manifest is missing -- in
    which case the value is near-constant and effectively useless, so the
    caller warns about it.
    """
    rec = manifest.get(stem, {})
    lufs = rec.get("original_lufs")
    if lufs is not None:
        return float(lufs)
    try:
        import pyloudnorm as pyln
        import soundfile as sf
        data, rate = sf.read(str(wav_path))
        val = pyln.Meter(rate).integrated_loudness(data)
        return float(val) if val != float("-inf") else -60.0
    except Exception:
        wav = load_wav(wav_path)
        return 20.0 * float(np.log10(
            max(float(torch.sqrt((wav ** 2).mean() + 1e-12).item()), 1e-9)))


def extract_speaking_rate(transcript: str, stem: str, manifest: dict,
                          wav_path: Path) -> float:
    """
    Words per second of VOICED speech.

    Dividing by total duration (as the old code did) confounds tempo with how
    much the person paused. Using post-VAD speech_seconds from the manifest
    measures articulation rate proper -- which is what C2SER's speaking-rate
    attribute is meant to capture.
    """
    n_words = len(transcript.split())
    if n_words == 0:
        return 0.0
    secs = manifest.get(stem, {}).get("speech_seconds")
    if secs is None:
        try:
            wav, sr = torchaudio.load(str(wav_path))
            secs = wav.shape[-1] / sr
        except Exception:
            secs = 1.0
    return float(n_words / max(float(secs), 0.1))


def discretise(value: float, mu: float, sigma: float) -> str:
    """C2SER's three-level mapping. Guards against a degenerate sigma."""
    if sigma is None or sigma < 1e-6:
        return "Medium"
    if value < mu - sigma:
        return "Low"
    if value > mu + sigma:
        return "High"
    return "Medium"


def mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if v is not None and not np.isnan(v)]
    if not vals:
        return 0.0, 1.0
    mu = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 1.0
    return float(mu), float(max(sd, 1e-6))


def compute_stats(raw: dict[str, dict], metas: dict[str, dict]) -> tuple[dict, dict]:
    """
    Compute (per_user_stats, global_stats) for pitch / energy / rate.

    Zero values are excluded: a pitch of 0 means "nothing voiced" and a rate of
    0 means "no words", and including them drags mu toward zero and inflates
    sigma, which is what previously made the Low/Medium/High buckets
    meaningless.
    """
    by_user: dict[str, dict[str, list]] = defaultdict(
        lambda: {"pitch": [], "energy": [], "rate": []})
    glob: dict[str, list] = {"pitch": [], "energy": [], "rate": []}

    for stem, vals in raw.items():
        uid = metas.get(stem, {}).get("user_id", "UNKNOWN")
        for key in ("pitch", "energy", "rate"):
            v = vals.get(key)
            if v is None:
                continue
            if key in ("pitch", "rate") and v <= 0:
                continue
            by_user[uid][key].append(v)
            glob[key].append(v)

    global_stats = {k: mean_std(v) for k, v in glob.items()}

    per_user = {}
    for uid, d in by_user.items():
        n = max(len(d["pitch"]), len(d["energy"]), len(d["rate"]))
        if n < MIN_CLIPS_FOR_USER_STATS:
            per_user[uid] = {**global_stats, "_source": "global_fallback", "_n": n}
            continue
        per_user[uid] = {
            "pitch": mean_std(d["pitch"]),
            "energy": mean_std(d["energy"]),
            "rate": mean_std(d["rate"]),
            "_source": "per_user",
            "_n": n,
        }
    return per_user, global_stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Handcrafted bank, in parallel
# ─────────────────────────────────────────────────────────────────────────────
def _hc_one(job):
    path, blocks, fast = job
    try:
        return path.stem, extract_features(path, blocks=blocks, fast=fast)
    except Exception as exc:
        return path.stem, {"set": "error", "features": {}, "dim": 0, "error": str(exc)}


def extract_handcrafted_parallel(wav_files, blocks, fast, workers):
    """
    The feature bank is pure CPU work with no shared state, so it parallelises
    cleanly. On ~20k public-corpus clips this is the difference between one
    hour and most of a day.
    """
    jobs = [(p, list(blocks), fast) for p in wav_files]
    out: dict[str, dict] = {}

    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            stem, rec = _hc_one(job)
            out[stem] = rec
            if i % 100 == 0 or i == len(jobs):
                print(f"  [{i:5d}/{len(jobs)}]", flush=True)
        return out

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (stem, rec) in enumerate(ex.map(_hc_one, jobs, chunksize=8), 1):
            out[stem] = rec
            if i % 100 == 0 or i == len(jobs):
                print(f"  [{i:5d}/{len(jobs)}]", flush=True)

    bad = [s for s, r in out.items() if r.get("set") == "error"]
    if bad:
        print(f"  [!] {len(bad)} clip(s) failed, e.g. {bad[:3]}")
    dims = {r["dim"] for r in out.values() if r.get("dim")}
    if len(dims) > 1:
        print(f"  [!] MIXED feature dimensions across clips: {sorted(dims)}. "
              f"Columns will not line up — usually openSMILE succeeding on some "
              f"files and not others.")
    return out


def main(cfg: Config) -> None:
    cfg.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(cfg.processed_audio_dir.glob("*.wav"))
    if not wav_files:
        raise SystemExit(f"[ERROR] No .wav in {cfg.processed_audio_dir}")

    # ── Source deduplication ─────────────────────────────────────────────────
    dup_of: dict[str, str] = {}          # duplicate stem -> representative stem
    if cfg.clip_sources and cfg.clip_sources.exists():
        by_source: dict[str, list[str]] = {}
        with open(cfg.clip_sources, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                c = json.loads(line)
                by_source.setdefault(c["source"], []).append(c["stem"])
        present = {w.stem for w in wav_files}
        for stems in by_source.values():
            kept = sorted(st for st in stems if st in present)
            for st in kept[1:]:
                dup_of[st] = kept[0]
        if dup_of:
            wav_files = [w for w in wav_files if w.stem not in dup_of]
            print(f"[INFO] Source dedup: {len(dup_of)} stem(s) share audio with "
                  f"another stem.\n       Computing {len(wav_files)} unique "
                  f"clip(s) and copying the result to the rest.")

    existing: dict[str, dict] = {}
    if cfg.resume and cfg.output_jsonl.exists():
        with open(cfg.output_jsonl, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                stem = rec.get("stem") or Path(rec.get("file", "")).stem
                if stem:
                    existing[stem] = rec
        todo = [w for w in wav_files if w.stem not in existing]
        print(f"[INFO] Resume: {len(existing)} clip(s) already in "
              f"{cfg.output_jsonl.name}, {len(todo)} to compute.")
        if not todo:
            print("[INFO] Nothing new. Per-user statistics are recomputed only "
                  "over new clips,\n       so if you changed the corpus "
                  "composition, rerun without --resume.")
        wav_files = todo or wav_files

    if not cfg.video_index_path.exists():
        raise SystemExit(
            f"[ERROR] video_index.json not found: {cfg.video_index_path}\n"
            f"        Pass --video_index. Without it every clip gets user_id="
            f"UNKNOWN,\n        per-user statistics collapse to global, and the "
            f"labels are lost.")
    video_lookup = load_video_index(cfg.video_index_path)
    manifest = load_manifest(cfg.manifest_path)
    print(f"[INFO] {len(wav_files)} clips | {len(video_lookup)} index entries "
          f"| {len(manifest)} manifest entries")

    # Deduplicated stems are dropped from wav_files but still need metadata,
    # since the fan-out at the end rebuilds their records from a twin.
    metas = {f.stem: video_lookup.get(f.stem, {}) for f in wav_files}
    for st in dup_of:
        metas.setdefault(st, video_lookup.get(st, {}))
    n_pass = sum(1 for m in metas.values() if is_passage(m))
    print(f"[INFO] passage={n_pass}  freestyle={len(wav_files) - n_pass}\n")

    # ── PASS 1: transcripts (both clip types) ────────────────────────────────
    asr: dict[str, dict] = {}
    if cfg.do_asr:
        print("[1/5] Transcribing (passage AND freestyle) ...")
        whisper_model = load_whisper(cfg.whisper_model_size)
        for i, p in enumerate(wav_files, 1):
            asr[p.stem] = transcribe(p, whisper_model)
            preview = asr[p.stem]["transcript"][:58]
            kind = "P" if is_passage(metas[p.stem]) else "F"
            print(f"  [{i:3d}/{len(wav_files)}] ({kind}) {preview}...")
        del whisper_model
        if DEVICE == "mps":
            torch.mps.empty_cache()
    else:
        print("[1/5] ASR skipped (--no_asr). Speaking rate falls back to the")
        print("      VAD voiced-duration estimate; the text branch (05) is")
        print("      unavailable for this corpus.")
        asr = {p.stem: {"transcript": "", "avg_logprob": -1.0,
                        "no_speech_prob": 0.5, "n_segments": 0} for p in wav_files}

    # ── PASS 2: raw acoustic values ──────────────────────────────────────────
    print("\n[2/5] Extracting raw acoustic values ...")
    raw: dict[str, dict] = {}
    for i, p in enumerate(wav_files, 1):
        t = asr[p.stem]["transcript"]
        raw[p.stem] = {
            "pitch": extract_pitch(p),
            "energy": extract_energy(p.stem, manifest, p),
            "rate": extract_speaking_rate(t, p.stem, manifest, p),
        }
        if i % 25 == 0 or i == len(wav_files):
            print(f"  [{i:3d}/{len(wav_files)}]")

    # ── PASS 3: per-user statistics ──────────────────────────────────────────
    print("\n[3/5] Computing PER-USER statistics ...")
    if existing:
        new_users = {m.get("user_id", "UNKNOWN") for m in metas.values()}
        old_users = {r.get("user_id", "UNKNOWN") for r in existing.values()}
        overlap = new_users & old_users
        if overlap:
            print(f"  [!] {len(overlap)} user(s) appear in BOTH the reused and "
                  f"the new clips,\n      e.g. {sorted(overlap)[:3]}. Their "
                  f"per-user mean and sigma are now computed\n      from the new "
                  f"clips only, so their discretised levels no longer match\n"
                  f"      the reused ones. Re-run WITHOUT --resume for those "
                  f"users to be correct.")
        else:
            print(f"  Resume is safe here: the {len(new_users)} new user(s) do "
                  f"not overlap the reused ones,\n  and per-user statistics are "
                  f"per user by construction.")
    per_user, global_stats = compute_stats(raw, metas)
    print(f"  global  pitch mu={global_stats['pitch'][0]:7.1f} sd={global_stats['pitch'][1]:6.1f} "
          f"| energy mu={global_stats['energy'][0]:6.1f} sd={global_stats['energy'][1]:5.1f} "
          f"| rate mu={global_stats['rate'][0]:5.2f} sd={global_stats['rate'][1]:4.2f}")
    for uid in sorted(per_user):
        s = per_user[uid]
        flag = "" if s["_source"] == "per_user" else "  <- GLOBAL FALLBACK"
        print(f"  {uid:<22} n={s['_n']:<4} pitch mu={s['pitch'][0]:7.1f} sd={s['pitch'][1]:6.1f}"
              f" | rate mu={s['rate'][0]:5.2f}{flag}")

    if global_stats["energy"][1] < 0.5:
        print("\n  [!] Global energy sigma is near zero. The manifest is probably")
        print("      missing, so energy was read off normalised audio. Re-run")
        print("      02_audio_preprocessing.py before trusting energy features.")

    # ── PASS 4: handcrafted feature bank (parallel) ──────────────────────────
    hc_by_stem: dict[str, dict] = {}
    if cfg.extract_handcrafted_feats:
        blocks = list(cfg.hc_blocks) + (["egemaps"] if cfg.prefer_egemaps else [])
        print(f"\n[4/5] Handcrafted feature bank: {blocks} "
              f"({cfg.hc_workers} workers) ...")
        hc_by_stem = extract_handcrafted_parallel(
            wav_files, blocks, cfg.hc_fast, cfg.hc_workers)
    else:
        hc_by_stem = {p.stem: {"set": "off", "features": {}, "dim": 0}
                      for p in wav_files}

    # ── PASS 5: emotion2vec + assemble records ───────────────────────────────
    print("\n[5/5] emotion2vec embeddings + writing records ...")
    e2v = load_emotion2vec(cfg.emotion2vec_model_id) if cfg.do_e2v else None
    if not cfg.do_e2v:
        print("  emotion2vec skipped (--no_e2v); embeddings are zero-filled.")

    records: dict[str, dict] = {}
    for i, p in enumerate(wav_files, 1):
        stem = p.stem
        meta = metas[stem]
        uid = meta.get("user_id", "UNKNOWN")
        stats = per_user.get(uid, global_stats)
        t0 = time.time()

        if cfg.do_e2v:
            try:
                e2v_label, e2v_emb = run_emotion2vec(p, e2v, cfg)
            except Exception as exc:
                print(f"  [WARN] emotion2vec failed on {stem}: {exc}")
                e2v_label, e2v_emb = "neutral", [0.0] * EMB_DIM
        else:
            e2v_label, e2v_emb = "neutral", [0.0] * EMB_DIM

        hc = hc_by_stem.get(stem, {"set": "off", "features": {}, "dim": 0})

        r = raw[stem]
        attrs = {
            "pitch_hz": round(r["pitch"], 2),
            "pitch_level": discretise(r["pitch"], *stats["pitch"]),
            "energy_lufs": round(r["energy"], 2),
            "energy_level": discretise(r["energy"], *stats["energy"]),
            "speaking_rate_wps": round(r["rate"], 3),
            "speaking_rate_level": discretise(r["rate"], *stats["rate"]),
            "stats_source": stats.get("_source", "global"),
        }

        clip_type = "passage" if is_passage(meta) else "freestyle"
        records[stem] = {
            "file": p.name,
            "stem": stem,
            "video_type": clip_type,
            "user_id": uid,
            "session_id": meta.get("session_id", ""),
            "reported_emotion": meta.get("reported_emotion", ""),
            "transcript": asr[stem]["transcript"],
            "asr_avg_logprob": round(asr[stem]["avg_logprob"], 4),
            "asr_no_speech_prob": round(asr[stem]["no_speech_prob"], 4),
            "acoustic_attrs": attrs,
            "acoustic_raw": {k: round(v, 4) for k, v in r.items()},
            "emotion2vec_label": e2v_label,
            "emotion2vec_embedding": e2v_emb,
            "duration_s": manifest.get(stem, {}).get("output_duration_s"),
            "snr_db": manifest.get(stem, {}).get("snr_db_estimate"),
            "handcrafted": hc,
        }
        print(f"  [{i:3d}/{len(wav_files)}] [{clip_type[:4]}] {stem[:28]:<28} "
              f"e2v={e2v_label:<9} pitch={attrs['pitch_level']:<6} "
              f"eng={attrs['energy_level']:<6} rate={attrs['speaking_rate_level']:<6} "
              f"{time.time() - t0:.1f}s")

    # ── DELTA features: freestyle minus passage, same session ────────────────
    print("\n[+] Computing passage-baseline delta features ...")
    by_session: dict[tuple, dict] = defaultdict(dict)
    for stem, rec in records.items():
        key = (rec["user_id"], rec["session_id"])
        by_session[key][rec["video_type"]] = rec

    n_delta = 0
    for key, pair in by_session.items():
        fs, pa = pair.get("freestyle"), pair.get("passage")
        if fs is None or pa is None:
            continue
        fa, pb = fs["acoustic_raw"], pa["acoustic_raw"]
        fs["delta_vs_passage"] = {
            "d_pitch_hz": round(fa["pitch"] - pb["pitch"], 3),
            "d_energy_lufs": round(fa["energy"] - pb["energy"], 3),
            "d_rate_wps": round(fa["rate"] - pb["rate"], 3),
            # Ratios are scale-free, so they survive differences in absolute
            # baseline better than raw differences do.
            "r_pitch": round(fa["pitch"] / pb["pitch"], 4) if pb["pitch"] > 1 else 1.0,
            "r_rate": round(fa["rate"] / pb["rate"], 4) if pb["rate"] > 0.1 else 1.0,
        }
        # Handcrafted delta: the same neutral-reference logic applied to every
        # named acoustic feature, not only to pitch/energy/rate. 11 can then
        # test each family with and without its delta twin and see which ones
        # the reference actually helps.
        f_hc = fs.get("handcrafted", {}).get("features", {})
        p_hc = pa.get("handcrafted", {}).get("features", {})
        if f_hc and p_hc:
            shared = set(f_hc) & set(p_hc)
            fs["handcrafted_delta"] = {
                k: round(float(f_hc[k]) - float(p_hc[k]), 6) for k in sorted(shared)}

        # Embedding delta: emotional departure from this session's own neutral
        # reading of a fixed script.
        f_emb = np.array(fs["emotion2vec_embedding"], dtype=np.float32)
        p_emb = np.array(pa["emotion2vec_embedding"], dtype=np.float32)
        if f_emb.shape == p_emb.shape and f_emb.size == EMB_DIM:
            fs["embedding_delta"] = (f_emb - p_emb).tolist()
            denom = float(np.linalg.norm(f_emb) * np.linalg.norm(p_emb))
            fs["embedding_cosine_to_passage"] = (
                round(float(np.dot(f_emb, p_emb) / denom), 4) if denom > 0 else 0.0)
        n_delta += 1

    print(f"  {n_delta}/{len(by_session)} sessions have both clips -> delta computed")
    incomplete = len(by_session) - n_delta
    if incomplete:
        print(f"  [!] {incomplete} session(s) missing one clip; those get zero deltas "
              f"in the classifiers.")

    # ── Write ────────────────────────────────────────────────────────────────
    # ── Fan out deduplicated clips ───────────────────────────────────────────
    if dup_of:
        n_fanned = 0
        for stem, rep in dup_of.items():
            src = records.get(rep)
            if src is None:
                continue
            meta = metas.get(stem, {})
            clone = dict(src)
            clone.update({
                "stem": stem,
                "video_type": "passage" if is_passage(meta) else "freestyle",
                "user_id": meta.get("user_id", src.get("user_id", "UNKNOWN")),
                "session_id": meta.get("session_id", ""),
                "reported_emotion": meta.get("reported_emotion", ""),
                "duplicate_of": rep,
            })
            records[stem] = clone
            n_fanned += 1
        print(f"[INFO] Copied {n_fanned} record(s) from their source twin. "
              f"Acoustic values and\n       embeddings are identical by "
              f"construction; only the session metadata differs.")

    if existing:
        merged = dict(existing)
        merged.update(records)          # freshly computed clips win
        records_out = merged
        print(f"[INFO] Merged {len(records)} new + {len(existing)} reused "
              f"= {len(records_out)} clips")
    else:
        records_out = records

    with open(cfg.output_jsonl, "w", encoding="utf-8") as f:
        for rec in records_out.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    hc_sets = {r.get("handcrafted", {}).get("set", "off") for r in records.values()}
    hc_dims = {r.get("handcrafted", {}).get("dim", 0) for r in records.values()}
    if hc_sets != {"off"}:
        print(f"\n[+] Handcrafted features: set(s)={sorted(hc_sets)} dim(s)={sorted(hc_dims)}")
        if len(hc_sets) > 1 or len(hc_dims) > 1:
            print("    [!] MIXED feature sets across clips. Columns will not line up.")
            print("        Usually means openSMILE succeeded on some files and fell back")
            print("        on others. Install openSMILE for all, or set")
            print("        prefer_egemaps=False to force librosa everywhere.")
        elif "librosa" in hc_sets:
            print("    Note: librosa fallback lacks jitter, shimmer and HNR — the voice")
            print("    quality measures that matter most for neutral-toned speech.")
            print("    pip install opensmile")

    print(f"\n{'=' * 74}")
    print(f"  Features -> {cfg.output_jsonl}  ({len(records_out)} clips)")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Feature extraction")
    p.add_argument("--audio_dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--video_index", type=Path, default=None,
                   help="video_index.json for this corpus (00 writes one per "
                        "dataset folder; without it, user ids and labels are "
                        "unknown and per-user statistics fall back to global)")
    p.add_argument("--manifest", type=Path, default=None,
                   help="preprocess_manifest.jsonl; defaults to "
                        "<audio_dir>/preprocess_manifest.jsonl")
    p.add_argument("--whisper", type=str, default=None,
                   help="tiny|base|small|medium|large")
    p.add_argument("--no_handcrafted", action="store_true",
                   help="Skip the handcrafted feature bank entirely")
    p.add_argument("--force_librosa", action="store_true",
                   help="Skip the eGeMAPS block even if openSMILE is installed")
    p.add_argument("--blocks", default=None,
                   help="Handcrafted families to compute, comma separated: "
                        "mfcc,dmfcc,ddmfcc,logmel,spectral,lsf,prosody")
    p.add_argument("--fast", action="store_true",
                   help="Skip temporal skew/kurtosis, coarser LSF grid")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel processes for the handcrafted pass")
    p.add_argument("--no_asr", action="store_true",
                   help="Skip Whisper. Use on RAVDESS/CREMA-D, whose sentences "
                        "are fixed and carry no session-specific text signal")
    p.add_argument("--clip_sources", type=Path, default=None,
                   help="clip_sources.jsonl from 00. Computes each unique "
                        "source wav once instead of once per stem — about half "
                        "the work on the public corpora")
    p.add_argument("--resume", action="store_true",
                   help="Keep clips already present in the output file and "
                        "compute only the new ones")
    p.add_argument("--no_e2v", action="store_true",
                   help="Skip emotion2vec (e.g. when only the handcrafted bank "
                        "or your own encoder is needed)")
    args = p.parse_args()

    cfg = Config()
    if args.audio_dir:
        cfg.processed_audio_dir = args.audio_dir
        cfg.manifest_path = args.audio_dir / "preprocess_manifest.jsonl"
    if args.manifest:
        cfg.manifest_path = args.manifest
    if args.video_index:
        cfg.video_index_path = args.video_index
    if args.out:
        cfg.output_jsonl = args.out
    if args.whisper:
        cfg.whisper_model_size = args.whisper
    if args.no_handcrafted:
        cfg.extract_handcrafted_feats = False
    if args.force_librosa:
        cfg.prefer_egemaps = False
    if args.blocks:
        cfg.hc_blocks = tuple(b.strip() for b in args.blocks.split(",") if b.strip())
    if args.fast:
        cfg.hc_fast = True
    cfg.hc_workers = args.workers
    if args.no_asr:
        cfg.do_asr = False
    if args.no_e2v:
        cfg.do_e2v = False
    if args.resume:
        cfg.resume = True
    if args.clip_sources:
        cfg.clip_sources = args.clip_sources
    elif args.video_index:
        guess = args.video_index.parent.parent / "clip_sources.jsonl"
        if guess.exists():
            cfg.clip_sources = guess
            print(f"[INFO] Using {guess} for source deduplication "
                  f"(--clip_sources to override)")

    main(cfg)