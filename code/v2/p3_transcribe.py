"""
p3_transcribe.py — Transcribe once, on the GPU, for the whole pipeline.
═══════════════════════════════════════════════════════════════════════════════
    python p3_transcribe.py                          # own + iemocap, whisper medium
    python p3_transcribe.py --datasets own
    python p3_transcribe.py --model large-v3         # better, slower
    python p3_transcribe.py --limit 10               # timing test

Output:  preprocessing/transcripts.jsonl   (resumable: re-running skips done clips)

WHAT GETS TRANSCRIBED
─────────────────────────────────────────────────────────────────────────────
own       Both clips. The passage transcript is never used as a text feature
          (the script is identical every session) but it is needed to compute
          reading rate.
iemocap   Whisper output is stored NEXT TO the official transcript. The text
          model is pre-trained on IEMOCAP and then evaluated on your Whisper
          transcripts; if it only ever saw perfect transcripts in pre-training
          it would meet ASR errors for the first time at test time. MASKSER
          shows that mismatch is what hurts text models most.
ravdess / cremad are fixed sentences with known text; nothing to transcribe.

Whisper reads the ENCODER version of the audio (cleaned, silences removed).
Audio is passed as an array, so ffmpeg is not needed for this step.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from config import ensure_dirs, get_device


def load_whisper(size: str, device: str):
    import whisper
    return whisper.load_model(size, device=device)


def transcribe_array(model, y: np.ndarray, device: str) -> dict:
    res = model.transcribe(y.astype(np.float32), fp16=(device == "cuda"),
                           language="en", condition_on_previous_text=False)
    segs = res.get("segments") or []
    return {
        "transcript": (res.get("text") or "").strip(),
        "asr_avg_logprob": float(np.mean([s["avg_logprob"] for s in segs])) if segs else None,
        "asr_no_speech_prob": float(np.mean([s["no_speech_prob"] for s in segs])) if segs else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Whisper transcription")
    ap.add_argument("--datasets", default="own,iemocap")
    ap.add_argument("--model", default="medium", help="tiny|base|small|medium|large-v3")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dirs = ensure_dirs()
    man = dirs["prep"] / "audio_manifest.jsonl"
    if not man.exists():
        raise SystemExit("[ERROR] Run p2_preprocess_audio.py first.")
    keep = {d.strip() for d in args.datasets.split(",")}
    clips = [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines() if l.strip()]
    clips = [c for c in clips if c["dataset"] in keep]

    out_path = dirs["prep"] / "transcripts.jsonl"
    done = set()
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                done.add(json.loads(l)["clip_id"])
    todo = [c for c in clips if c["clip_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[INFO] {len(clips)} clips in {sorted(keep)} | {len(done)} done | {len(todo)} to do")
    if not todo:
        return

    device = get_device()
    model = load_whisper(args.model, device)
    print(f"[INFO] Whisper {args.model} loaded on {device}")

    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i, c in enumerate(todo, 1):
            y, sr = sf.read(c["encoder_path"], dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr != 16_000:        # Whisper assumes 16 kHz; anything else transcribes garbage silently
                print(f"  [ERROR] {c['clip_id']}: sample rate {sr}, expected 16000. Re-run p2.")
                continue
            try:
                r = transcribe_array(model, y, device)
            except Exception as exc:
                print(f"  [ERROR] {c['clip_id']}: {exc}")
                continue
            r.update({"clip_id": c["clip_id"], "asr_model": f"whisper-{args.model}"})
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-6)
                print(f"  [{i}/{len(todo)}] {rate:.2f} clips/s  "
                      f"ETA {(len(todo) - i) / max(rate, 1e-6) / 60:.1f} min  | {r['transcript'][:60]}")
    print(f"[INFO] -> {out_path}")


if __name__ == "__main__":
    main()
