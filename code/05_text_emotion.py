"""
05_text_emotion.py
═══════════════════════════════════════════════════════════════════════════════
TEXT-ONLY session emotion recognition, from the FREESTYLE transcript.

The passage transcript is deliberately excluded. It is the same script every
session, so its lexical content is constant across the entire corpus and
carries zero session-specific signal. (Its ACOUSTICS are still valuable and
are used in 04_audio_emotion.py as a neutral reference.)

    python 05_text_emotion.py --cv louo
    python 05_text_emotion.py --cv all
    python 05_text_emotion.py --cv all --no_augment     # ablation

ASR-NOISE AUGMENTATION
─────────────────────────────────────────────────────────────────────────────
These transcripts come from Whisper on spontaneous, disfluent speech, so they
contain real errors. MASKSER's core result is that training the text branch on
deliberately corrupted transcripts is what makes it robust -- in their ablation,
removing the augmentation cost 7.5 to 11.2 points of weighted accuracy, far
more than any fusion refinement.

MASKSER used ChatGPT-4 for the corruption. That is not available offline here,
so this implements the cheaper variants they also validated:
  - word masking / dropout
  - word-order jitter on adjacent tokens
  - random word repetition (mirrors ASR stutter artefacts)
Augmented copies are generated ONLY for training folds, never for test folds,
so this cannot leak.

QUALITY FEATURES
─────────────────────────────────────────────────────────────────────────────
Whisper's avg_logprob and no_speech_prob are appended to the feature vector so
the model can learn to discount unreliable transcripts rather than trusting a
garbled one as much as a clean one.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (
    EMOTION_LABELS, N_EMOTIONS, DEVICE,
    load_video_index, load_features, resolve_stem,
    dominant_emotion, emotions_to_multilabel,
    run_cv, save_results,
)

ROOT = Path(r"C:\Users\2022g\Downloads\RAVDESS_IEMOCAP_CREMAD")
TEXT_EMB_DIM = 384
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_encoder = None


def get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(TEXT_MODEL)
        print(f"[INFO] Sentence encoder loaded: {TEXT_MODEL}")
    return _encoder


def encode_batch(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, TEXT_EMB_DIM), dtype=np.float32)
    embs = get_encoder().encode(texts, batch_size=32, show_progress_bar=False,
                                convert_to_numpy=True)
    return embs.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ASR-noise augmentation
# ─────────────────────────────────────────────────────────────────────────────
def corrupt_transcript(text: str, rng: random.Random,
                       mask_p: float = 0.12,
                       swap_p: float = 0.05,
                       repeat_p: float = 0.04) -> str:
    """Simulate the kinds of errors Whisper actually makes on spontaneous speech."""
    words = text.split()
    if len(words) < 4:
        return text

    out = []
    for w in words:
        r = rng.random()
        if r < mask_p:
            continue                        # deletion
        if r < mask_p + repeat_p:
            out.extend([w, w])              # stutter / duplication
            continue
        out.append(w)

    # Adjacent transpositions
    for i in range(len(out) - 1):
        if rng.random() < swap_p:
            out[i], out[i + 1] = out[i + 1], out[i]

    return " ".join(out) if out else text


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(video_index: dict, features: dict):
    session_map: dict[tuple, dict] = defaultdict(dict)
    for (uid, sid, vtype), info in video_index.items():
        session_map[(uid, sid)][vtype] = info

    metas, texts, extra, y7, ym = [], [], [], [], []
    n_empty = 0

    for (uid, sid), tasks in sorted(session_map.items()):
        reported = (tasks.get("freestyle", {}).get("reported_emotions")
                    or tasks.get("passage", {}).get("reported_emotions") or [])
        if not reported:
            continue
        dom = dominant_emotion(reported)
        if dom not in EMOTION_LABELS:
            continue

        f_rec = resolve_stem(tasks.get("freestyle", {}), features)
        if not f_rec:
            continue

        transcript = (f_rec.get("transcript") or "").strip()
        if not transcript:
            n_empty += 1

        n_words = len(transcript.split())
        dur = f_rec.get("duration_s") or 1.0
        extra.append(np.array([
            float(f_rec.get("asr_avg_logprob", -1.0)),
            float(f_rec.get("asr_no_speech_prob", 0.5)),
            float(n_words),
            float(n_words / max(float(dur), 0.1)),
        ], dtype=np.float32))

        metas.append({"user_id": uid, "session_id": sid,
                      "dominant": dom, "reported": reported})
        texts.append(transcript)
        y7.append(EMOTION_LABELS.index(dom))
        ym.append(emotions_to_multilabel(reported))

    if not texts:
        raise SystemExit("[ERROR] No usable freestyle transcripts.")
    if n_empty:
        print(f"[WARN] {n_empty} session(s) have an empty transcript.")

    print(f"[INFO] Encoding {len(texts)} transcripts ...")
    emb = encode_batch(texts)
    extra_arr = np.stack(extra)
    X = np.hstack([emb, extra_arr]).astype(np.float32)

    groups = np.array([m["user_id"] for m in metas])
    return (metas, X, np.array(y7, dtype=np.int64),
            np.stack(ym).astype(np.float32), groups, texts, extra_arr)


def main():
    p = argparse.ArgumentParser(description="Text-only session emotion recognition")
    p.add_argument("--cv", default="louo", choices=["kfold", "grouped", "louo", "all"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--features", type=Path, default=ROOT / "code/data/features.jsonl")
    p.add_argument("--video_index", type=Path,
                   default=ROOT / "firebase_data" / "video_index.json")
    p.add_argument("--out_dir", type=Path, default=ROOT / "code/data/results")
    p.add_argument("--augment", type=int, default=2,
                   help="Corrupted copies per training transcript (0 = off)")
    p.add_argument("--no_augment", action="store_true")
    args = p.parse_args()

    n_copies = 0 if args.no_augment else args.augment

    print(f"[INFO] Device: {DEVICE}")
    video_index = load_video_index(args.video_index)
    features = load_features(args.features)

    metas, X, y7, ym, groups, texts, extra = build_dataset(video_index, features)
    tag = "text" + ("" if n_copies else "_noaug")

    print(f"[INFO] Feature dim: {X.shape[1]} "
          f"({TEXT_EMB_DIM} sentence emb + 4 ASR-quality)")
    print(f"[INFO] ASR-noise augmentation: "
          f"{n_copies} extra copies per training sample"
          if n_copies else "[INFO] ASR-noise augmentation: OFF")

    words = [len(t.split()) for t in texts]
    print(f"[INFO] Transcript length: mean={np.mean(words):.0f} words  "
          f"median={np.median(words):.0f}  min={min(words)}  max={max(words)}")

    methods = ["kfold", "grouped", "louo"] if args.cv == "all" else [args.cv]
    all_results = {}

    # Augmentation runs inside each fold, on the TRAINING indices only.
    # run_cv hands us the train indices and appends whatever we return, so
    # corrupted copies of test sessions can never reach the training set.
    def make_augment_fn(seed: int):
        def augment_fn(train_idx: np.ndarray):
            rng = random.Random(seed + int(train_idx[0]))
            aug_texts, aug_y, aug_ym, aug_extra = [], [], [], []
            for _ in range(n_copies):
                for i in train_idx:
                    t = texts[i]
                    if not t.strip():
                        continue
                    aug_texts.append(corrupt_transcript(t, rng))
                    aug_y.append(y7[i])
                    aug_ym.append(ym[i])
                    aug_extra.append(extra[i])
            if not aug_texts:
                return None
            aug_X = np.hstack([encode_batch(aug_texts),
                               np.stack(aug_extra)]).astype(np.float32)
            return (aug_X,
                    np.array(aug_y, dtype=np.int64),
                    np.stack(aug_ym).astype(np.float32))
        return augment_fn

    aug_fn = make_augment_fn(args.seed) if n_copies > 0 else None

    for method in methods:
        res = run_cv(X, y7, ym, groups, method,
                     n_folds=args.folds, seed=args.seed, tag=tag,
                     augment_fn=aug_fn)
        res["augment_copies"] = n_copies
        all_results[method] = res
        save_results(res, args.out_dir / f"{tag}_{method}.json")

    if len(methods) > 1:
        print(f"\n{'=' * 74}")
        print(f"  COMPARISON -- {tag}")
        print(f"{'=' * 74}")
        print(f"  {'CV method':<12} {'acc7':>7} {'mF1_7':>7} {'wF1_7':>7} "
              f"{'acc3':>7} {'mF1_3':>7} {'flex':>7}")
        print(f"  {'-' * 62}")
        for m in methods:
            q = all_results[m]["pooled"]
            print(f"  {m:<12} {q['acc7']:>7.4f} {q['f1_macro7']:>7.4f} "
                  f"{q['f1_weighted7']:>7.4f} {q['acc3_head']:>7.4f} "
                  f"{q['f1_macro3_head']:>7.4f} {q['flexible_acc']:>7.4f}")

    print(f"\n[INFO] All results -> {args.out_dir}")


if __name__ == "__main__":
    main()
