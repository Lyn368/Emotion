"""
04_audio_emotion.py
═══════════════════════════════════════════════════════════════════════════════
AUDIO-ONLY session emotion recognition.
No transcript, no text embeddings -- acoustics only. This is the honest audio
baseline, and the number to compare the text model and the fusion against.

Produces 7-class AND 3-class results in one run (the model has a dedicated
head for each), under whichever CV scheme you ask for.

    python 04_audio_emotion.py --cv louo
    python 04_audio_emotion.py --cv grouped --folds 5
    python 04_audio_emotion.py --cv kfold
    python 04_audio_emotion.py --cv all          # runs all three, prints a table

FEATURE SET (per session)
─────────────────────────────────────────────────────────────────────────────
  freestyle emotion2vec embedding        768
  passage   emotion2vec embedding        768   (session-specific neutral reference)
  embedding delta (freestyle - passage)  768   [--no_delta to disable]
  freestyle acoustic  (3 raw + 3 level)    6
  passage   acoustic  (3 raw + 3 level)    6
  delta acoustic (d_pitch, d_energy, d_rate, r_pitch, r_rate)  5
  cosine(freestyle, passage)               1
  emotion2vec label one-hot, both clips   14

The delta block is the point of keeping the passage. Its lexical content is
identical every session so it carries no semantic information, but that is
exactly what makes it a controlled acoustic reference: any difference between
freestyle and passage within a session is attributable to speaker state, not
to what was said or to the microphone.

Run with --no_delta to produce the ablation for the report.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (
    EMOTION_LABELS, N_EMOTIONS, CV_METHODS, DEVICE,
    load_video_index, load_features, resolve_stem,
    dominant_emotion, emotions_to_multilabel, parse_reported_emotions,
    run_cv, save_results,
)
from features_audio import block_of

ROOT = Path(r"C:\Users\2022g\Downloads\RAVDESS_IEMOCAP_CREMAD")
EMB_DIM = 768
ACOUSTIC_DIM = 6
DELTA_ACOUSTIC_DIM = 5

LEVEL_MAP = {"Low": 0.0, "Medium": 0.5, "High": 1.0}


def acoustic_vec(attrs: dict) -> np.ndarray:
    return np.array([
        LEVEL_MAP.get(attrs.get("pitch_level", "Medium"), 0.5),
        LEVEL_MAP.get(attrs.get("energy_level", "Medium"), 0.5),
        LEVEL_MAP.get(attrs.get("speaking_rate_level", "Medium"), 0.5),
        float(attrs.get("pitch_hz", 0.0)),
        float(attrs.get("energy_lufs", 0.0)),
        float(attrs.get("speaking_rate_wps", 0.0)),
    ], dtype=np.float32)


def delta_vec(rec: dict) -> np.ndarray:
    d = rec.get("delta_vs_passage", {})
    return np.array([
        float(d.get("d_pitch_hz", 0.0)),
        float(d.get("d_energy_lufs", 0.0)),
        float(d.get("d_rate_wps", 0.0)),
        float(d.get("r_pitch", 1.0)),
        float(d.get("r_rate", 1.0)),
    ], dtype=np.float32)


def label_onehot(label: str) -> np.ndarray:
    v = np.zeros(N_EMOTIONS, dtype=np.float32)
    if label in EMOTION_LABELS:
        v[EMOTION_LABELS.index(label)] = 1.0
    return v


def emb_of(rec: dict, key: str = "emotion2vec_embedding") -> np.ndarray:
    e = rec.get(key, [])
    arr = np.asarray(e, dtype=np.float32)
    return arr if arr.size == EMB_DIM else np.zeros(EMB_DIM, dtype=np.float32)


def hc_keys_by_block(features: dict) -> dict:
    """Handcrafted keys common to every clip, grouped by family."""
    common = None
    for r in features.values():
        f = r.get("handcrafted", {}).get("features", {})
        if not f:
            continue
        common = set(f) if common is None else (common & set(f))
    out = defaultdict(list)
    for k in sorted(common or []):
        out[block_of(k)].append(k)
    return dict(out)


def _hc_vec(rec: dict, keys: list) -> np.ndarray:
    f = (rec or {}).get("handcrafted", {}).get("features", {})
    return np.nan_to_num(np.array([float(f.get(k, 0.0)) for k in keys],
                                  dtype=np.float32))


def build_dataset(video_index: dict, features: dict, use_delta: bool = True,
                  encoder: str = "emotion2vec", hc_blocks: str = ""):
    """
    Assemble one feature row per (user_id, session_id).

    encoder   : 'emotion2vec' | 'custom' | 'both' | 'none'
                'custom' uses the embedding written by 09_extract_embeddings.py
                from the encoder you trained in 08. Everything downstream —
                the CV protocol, the classifier, the metrics — is identical, so
                the only thing that changes between runs is the representation.
    hc_blocks : comma-separated handcrafted families to append, e.g.
                'mfcc,dmfcc,spectral,lsf,prosody'. Empty means none.
    """
    session_map: dict[tuple, dict] = defaultdict(dict)
    for (uid, sid, vtype), info in video_index.items():
        session_map[(uid, sid)][vtype] = info

    want_e2v = encoder in ("emotion2vec", "both")
    want_cus = encoder in ("custom", "both")
    cus_dim = max((len(r.get("custom_embedding", [])) for r in features.values()),
                  default=0)
    if want_cus and not cus_dim:
        raise SystemExit("[ERROR] No custom_embedding in features.jsonl. "
                         "Run 09_extract_embeddings.py first.")

    blocks_wanted = [b.strip() for b in hc_blocks.split(",") if b.strip()]
    hcb = hc_keys_by_block(features) if blocks_wanted else {}
    hc_keys = [k for b in blocks_wanted for k in hcb.get(b, [])]

    metas, rows, y7, ym = [], [], [], []
    zero_emb = np.zeros(EMB_DIM, dtype=np.float32)
    zero_cus = np.zeros(cus_dim, dtype=np.float32)
    zero_ac = np.zeros(ACOUSTIC_DIM, dtype=np.float32)
    n_missing_passage = 0

    for (uid, sid), tasks in sorted(session_map.items()):
        reported = (tasks.get("freestyle", {}).get("reported_emotions")
                    or tasks.get("passage", {}).get("reported_emotions") or [])
        if not reported:
            continue
        dom = dominant_emotion(reported)
        if dom not in EMOTION_LABELS:
            continue

        f_rec = resolve_stem(tasks.get("freestyle", {}), features)
        p_rec = resolve_stem(tasks.get("passage", {}), features)
        if not f_rec and not p_rec:
            continue
        if not p_rec:
            n_missing_passage += 1

        blocks = []

        if want_e2v:
            f_emb = emb_of(f_rec) if f_rec else zero_emb
            p_emb = emb_of(p_rec) if p_rec else zero_emb
            f_lbl = label_onehot(f_rec.get("emotion2vec_label", "neutral")) if f_rec \
                else np.zeros(N_EMOTIONS, np.float32)
            p_lbl = label_onehot(p_rec.get("emotion2vec_label", "neutral")) if p_rec \
                else np.zeros(N_EMOTIONS, np.float32)
            blocks += [f_emb, f_lbl, p_lbl]
            if use_delta:
                d_emb = np.asarray(f_rec.get("embedding_delta", []), dtype=np.float32)
                if d_emb.size != EMB_DIM:
                    d_emb = f_emb - p_emb          # recompute if not precomputed
                blocks += [d_emb,
                           np.array([float(f_rec.get("embedding_cosine_to_passage", 0.0))],
                                    dtype=np.float32)]
            else:
                blocks.append(p_emb)

        if want_cus:
            f_c = np.asarray((f_rec or {}).get("custom_embedding", []), dtype=np.float32)
            p_c = np.asarray((p_rec or {}).get("custom_embedding", []), dtype=np.float32)
            f_c = f_c if f_c.size == cus_dim else zero_cus
            p_c = p_c if p_c.size == cus_dim else zero_cus
            probs = np.asarray((f_rec or {}).get("custom_probs7", []), dtype=np.float32)
            if probs.size != N_EMOTIONS:
                probs = np.zeros(N_EMOTIONS, dtype=np.float32)
            blocks += [f_c, probs]
            blocks.append(f_c - p_c if use_delta else p_c)

        f_ac = acoustic_vec(f_rec.get("acoustic_attrs", {})) if f_rec else zero_ac
        p_ac = acoustic_vec(p_rec.get("acoustic_attrs", {})) if p_rec else zero_ac
        blocks += [f_ac, p_ac]
        if use_delta:
            blocks.append(delta_vec(f_rec))

        if hc_keys:
            f_hc = _hc_vec(f_rec, hc_keys)
            blocks.append(f_hc)
            if use_delta:
                blocks.append(f_hc - _hc_vec(p_rec, hc_keys))

        metas.append({"user_id": uid, "session_id": sid,
                      "dominant": dom, "reported": reported})
        rows.append(np.concatenate(blocks))
        y7.append(EMOTION_LABELS.index(dom))
        ym.append(emotions_to_multilabel(reported))

    if not rows:
        raise SystemExit("[ERROR] No usable sessions. Check video_index / features paths.")

    if n_missing_passage:
        print(f"[WARN] {n_missing_passage} session(s) have no passage clip; "
              f"their passage and delta blocks are zero-filled.")
    if hc_keys:
        print(f"[INFO] Handcrafted families {blocks_wanted}: {len(hc_keys)} features"
              f"{' (+ delta twin)' if use_delta else ''}")

    X = np.stack(rows).astype(np.float32)
    groups = np.array([m["user_id"] for m in metas])
    return metas, X, np.array(y7, dtype=np.int64), np.stack(ym).astype(np.float32), groups


def print_distribution(y7: np.ndarray, groups: np.ndarray) -> None:
    print(f"\n  Sessions: {len(y7)} | Users: {len(set(groups))}")
    print("  Class distribution (dominant emotion):")
    counts = np.bincount(y7, minlength=N_EMOTIONS)
    for i, lbl in enumerate(EMOTION_LABELS):
        bar = "#" * int(40 * counts[i] / max(counts.max(), 1))
        print(f"    {lbl:<10} {counts[i]:4d}  {bar}")
    print("  Sessions per user:")
    for uid in sorted(set(groups)):
        print(f"    {uid:<24} {int((groups == uid).sum()):4d}")


def main():
    p = argparse.ArgumentParser(description="Audio-only session emotion recognition")
    p.add_argument("--cv", default="louo", choices=["kfold", "grouped", "louo", "all"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--features", type=Path, default=ROOT / "code/data/features.jsonl")
    p.add_argument("--video_index", type=Path,
                   default=ROOT / "firebase_data" / "video_index.json")
    p.add_argument("--out_dir", type=Path, default=ROOT / "code/data/results")
    p.add_argument("--no_delta", action="store_true",
                   help="Ablation: drop the passage-baseline delta block")
    p.add_argument("--encoder", default="emotion2vec",
                   choices=["emotion2vec", "custom", "both", "none"],
                   help="Which speech representation to classify from")
    p.add_argument("--hc_blocks", default="",
                   help="Handcrafted families to append: "
                        "mfcc,dmfcc,ddmfcc,logmel,spectral,lsf,prosody")
    args = p.parse_args()

    print(f"[INFO] Device: {DEVICE}")
    video_index = load_video_index(args.video_index)
    features = load_features(args.features)
    print(f"[INFO] {len(video_index)} index entries | {len(features)} feature records")

    use_delta = not args.no_delta
    metas, X, y7, ym, groups = build_dataset(
        video_index, features, use_delta=use_delta,
        encoder=args.encoder, hc_blocks=args.hc_blocks)
    tag = f"audio_{args.encoder}" + ("" if use_delta else "_nodelta")
    if args.hc_blocks:
        tag += "_hc"

    print(f"[INFO] Feature dim: {X.shape[1]}  (delta block: "
          f"{'ON' if use_delta else 'OFF'})")
    print_distribution(y7, groups)

    methods = ["kfold", "grouped", "louo"] if args.cv == "all" else [args.cv]
    all_results = {}

    for method in methods:
        res = run_cv(X, y7, ym, groups, method,
                     n_folds=args.folds, seed=args.seed, tag=tag)
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

        if "kfold" in all_results and "louo" in all_results:
            gap = (all_results["kfold"]["pooled"]["f1_macro7"]
                   - all_results["louo"]["pooled"]["f1_macro7"])
            print(f"\n  Leakage gap (kfold - louo, macro F1): {gap:+.4f}")
            print("  This is the amount standard K-Fold overstates performance by")
            print("  letting the model recognise users it has already seen. Report")
            print("  grouped or LOUO as the headline number, never kfold.")

    print(f"\n[INFO] All results -> {args.out_dir}")


if __name__ == "__main__":
    main()
