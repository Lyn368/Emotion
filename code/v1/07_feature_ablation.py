"""
07_feature_ablation.py
═══════════════════════════════════════════════════════════════════════════════
Settle the "which features help?" question by measurement instead of argument.

Builds the audio feature vector from selectable BLOCKS, runs the same CV
protocol over each configuration, and reports paired significance tests
between them. Every configuration sees identical sessions in identical order,
so the comparisons are valid.

    python 07_feature_ablation.py --cv louo
    python 07_feature_ablation.py --cv louo --configs e2v,hc,e2v_pca,delta_only
    python 07_feature_ablation.py --cv grouped --list

BLOCKS
─────────────────────────────────────────────────────────────────────────────
  e2v_free    freestyle emotion2vec embedding            768
  e2v_pass    passage emotion2vec embedding              768
  e2v_delta   freestyle - passage embedding              768
  e2v_pca     PCA of [freestyle, delta] to n components   32 (configurable)
  hc_free     handcrafted (eGeMAPS or librosa), freestyle ~88
  hc_delta    handcrafted, freestyle - passage           ~88
  attrs       discretised + raw pitch/energy/rate         12
  scalar_d    Δpitch Δenergy Δrate + 2 ratios              5
  e2v_label   emotion2vec label one-hot, both clips       14
  cosine      cosine(freestyle, passage)                   1

WHY [e2v_free, e2v_pass, e2v_delta] IS A MISTAKE
─────────────────────────────────────────────────────────────────────────────
delta = free - pass, so the three blocks together span exactly the same linear
space as the first two. The delta block adds 768 dimensions and ~200k
first-layer parameters while giving a linear layer nothing it could not
already reach. The current pipeline ships this configuration; `full` reproduces
it so the cost is measurable, and `no_pass` tests the fix.

PCA IS FITTED INSIDE EACH FOLD
─────────────────────────────────────────────────────────────────────────────
Fitting PCA on the whole corpus before splitting would leak test-set structure
into the projection. `e2v_pca` therefore fits on training rows only, per fold,
and applies the fitted transform to the held-out rows.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from common import (
    EMOTION_LABELS, N_EMOTIONS, DEVICE,
    load_video_index, load_features, resolve_stem,
    dominant_emotion, emotions_to_multilabel,
    make_splits, make_inner_split, train_fold_explicit, evaluate_fold,
    IDX7_TO_IDX3,
)
from stats import (
    cluster_bootstrap_ci, paired_bootstrap_test,
    permutation_test_predictions, macro_f1, accuracy, unweighted_accuracy,
)
from handcrafted import feature_vector as hc_vector

ROOT = Path("/Users/ytinglin/Downloads/ProjectCE903/25-26_CE903-SP_team01/MLmodel/data")
EMB_DIM = 768
LEVEL_MAP = {"Low": 0.0, "Medium": 0.5, "High": 1.0}

# Named configurations. Each is a list of block names.
CONFIGS: dict[str, list[str]] = {
    # Current shipping configuration, with the redundant passage block.
    "full":       ["e2v_free", "e2v_pass", "e2v_delta", "attrs", "scalar_d",
                   "e2v_label", "cosine"],
    # Same information, without the linear redundancy.
    "no_pass":    ["e2v_free", "e2v_delta", "attrs", "scalar_d", "e2v_label", "cosine"],
    # Embeddings only, no passage reference at all -> the ablation that isolates
    # the paper's central contribution.
    "e2v":        ["e2v_free", "attrs", "e2v_label"],
    # Compact handcrafted set instead of embeddings.
    "hc":         ["hc_free", "attrs"],
    "hc_delta":   ["hc_free", "hc_delta", "attrs", "scalar_d"],
    # Dimensionality-reduced embeddings.
    "e2v_pca":    ["e2v_pca", "attrs", "scalar_d", "e2v_label", "cosine"],
    "e2v_pca_hc": ["e2v_pca", "hc_free", "hc_delta", "attrs", "scalar_d"],
    # Minimal: is the scalar passage reference alone informative?
    "scalars":    ["attrs", "scalar_d", "cosine"],
    "delta_only": ["e2v_delta", "scalar_d"],
}

PCA_COMPONENTS = 32


# ─────────────────────────────────────────────────────────────────────────────
# Block builders
# ─────────────────────────────────────────────────────────────────────────────
def _emb(rec: dict, key: str = "emotion2vec_embedding") -> np.ndarray:
    a = np.asarray(rec.get(key, []), dtype=np.float32)
    return a if a.size == EMB_DIM else np.zeros(EMB_DIM, dtype=np.float32)


def _attrs(rec: dict) -> np.ndarray:
    a = rec.get("acoustic_attrs", {})
    return np.array([
        LEVEL_MAP.get(a.get("pitch_level", "Medium"), 0.5),
        LEVEL_MAP.get(a.get("energy_level", "Medium"), 0.5),
        LEVEL_MAP.get(a.get("speaking_rate_level", "Medium"), 0.5),
        float(a.get("pitch_hz", 0.0)),
        float(a.get("energy_lufs", 0.0)),
        float(a.get("speaking_rate_wps", 0.0)),
    ], dtype=np.float32)


def _label_onehot(lbl: str) -> np.ndarray:
    v = np.zeros(N_EMOTIONS, dtype=np.float32)
    if lbl in EMOTION_LABELS:
        v[EMOTION_LABELS.index(lbl)] = 1.0
    return v


def build_blocks(video_index: dict, features: dict, hc_keys: list[str] | None):
    """
    Build every block once, for every session. Configurations then select
    among them, so no session is ever rebuilt differently between configs.
    """
    session_map: dict[tuple, dict] = defaultdict(dict)
    for (uid, sid, vtype), info in video_index.items():
        session_map[(uid, sid)][vtype] = info

    blocks: dict[str, list] = defaultdict(list)
    metas, y7, ym = [], [], []
    zero_emb = np.zeros(EMB_DIM, dtype=np.float32)
    n_hc = len(hc_keys) if hc_keys else 0

    for (uid, sid), tasks in sorted(session_map.items()):
        reported = (tasks.get("freestyle", {}).get("reported_emotions")
                    or tasks.get("passage", {}).get("reported_emotions") or [])
        if not reported:
            continue
        dom = dominant_emotion(reported)
        if dom not in EMOTION_LABELS:
            continue

        f = resolve_stem(tasks.get("freestyle", {}), features)
        p = resolve_stem(tasks.get("passage", {}), features)
        if not f:
            continue

        f_emb = _emb(f)
        p_emb = _emb(p) if p else zero_emb

        d_emb = np.asarray(f.get("embedding_delta", []), dtype=np.float32)
        if d_emb.size != EMB_DIM:
            d_emb = f_emb - p_emb

        blocks["e2v_free"].append(f_emb)
        blocks["e2v_pass"].append(p_emb)
        blocks["e2v_delta"].append(d_emb)
        blocks["attrs"].append(np.concatenate([_attrs(f), _attrs(p) if p else np.zeros(6, np.float32)]))

        dv = f.get("delta_vs_passage", {})
        blocks["scalar_d"].append(np.array([
            float(dv.get("d_pitch_hz", 0.0)), float(dv.get("d_energy_lufs", 0.0)),
            float(dv.get("d_rate_wps", 0.0)), float(dv.get("r_pitch", 1.0)),
            float(dv.get("r_rate", 1.0))], dtype=np.float32))

        blocks["e2v_label"].append(np.concatenate([
            _label_onehot(f.get("emotion2vec_label", "neutral")),
            _label_onehot(p.get("emotion2vec_label", "neutral")) if p
            else np.zeros(N_EMOTIONS, np.float32)]))

        blocks["cosine"].append(
            np.array([float(f.get("embedding_cosine_to_passage", 0.0))], dtype=np.float32))

        if hc_keys:
            hf, _ = hc_vector(f, hc_keys)
            hp, _ = hc_vector(p, hc_keys) if p else (np.zeros(n_hc, np.float32), None)
            blocks["hc_free"].append(hf)
            blocks["hc_delta"].append(hf - hp)

        metas.append({"user_id": uid, "session_id": sid})
        y7.append(EMOTION_LABELS.index(dom))
        ym.append(emotions_to_multilabel(reported))

    if not metas:
        raise SystemExit("[ERROR] No usable sessions.")

    out = {k: np.stack(v).astype(np.float32) for k, v in blocks.items()}
    return (metas, out, np.array(y7, dtype=np.int64),
            np.stack(ym).astype(np.float32),
            np.array([m["user_id"] for m in metas]))


def assemble(blocks: dict, names: list[str]) -> np.ndarray | None:
    """Concatenate the named blocks. Returns None if any is unavailable."""
    parts = []
    for n in names:
        if n == "e2v_pca":
            continue                       # handled per fold
        if n not in blocks:
            return None
        parts.append(blocks[n])
    return np.hstack(parts).astype(np.float32) if parts else None


# ─────────────────────────────────────────────────────────────────────────────
# Run one configuration
# ─────────────────────────────────────────────────────────────────────────────
def run_config(
    name: str, block_names: list[str], blocks: dict,
    y7: np.ndarray, ym: np.ndarray, groups: np.ndarray,
    method: str, n_folds: int, seed: int,
) -> dict | None:
    X = assemble(blocks, block_names)
    uses_pca = "e2v_pca" in block_names

    if X is None and not uses_pca:
        print(f"  [SKIP] {name}: a required block is missing "
              f"(handcrafted features not extracted?)")
        return None

    pooled_true, pooled_pred, pooled_groups, dims = [], [], [], []

    for tr, te, _ in make_splits(y7, groups, method, n_folds, seed):
        if len(np.unique(y7[tr])) < 2:
            continue

        if uses_pca:
            # Fit PCA on training rows only, inside the fold.
            src = np.hstack([blocks["e2v_free"], blocks["e2v_delta"]])
            k = min(PCA_COMPONENTS, len(tr) - 1, src.shape[1])
            pca = PCA(n_components=k, random_state=seed).fit(src[tr])
            proj = pca.transform(src).astype(np.float32)
            rest = assemble(blocks, [b for b in block_names if b != "e2v_pca"])
            Xf = np.hstack([proj, rest]).astype(np.float32) if rest is not None else proj
        else:
            Xf = X

        dims.append(Xf.shape[1])

        in_tr, in_val = make_inner_split(y7[tr], groups[tr], method, seed=seed)
        rtr, rval = tr[in_tr], tr[in_val]

        model, norm = train_fold_explicit(
            Xf[rtr], y7[rtr], ym[rtr], Xf[rval], y7[rval], ym[rval], seed=seed)
        res = evaluate_fold(model, Xf[te], y7[te], ym[te], norm)

        pooled_true.append(y7[te])
        pooled_pred.append(np.array(res["y7_pred"]))
        pooled_groups.append(groups[te])

    if not pooled_true:
        return None

    yt = np.concatenate(pooled_true)
    yp = np.concatenate(pooled_pred)
    gp = np.concatenate(pooled_groups)
    t3, p3 = IDX7_TO_IDX3[yt], IDX7_TO_IDX3[yp]

    ci = cluster_bootstrap_ci(yt, yp, gp, "macro_f1", seed=seed)
    perm = permutation_test_predictions(yt, yp, "macro_f1", seed=seed)

    print(f"  {name:<14} dim={int(np.mean(dims)):>5}  "
          f"acc7={accuracy(yt, yp):.3f}  "
          f"mF1={ci['point']:.3f} [{ci['ci_low']:.3f},{ci['ci_high']:.3f}]  "
          f"UA={unweighted_accuracy(yt, yp):.3f}  "
          f"mF1_3={macro_f1(t3, p3):.3f}  p={perm['p_value']:.3f}")

    return {
        "config": name, "blocks": block_names, "dim": int(np.mean(dims)),
        "acc7": accuracy(yt, yp), "macro_f1": ci["point"],
        "ci": [ci["ci_low"], ci["ci_high"]],
        "ua": unweighted_accuracy(yt, yp),
        "macro_f1_3class": macro_f1(t3, p3),
        "permutation_p": perm["p_value"],
        "y_true": yt.tolist(), "y_pred": yp.tolist(),
        "groups": gp.tolist(),
    }


def main():
    p = argparse.ArgumentParser(description="Audio feature-block ablation")
    p.add_argument("--cv", default="louo", choices=["kfold", "grouped", "louo"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--configs", default="full,no_pass,e2v,hc,e2v_pca,scalars",
                   help=f"Comma-separated. Available: {','.join(CONFIGS)}")
    p.add_argument("--list", action="store_true", help="List configurations and exit")
    p.add_argument("--features", type=Path, default=ROOT / "features.jsonl")
    p.add_argument("--video_index", type=Path,
                   default=ROOT / "firebase_data" / "video_index.json")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "feature_ablation.json")
    args = p.parse_args()

    if args.list:
        for k, v in CONFIGS.items():
            print(f"  {k:<14} {' + '.join(v)}")
        return

    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise SystemExit(f"[ERROR] Unknown config(s): {unknown}. "
                         f"Available: {list(CONFIGS)}")

    video_index = load_video_index(args.video_index)
    features = load_features(args.features)

    # Determine the handcrafted key list once, from any record that has it.
    hc_keys = None
    for rec in features.values():
        hc = rec.get("handcrafted", {}).get("features")
        if hc:
            hc_keys = sorted(hc.keys())
            hc_set = rec["handcrafted"].get("set", "?")
            print(f"[INFO] Handcrafted features: {hc_set}, {len(hc_keys)} dims")
            break
    if hc_keys is None:
        print("[INFO] No handcrafted features in features.jsonl. "
              "hc_* configs will be skipped.")
        print("       Re-run 03_feature_extraction.py with handcrafted extraction on.")

    metas, blocks, y7, ym, groups = build_blocks(video_index, features, hc_keys)
    print(f"[INFO] {len(y7)} sessions | {len(set(groups))} speakers | CV={args.cv}")
    print(f"[INFO] Block dims: "
          + ", ".join(f"{k}={v.shape[1]}" for k, v in sorted(blocks.items())))

    print(f"\n{'=' * 78}")
    print(f"  FEATURE ABLATION — {args.cv}   (mF1 with 95% speaker-bootstrap CI)")
    print(f"{'=' * 78}")

    results = {}
    for n in names:
        r = run_config(n, CONFIGS[n], blocks, y7, ym, groups,
                       args.cv, args.folds, args.seed)
        if r:
            results[n] = r

    # ── Pairwise significance ────────────────────────────────────────────────
    if len(results) > 1:
        print(f"\n  Paired speaker-bootstrap tests (macro F1)")
        print(f"  {'-' * 74}")
        print(f"  {'A vs B':<30} {'ΔmF1':>8} {'95% CI':>20} {'p':>8}")
        print(f"  {'-' * 74}")
        for a, b in itertools.combinations(results, 2):
            ra, rb = results[a], results[b]
            t = paired_bootstrap_test(
                np.array(ra["y_true"]), np.array(ra["y_pred"]),
                np.array(rb["y_pred"]), np.array(ra["groups"]),
                "macro_f1", seed=args.seed)
            star = " *" if t["significant_05"] else ""
            print(f"  {a + ' vs ' + b:<30} {t['delta']:>+8.4f} "
                  f"[{t['delta_ci_low']:>+.3f},{t['delta_ci_high']:>+.3f}]"
                  f"{t['p_value']:>9.4f}{star}")
        print(f"\n  * p < 0.05. Delta is A minus B; positive means A is better.")
        print(f"  A difference whose CI spans zero is not evidence of anything —")
        print(f"  report it as no detected difference, not as a small improvement.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"cv": args.cv, "n_sessions": int(len(y7)),
                   "n_speakers": int(len(set(groups))),
                   "results": results}, f, indent=2)
    print(f"\n[INFO] Results -> {args.out}")


if __name__ == "__main__":
    main()
