"""
11_feature_selection.py
═══════════════════════════════════════════════════════════════════════════════
Answer "which of these features actually help ME" by measurement, on YOUR data,
rather than by taking a feature list from a paper written on a different corpus.

    # what each family is worth on its own, and what the full set loses without it
    python 11_feature_selection.py --cv grouped --mode single,loo

    # build the set up greedily, then rank the individual features inside it
    python 11_feature_selection.py --cv grouped --mode greedy,perfeature

    # same thing on the public corpus
    python 11_feature_selection.py --data data_public --cv grouped --mode single,loo

FOUR MODES
─────────────────────────────────────────────────────────────────────────────
  single      each block alone. Answers "is there signal in MFCCs at all?"
  loo         full set minus one block. Answers "does this block add anything
              the others do not already carry?" A block can score well alone
              and still be worthless here — that means it is redundant, which
              is the usual outcome for correlated acoustic families.
  greedy      forward selection: start empty, add whichever block gains the
              most macro F1, stop when nothing gains. Prone to overfitting the
              CV, so treat the result as a hypothesis and confirm it with a
              paired bootstrap, which this prints.
  perfeature  inside the winning set, rank individual features by permutation
              importance (shuffle one column in the test fold, measure the
              drop) and by mutual information. Permutation importance is the
              one to trust: MI is univariate and ignores redundancy.

WHY THIS MATTERS HERE SPECIFICALLY
─────────────────────────────────────────────────────────────────────────────
The audio branch feeds thousands of dimensions into a few hundred sessions.
At that ratio, extra features do not add information, they add variance. The
useful operation is subtraction. Expect the answer to be a small set — and a
small set is also what makes the paper's feature table readable.

Every configuration is evaluated on identical sessions in identical order, so
the paired bootstrap between them is valid.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (
    EMOTION_LABELS, N_EMOTIONS, DEVICE,
    load_video_index, load_features, resolve_stem,
    dominant_emotion, emotions_to_multilabel,
    make_splits, make_inner_split, train_fold_explicit, evaluate_fold,
)
from stats import paired_bootstrap_test, cluster_bootstrap_ci
from features_audio import block_of


def base_name(n: str) -> str:
    """Strip the delta prefix without mangling dmfcc/ddmfcc names."""
    return n[2:] if n.startswith("d_") else n

LEVEL_MAP = {"Low": 0.0, "Medium": 0.5, "High": 1.0}
HC_BLOCKS = ["mfcc", "dmfcc", "ddmfcc", "logmel", "spectral", "lsf", "prosody", "egemaps"]


# ─────────────────────────────────────────────────────────────────────────────
# Block construction
# ─────────────────────────────────────────────────────────────────────────────
def _emb(rec: dict, key: str, dim: int) -> np.ndarray:
    v = np.asarray(rec.get(key, []), dtype=np.float32) if rec else np.zeros(0, np.float32)
    return v if v.size == dim else np.zeros(dim, dtype=np.float32)


def _attrs(rec: dict) -> np.ndarray:
    a = (rec or {}).get("acoustic_attrs", {})
    return np.array([
        LEVEL_MAP.get(a.get("pitch_level", "Medium"), 0.5),
        LEVEL_MAP.get(a.get("energy_level", "Medium"), 0.5),
        LEVEL_MAP.get(a.get("speaking_rate_level", "Medium"), 0.5),
        float(a.get("pitch_hz", 0.0)),
        float(a.get("energy_lufs", 0.0)),
        float(a.get("speaking_rate_wps", 0.0)),
    ], dtype=np.float32)


def _scalar_delta(rec: dict) -> np.ndarray:
    d = (rec or {}).get("delta_vs_passage", {})
    return np.array([
        float(d.get("d_pitch_hz", 0.0)), float(d.get("d_energy_lufs", 0.0)),
        float(d.get("d_rate_wps", 0.0)), float(d.get("r_pitch", 1.0)),
        float(d.get("r_rate", 1.0)),
    ], dtype=np.float32)


def hc_keys_by_block(features: dict) -> dict[str, list[str]]:
    common = None
    for r in features.values():
        f = r.get("handcrafted", {}).get("features", {})
        if not f:
            continue
        common = set(f) if common is None else (common & set(f))
    out: dict[str, list[str]] = defaultdict(list)
    for k in sorted(common or []):
        out[block_of(k)].append(k)
    return dict(out)


def build(video_index: dict, features: dict):
    """Return (metas, blocks, feature_names, y7, ym, groups)."""
    by_session = defaultdict(dict)
    for (uid, sid, vtype), info in video_index.items():
        by_session[(uid, sid)][vtype] = info

    hcb = hc_keys_by_block(features)
    e2v_dim = max((len(r.get("emotion2vec_embedding", [])) for r in features.values()),
                  default=0)
    cus_dim = max((len(r.get("custom_embedding", [])) for r in features.values()),
                  default=0)

    metas, y7, ym = [], [], []
    acc: dict[str, list] = defaultdict(list)

    for (uid, sid), tasks in sorted(by_session.items()):
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

        if e2v_dim:
            fe, pe = _emb(f, "emotion2vec_embedding", e2v_dim), _emb(p, "emotion2vec_embedding", e2v_dim)
            acc["e2v"].append(fe)
            acc["e2v_delta"].append(fe - pe)
        if cus_dim:
            fc, pc = _emb(f, "custom_embedding", cus_dim), _emb(p, "custom_embedding", cus_dim)
            acc["custom"].append(fc)
            acc["custom_delta"].append(fc - pc)
            pr = np.asarray(f.get("custom_probs7", []), dtype=np.float32)
            acc["custom_probs"].append(pr if pr.size == N_EMOTIONS
                                       else np.zeros(N_EMOTIONS, np.float32))

        acc["attrs"].append(np.concatenate([_attrs(f), _attrs(p)]))
        acc["scalar_d"].append(_scalar_delta(f))

        for b, keys in hcb.items():
            fv = np.array([float(f.get("handcrafted", {}).get("features", {}).get(k, 0.0))
                           for k in keys], dtype=np.float32)
            pv = np.array([float((p or {}).get("handcrafted", {}).get("features", {}).get(k, 0.0))
                           for k in keys], dtype=np.float32)
            acc[b].append(np.nan_to_num(fv))
            acc[f"{b}_delta"].append(np.nan_to_num(fv - pv))

        metas.append({"user_id": uid, "session_id": sid, "dominant": dom})
        y7.append(EMOTION_LABELS.index(dom))
        ym.append(emotions_to_multilabel(reported))

    if not metas:
        raise SystemExit("[ERROR] No usable sessions.")

    blocks = {k: np.stack(v).astype(np.float32) for k, v in acc.items() if v}

    names: dict[str, list[str]] = {}
    for b, keys in hcb.items():
        names[b] = list(keys)
        names[f"{b}_delta"] = [f"d_{k}" for k in keys]
    names["attrs"] = ["f_pitch_lvl", "f_energy_lvl", "f_rate_lvl", "f_pitch_hz",
                      "f_energy_lufs", "f_rate_wps", "p_pitch_lvl", "p_energy_lvl",
                      "p_rate_lvl", "p_pitch_hz", "p_energy_lufs", "p_rate_wps"]
    names["scalar_d"] = ["d_pitch_hz", "d_energy_lufs", "d_rate_wps", "r_pitch", "r_rate"]
    if e2v_dim:
        names["e2v"] = [f"e2v{i}" for i in range(e2v_dim)]
        names["e2v_delta"] = [f"de2v{i}" for i in range(e2v_dim)]
    if cus_dim:
        names["custom"] = [f"cus{i}" for i in range(cus_dim)]
        names["custom_delta"] = [f"dcus{i}" for i in range(cus_dim)]
        names["custom_probs"] = [f"cusp_{l}" for l in EMOTION_LABELS]

    return metas, blocks, names, np.array(y7, np.int64), np.stack(ym).astype(np.float32), \
        np.array([m["user_id"] for m in metas])


def assemble(blocks: dict, chosen: list[str]) -> np.ndarray:
    parts = [blocks[b] for b in chosen if b in blocks]
    if not parts:
        raise ValueError("empty configuration")
    return np.hstack(parts).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CV evaluation of one configuration
# ─────────────────────────────────────────────────────────────────────────────
def run_config(X, y7, ym, groups, method, n_folds, seed, permute_cols=None):
    """
    Returns (pooled_pred, pooled_true, pooled_groups, order, perm_importance).

    `permute_cols`: list of column indices to measure by shuffling that column
    in the test fold. Importance = macro F1 drop, averaged over folds.
    """
    from sklearn.metrics import f1_score

    preds, trues, gs, order = [], [], [], []
    drops = defaultdict(list)

    for tr, te, _ in make_splits(y7, groups, method, n_folds, seed):
        if len(np.unique(y7[tr])) < 2:
            continue
        in_tr, in_val = make_inner_split(y7[tr], groups[tr], method, seed=seed)
        model, norm = train_fold_explicit(
            X[tr[in_tr]], y7[tr[in_tr]], ym[tr[in_tr]],
            X[tr[in_val]], y7[tr[in_val]], ym[tr[in_val]],
            device=DEVICE, seed=seed)
        res = evaluate_fold(model, X[te], y7[te], ym[te], norm, device=DEVICE)
        base_pred = np.array(res["y7_pred"])
        preds.append(base_pred)
        trues.append(y7[te])
        gs.append(groups[te])
        order.extend(te.tolist())

        if permute_cols:
            base = f1_score(y7[te], base_pred, average="macro", zero_division=0)
            rng = np.random.RandomState(seed)
            for col in permute_cols:
                Xp = X[te].copy()
                Xp[:, col] = Xp[rng.permutation(len(Xp)), col]
                r = evaluate_fold(model, Xp, y7[te], ym[te], norm, device=DEVICE)
                drops[col].append(base - f1_score(y7[te], r["y7_pred"],
                                                  average="macro", zero_division=0))

    return (np.concatenate(preds), np.concatenate(trues), np.concatenate(gs),
            np.array(order), {c: float(np.mean(v)) for c, v in drops.items()})


def score(y, p) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    ua = [float((p[y == c] == c).mean()) for c in np.unique(y) if (y == c).sum()]
    return {
        "accuracy": float(accuracy_score(y, p)),
        "ua": float(np.mean(ua)) if ua else 0.0,
        "f1_macro": float(f1_score(y, p, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y, p, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y, p, average="weighted", zero_division=0)),
    }


def main():
    p = argparse.ArgumentParser(description="Which features actually help?")
    p.add_argument("--data", type=Path,
                   default=Path(r"C:\Users\2022g\Downloads\RAVDESS_IEMOCAP_CREMAD"))
    p.add_argument("--features", type=Path, default=None)
    p.add_argument("--video_index", type=Path, default=None)
    p.add_argument("--cv", default="grouped", choices=["kfold", "grouped", "louo"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", default="single,loo")
    p.add_argument("--blocks", default="",
                   help="Restrict the candidate pool, comma separated")
    p.add_argument("--greedy_rounds", type=int, default=6)
    p.add_argument("--top_k", type=int, default=40, help="Features to list in perfeature")
    p.add_argument("--perm_budget", type=int, default=250,
                   help="Max columns to permute (highest-MI columns are chosen)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    feat_path = args.features or args.data / "code/data/features.jsonl"
    idx_path = args.video_index or args.data / "firebase_data" / "video_index.json"
    modes = [m.strip() for m in args.mode.split(",") if m.strip()]

    video_index = load_video_index(idx_path)
    features = load_features(feat_path)
    metas, blocks, names, y7, ym, groups = build(video_index, features)

    pool = [b for b in blocks if not args.blocks] or \
           [b.strip() for b in args.blocks.split(",") if b.strip() in blocks]
    pool = sorted(pool)

    print(f"[INFO] {len(metas)} sessions | {len(set(groups))} speakers | CV={args.cv}")
    print("[INFO] candidate blocks:")
    for b in pool:
        print(f"    {b:<16} {blocks[b].shape[1]:5d} dims")

    results: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}
    ref_order = None

    def evaluate(tag: str, chosen: list[str], permute_cols=None):
        nonlocal ref_order
        X = assemble(blocks, chosen)
        p_, t_, g_, order, perm = run_config(X, y7, ym, groups, args.cv,
                                             args.folds, args.seed, permute_cols)
        # Re-sort pooled predictions into a fixed session order so every
        # configuration lines up for the paired bootstrap.
        srt = np.argsort(order)
        p_, t_, g_ = p_[srt], t_[srt], g_[srt]
        if ref_order is None:
            ref_order = order[srt]
        s = score(t_, p_)
        results[tag] = {"blocks": chosen, "dims": int(X.shape[1]), **s}
        preds[tag] = p_
        print(f"  {tag:<28} dims={X.shape[1]:<6} acc={s['accuracy']:.4f} "
              f"UA={s['ua']:.4f} mF1={s['f1_macro']:.4f} "
              f"microF1={s['f1_micro']:.4f} wF1={s['f1_weighted']:.4f}")
        return s["f1_macro"], t_, g_, perm

    y_true_sorted = g_sorted = None

    # ── single ───────────────────────────────────────────────────────────────
    if "single" in modes:
        print(f"\n{'=' * 74}\n  EACH BLOCK ALONE\n{'=' * 74}")
        for b in pool:
            _, y_true_sorted, g_sorted, _ = evaluate(f"only:{b}", [b])

    # ── leave-one-out ────────────────────────────────────────────────────────
    if "loo" in modes:
        print(f"\n{'=' * 74}\n  FULL SET, MINUS ONE BLOCK\n{'=' * 74}")
        full, y_true_sorted, g_sorted, _ = evaluate("full", pool)
        for b in pool:
            rest = [x for x in pool if x != b]
            if not rest:
                continue
            s, y_true_sorted, g_sorted, _ = evaluate(f"full-minus:{b}", rest)
            print(f"      dropping {b:<16} changes macro F1 by {full - s:+.4f}"
                  f"   ({'it carries something unique' if full - s > 0.01 else 'redundant or useless'})")

    # ── greedy forward selection ─────────────────────────────────────────────
    chosen: list[str] = []
    if "greedy" in modes:
        print(f"\n{'=' * 74}\n  GREEDY FORWARD SELECTION\n{'=' * 74}")
        best_so_far = -1.0
        remaining = pool[:]
        for rnd in range(1, args.greedy_rounds + 1):
            gains = []
            for b in remaining:
                s, y_true_sorted, g_sorted, _ = evaluate(
                    f"r{rnd}:{'+'.join(chosen + [b])[:60]}", chosen + [b])
                gains.append((s, b))
            gains.sort(reverse=True)
            best, blk = gains[0]
            if best <= best_so_far + 1e-4:
                print(f"  round {rnd}: no block improves on {best_so_far:.4f}. Stop.")
                break
            chosen.append(blk)
            remaining.remove(blk)
            best_so_far = best
            print(f"  round {rnd}: + {blk}  -> macro F1 {best:.4f}")
        print(f"\n  selected blocks: {chosen or '(none)'}")

    # ── per-feature ranking ──────────────────────────────────────────────────
    perfeature = {}
    if "perfeature" in modes:
        target = chosen or pool
        print(f"\n{'=' * 74}\n  PER-FEATURE IMPORTANCE  (blocks: {target})\n{'=' * 74}")
        X = assemble(blocks, target)
        # Track the originating block per column instead of guessing it from
        # the feature name: an embedding dimension has no family prefix, and
        # falling through to a default would file it under the wrong family.
        col_names, col_block = [], []
        for b in target:
            ns = names.get(b, [f"{b}{i}" for i in range(blocks[b].shape[1])])
            col_names.extend(ns)
            col_block.extend([b] * len(ns))

        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(X, y7, random_state=args.seed)
        cols = list(np.argsort(mi)[::-1][:args.perm_budget])
        print(f"  permuting the {len(cols)} highest-MI columns of {X.shape[1]}")

        _, y_true_sorted, g_sorted, perm = evaluate(
            "perfeature_base", target, permute_cols=cols)

        ranked = sorted(perm.items(), key=lambda kv: kv[1], reverse=True)
        print(f"\n  {'feature':<34}{'block':<16}{'ΔmacroF1':>10}{'MI':>10}")
        print("  " + "-" * 68)
        for col, drop in ranked[:args.top_k]:
            nm = col_names[col] if col < len(col_names) else f"col{col}"
            bl = col_block[col] if col < len(col_block) else "?"
            print(f"  {nm[:33]:<34}{bl:<16}{drop:>+10.4f}{mi[col]:>10.4f}")

        by_block = defaultdict(float)
        for col, drop in perm.items():
            bl = col_block[col] if col < len(col_block) else "?"
            by_block[bl] += max(drop, 0.0)
        print("\n  summed positive importance by family:")
        for b, v in sorted(by_block.items(), key=lambda kv: -kv[1]):
            print(f"    {b:<16} {v:+.4f}")

        perfeature = {
            "blocks_used": target,
            "top_features": [
                {"name": col_names[c] if c < len(col_names) else f"col{c}",
                 "block": col_block[c] if c < len(col_block) else "?",
                 "perm_drop_macro_f1": d, "mutual_info": float(mi[c])}
                for c, d in ranked[:args.top_k]],
            "importance_by_family": dict(by_block),
        }

    # ── significance between the strongest configurations ────────────────────
    if len(preds) > 1 and y_true_sorted is not None:
        ranked_cfg = sorted(results, key=lambda k: results[k]["f1_macro"], reverse=True)
        top = ranked_cfg[:min(4, len(ranked_cfg))]
        print(f"\n{'=' * 74}\n  PAIRED BOOTSTRAP — best configurations\n{'=' * 74}")
        base = top[0]
        ci = cluster_bootstrap_ci(y_true_sorted, preds[base], g_sorted,
                                  metric="macro_f1", seed=args.seed)
        print(f"  {base}: macro F1 {ci['point']:.4f} "
              f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")
        for other in top[1:]:
            t = paired_bootstrap_test(y_true_sorted, preds[base], preds[other],
                                      g_sorted, metric="macro_f1", seed=args.seed)
            print(f"  vs {other:<28} Δ={t['delta']:+.4f} "
                  f"[{t['delta_ci_low']:+.3f},{t['delta_ci_high']:+.3f}] "
                  f"p={t['p_value']:.4f}"
                  f"{'  significant' if t['significant_05'] else ''}")

    out = args.out or args.data / "results" / "feature_selection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "cv": args.cv, "n_sessions": len(metas), "n_speakers": len(set(groups)),
        "configurations": results,
        "greedy_selection": chosen,
        "perfeature": perfeature,
    }, indent=2))
    print(f"\n[INFO] Results -> {out}")


if __name__ == "__main__":
    main()
