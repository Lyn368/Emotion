"""
10_benchmark_public.py
═══════════════════════════════════════════════════════════════════════════════
The comparison table for the paper: your encoder against emotion2vec and
against the handcrafted bank, on the 20% of speakers that 00 reserved and that
08 never saw.

    python 10_benchmark_public.py --data data_public \
        --systems custom_zeroshot,custom_mlp,e2v_mlp,hc_mlp,custom_mlp_delta,e2v_mlp_delta

SYSTEMS
─────────────────────────────────────────────────────────────────────────────
  custom_zeroshot   argmax of your encoder's own 7-class head. No further
                    training. This is the number directly comparable to a
                    published end-to-end SER model.
  e2v_zeroshot      emotion2vec's own label prediction. Free baseline.
  custom_mlp        your encoder's embedding, frozen, into the shared MLP.
  e2v_mlp           emotion2vec embedding, frozen, into the same MLP.
  hc_mlp            the handcrafted librosa bank into the same MLP.
  *_delta           same, plus the neutral-reference delta block.

The *_mlp systems are the FAIR encoder comparison: same classifier, same
sessions, same split, only the representation changes. custom_zeroshot is not
comparable to e2v_mlp — one is end-to-end, the other is frozen features plus a
trained head — so do not put them in the same column of a table without
saying which is which.

METRICS
─────────────────────────────────────────────────────────────────────────────
accuracy (= WA), UA (mean per-class recall, = macro recall), macro F1,
micro F1, weighted F1, each with a speaker-clustered bootstrap CI.

Note for the write-up: in a single-label multiclass task micro F1 equals
accuracy exactly — same quantity, two names. Reporting both is fine, but say
so in a footnote or a reviewer will flag it as padding.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from common import (
    EMOTION_LABELS, N_EMOTIONS, THREE_CLASS_LABELS, IDX7_TO_IDX3, DEVICE,
    emotions_to_multilabel, make_inner_split, train_fold_explicit, evaluate_fold,
)
from stats import cluster_bootstrap_ci, paired_bootstrap_test

DEFAULT_SYSTEMS = "custom_zeroshot,e2v_zeroshot,custom_mlp,e2v_mlp,hc_mlp"


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_index(path: Path) -> dict:
    """video_index.json with the corpus/split fields 00 writes."""
    entries = json.loads(Path(path).read_text())
    out = {}
    for e in entries:
        out[(e["user_id"], e["session_id"], e.get("video_type", "").lower())] = e
    return out


def load_features(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r.get("stem") or Path(r["file"]).stem] = r
    return out


def stem_for(entry: dict) -> str:
    return entry.get("video_id", "").replace("-webm", "")


def hc_common_keys(features: dict) -> list[str]:
    keys = None
    for r in features.values():
        f = r.get("handcrafted", {}).get("features", {})
        if not f:
            continue
        keys = set(f) if keys is None else (keys & set(f))
    return sorted(keys or [])


def build_sessions(index: dict, features: dict) -> list[dict]:
    """One row per session, carrying both clips' feature records."""
    by_session = defaultdict(dict)
    for (uid, sid, vtype), e in index.items():
        by_session[(uid, sid)][vtype] = e

    rows = []
    for (uid, sid), tasks in sorted(by_session.items()):
        fs_e, pa_e = tasks.get("freestyle"), tasks.get("passage")
        if fs_e is None:
            continue
        fs = features.get(stem_for(fs_e))
        if not fs:
            continue
        pa = features.get(stem_for(pa_e)) if pa_e else None
        label = (fs_e.get("reported_emotion") or "").strip().lower()
        if label not in EMOTION_LABELS:
            continue
        rows.append({
            "user_id": uid, "session_id": sid,
            "corpus": fs_e.get("corpus", "unknown"),
            "split": fs_e.get("split", "train"),
            "y7": EMOTION_LABELS.index(label),
            "free": fs, "pass": pa,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Feature matrices
# ─────────────────────────────────────────────────────────────────────────────
def _vec(rec: dict | None, key: str, dim: int) -> np.ndarray:
    if not rec:
        return np.zeros(dim, dtype=np.float32)
    v = np.asarray(rec.get(key, []), dtype=np.float32)
    return v if v.size == dim else np.zeros(dim, dtype=np.float32)


def _hc(rec: dict | None, keys: list[str]) -> np.ndarray:
    if not rec:
        return np.zeros(len(keys), dtype=np.float32)
    f = rec.get("handcrafted", {}).get("features", {})
    return np.nan_to_num(np.array([float(f.get(k, 0.0)) for k in keys],
                                  dtype=np.float32))


def build_matrix(rows: list[dict], source: str, use_delta: bool,
                 hc_keys: list[str]) -> np.ndarray:
    """source: 'custom' | 'e2v' | 'hc'"""
    if source == "custom":
        dim = max((len(r["free"].get("custom_embedding", [])) for r in rows), default=0)
        if not dim:
            raise SystemExit("[ERROR] No custom_embedding — run 09 first.")
        key = "custom_embedding"
    elif source == "e2v":
        dim = max((len(r["free"].get("emotion2vec_embedding", [])) for r in rows), default=0)
        if not dim:
            raise SystemExit("[ERROR] No emotion2vec_embedding — run 03 first.")
        key = "emotion2vec_embedding"
    elif source == "hc":
        if not hc_keys:
            raise SystemExit("[ERROR] No handcrafted features shared by all clips. "
                             "Re-run 03 with the handcrafted bank enabled.")
        dim, key = len(hc_keys), None
    else:
        raise ValueError(source)

    out = []
    for r in rows:
        if source == "hc":
            f = _hc(r["free"], hc_keys)
            blocks = [f]
            if use_delta:
                blocks.append(f - _hc(r["pass"], hc_keys))
        else:
            f = _vec(r["free"], key, dim)
            blocks = [f]
            if use_delta:
                blocks.append(f - _vec(r["pass"], key, dim))
        out.append(np.concatenate(blocks))
    return np.stack(out).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    ua = []
    for c in np.unique(y_true):
        m = y_true == c
        if m.sum():
            ua.append(float((y_pred[m] == c).mean()))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "ua": float(np.mean(ua)) if ua else 0.0,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n": int(len(y_true)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Systems
# ─────────────────────────────────────────────────────────────────────────────
def predict_zeroshot(rows: list[dict], which: str) -> np.ndarray:
    preds = []
    for r in rows:
        if which == "custom":
            p = r["free"].get("custom_probs7")
            preds.append(int(np.argmax(p)) if p else EMOTION_LABELS.index("neutral"))
        else:
            lbl = (r["free"].get("emotion2vec_label") or "neutral").lower()
            preds.append(EMOTION_LABELS.index(lbl) if lbl in EMOTION_LABELS
                         else EMOTION_LABELS.index("neutral"))
    return np.array(preds, dtype=np.int64)


def train_and_predict(X_tr, y_tr, g_tr, X_te, y_te, seed: int) -> np.ndarray:
    """Train the shared MLP on the public TRAIN split, predict on the holdout."""
    ym_tr = np.stack([emotions_to_multilabel([EMOTION_LABELS[i]]) for i in y_tr])
    ym_te = np.stack([emotions_to_multilabel([EMOTION_LABELS[i]]) for i in y_te])

    in_tr, in_val = make_inner_split(y_tr, g_tr, "grouped", seed=seed)
    model, norm = train_fold_explicit(
        X_tr[in_tr], y_tr[in_tr], ym_tr[in_tr],
        X_tr[in_val], y_tr[in_val], ym_tr[in_val],
        device=DEVICE, seed=seed)
    res = evaluate_fold(model, X_te, y_te, ym_te, norm, device=DEVICE)
    return np.array(res["y7_pred"], dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def report(name: str, y: np.ndarray, p: np.ndarray, groups: np.ndarray,
           corpora: np.ndarray, seed: int) -> dict:
    m = metrics(y, p)
    ci = cluster_bootstrap_ci(y, p, groups, metric="macro_f1", seed=seed)
    ci_acc = cluster_bootstrap_ci(y, p, groups, metric="accuracy", seed=seed)

    print(f"\n  {name}")
    print(f"    acc {m['accuracy']:.4f} [{ci_acc['ci_low']:.3f},{ci_acc['ci_high']:.3f}]"
          f"   UA {m['ua']:.4f}"
          f"   macroF1 {m['f1_macro']:.4f} [{ci['ci_low']:.3f},{ci['ci_high']:.3f}]"
          f"   microF1 {m['f1_micro']:.4f}   wF1 {m['f1_weighted']:.4f}")

    per_corpus = {}
    for c in sorted(set(corpora)):
        mask = corpora == c
        pc = metrics(y[mask], p[mask])
        pc3 = metrics(IDX7_TO_IDX3[y[mask]], IDX7_TO_IDX3[p[mask]])
        per_corpus[c] = {"7class": pc, "3class_posthoc": pc3}
        print(f"      {c:<9} n={pc['n']:<5} acc {pc['accuracy']:.4f}  UA {pc['ua']:.4f}  "
              f"mF1 {pc['f1_macro']:.4f}   |  3-class acc {pc3['accuracy']:.4f}  "
              f"mF1 {pc3['f1_macro']:.4f}")

    m3 = metrics(IDX7_TO_IDX3[y], IDX7_TO_IDX3[p])
    print(f"    3-class (post-hoc mapping)  acc {m3['accuracy']:.4f}  "
          f"UA {m3['ua']:.4f}  macroF1 {m3['f1_macro']:.4f}")

    return {"metrics7": m, "metrics3_posthoc": m3, "per_corpus": per_corpus,
            "ci_macro_f1": ci, "ci_accuracy": ci_acc,
            "y_pred": p.tolist()}


def markdown_table(results: dict) -> str:
    head = ("| System | Acc (WA) | UA | Macro F1 | Micro F1 | Weighted F1 | "
            "3-class Acc |\n|---|---|---|---|---|---|---|\n")
    rows = ""
    for name, r in results.items():
        m, m3 = r["metrics7"], r["metrics3_posthoc"]
        rows += (f"| {name} | {m['accuracy']:.3f} | {m['ua']:.3f} | "
                 f"{m['f1_macro']:.3f} | {m['f1_micro']:.3f} | "
                 f"{m['f1_weighted']:.3f} | {m3['accuracy']:.3f} |\n")
    return head + rows


def main():
    p = argparse.ArgumentParser(description="Held-out public benchmark")
    p.add_argument("--data", type=Path, default=Path("data_public"))
    p.add_argument("--features", type=Path, default=None)
    p.add_argument("--video_index", type=Path, default=None)
    p.add_argument("--systems", default=DEFAULT_SYSTEMS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--compare", default="",
                   help="Pairs for paired bootstrap, e.g. custom_mlp:e2v_mlp,"
                        "custom_mlp_delta:custom_mlp")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    feat_path = args.features or args.data / "features.jsonl"
    idx_path = args.video_index or args.data / "firebase_data" / "video_index.json"

    index = load_index(idx_path)
    features = load_features(feat_path)
    rows = build_sessions(index, features)
    if not rows:
        raise SystemExit("[ERROR] No usable sessions.")

    hc_keys = hc_common_keys(features)
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "holdout"]
    if not test_rows:
        raise SystemExit("[ERROR] No holdout rows. Re-run 00 with --holdout > 0.")

    print(f"[INFO] {len(rows)} sessions | train {len(train_rows)} | "
          f"holdout {len(test_rows)} | handcrafted keys {len(hc_keys)}")
    print(f"[INFO] holdout speakers: "
          f"{len({r['user_id'] for r in test_rows})}, corpora: "
          f"{sorted({r['corpus'] for r in test_rows})}")

    y_tr = np.array([r["y7"] for r in train_rows], dtype=np.int64)
    g_tr = np.array([r["user_id"] for r in train_rows])
    y_te = np.array([r["y7"] for r in test_rows], dtype=np.int64)
    g_te = np.array([r["user_id"] for r in test_rows])
    c_te = np.array([r["corpus"] for r in test_rows])

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    results: dict[str, dict] = {}
    preds: dict[str, np.ndarray] = {}

    for name in systems:
        print(f"\n{'=' * 74}\n  {name}\n{'=' * 74}")
        if name == "custom_zeroshot":
            pred = predict_zeroshot(test_rows, "custom")
        elif name == "e2v_zeroshot":
            pred = predict_zeroshot(test_rows, "e2v")
        else:
            base = name.replace("_delta", "").replace("_mlp", "")
            use_delta = name.endswith("_delta")
            X_tr = build_matrix(train_rows, base, use_delta, hc_keys)
            X_te = build_matrix(test_rows, base, use_delta, hc_keys)
            print(f"  feature dim {X_tr.shape[1]}  train n={len(X_tr)}")
            pred = train_and_predict(X_tr, y_tr, g_tr, X_te, y_te, args.seed)

        preds[name] = pred
        results[name] = report(name, y_te, pred, g_te, c_te, args.seed)

    # ── Significance between systems ─────────────────────────────────────────
    pairs = [tuple(x.split(":")) for x in args.compare.split(",") if ":" in x]
    comparisons = {}
    if pairs:
        print(f"\n{'=' * 74}\n  PAIRED BOOTSTRAP (speaker-level resampling)\n{'=' * 74}")
        for a, b in pairs:
            if a not in preds or b not in preds:
                print(f"  [skip] {a} vs {b}: system not run")
                continue
            t = paired_bootstrap_test(y_te, preds[a], preds[b], g_te,
                                      metric="macro_f1", seed=args.seed)
            comparisons[f"{a}_vs_{b}"] = t
            print(f"  {a} vs {b}: Δ macro F1 = {t['delta']:+.4f} "
                  f"[{t['delta_ci_low']:+.3f}, {t['delta_ci_high']:+.3f}]  "
                  f"p={t['p_value']:.4f}"
                  f"{'  significant' if t['significant_05'] else ''}")

    # ── Detail on the best system ────────────────────────────────────────────
    best = max(results, key=lambda k: results[k]["metrics7"]["f1_macro"])
    print(f"\n{'=' * 74}\n  Detail — {best}\n{'=' * 74}")
    print(classification_report(y_te, preds[best], labels=list(range(N_EMOTIONS)),
                                target_names=EMOTION_LABELS, zero_division=0, digits=3))
    cm = confusion_matrix(y_te, preds[best], labels=list(range(N_EMOTIONS)))
    print("  confusion matrix (rows = true):")
    print("            " + "".join(l[:5].rjust(7) for l in EMOTION_LABELS))
    for i, l in enumerate(EMOTION_LABELS):
        print(f"  {l:<10}" + "".join(str(v).rjust(7) for v in cm[i]))

    print(f"\n{'=' * 74}\n  TABLE FOR THE PAPER (held-out public speakers)\n{'=' * 74}")
    print(markdown_table(results))

    out = args.out or args.data / "results" / "public_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_train": len(train_rows), "n_holdout": len(test_rows),
        "holdout_speakers": sorted({r["user_id"] for r in test_rows}),
        "systems": {k: {kk: vv for kk, vv in v.items() if kk != "y_pred"}
                    for k, v in results.items()},
        "comparisons": comparisons,
        "y_true": y_te.tolist(),
        "predictions": {k: v.tolist() for k, v in preds.items()},
        "labels": EMOTION_LABELS,
    }, indent=2))
    (out.with_suffix(".md")).write_text(markdown_table(results))
    print(f"[INFO] Results -> {out}")


if __name__ == "__main__":
    main()
