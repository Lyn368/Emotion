"""
report.py - Metrics, CSVs and plots, shared by the audio, text and fusion stages.

One call writes a complete, self-describing result folder:

    write_report(out_dir, y_true, y_pred, proba, speakers, keys, title=...)

    out_dir/
        metrics.json            7- and 3-class metrics with 95% speaker-bootstrap CIs
        predictions_oof.csv     one row per session: truth, prediction, confidence, probs
        errors.csv              misclassified sessions only, most confident first
        plots/
            confusion_7.png     counts + row-normalised
            confusion_3.png
            tp_fp_fn_tn_7.png   the four outcome types per class
            tp_fp_fn_tn_3.png
            per_class_f1_7.png  precision / recall / F1 with CIs
            per_speaker.png     accuracy per held-out speaker
            distribution.png    true vs predicted class counts
            confidence.png      confidence of correct vs wrong predictions
            calibration.png     reliability diagram + ECE

Plots are ASCII-only in their text so they render the same on Windows.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")                      # no display needed (works over SSH / in scripts)
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from labels import EMOTIONS, VALENCE, TO_VALENCE
from stats import cluster_bootstrap_ci, permutation_test_predictions, unweighted_accuracy

TO_V_IDX = np.array([VALENCE.index(TO_VALENCE[e]) for e in EMOTIONS])
C_OK, C_BAD, C_ALT, C_GREY = "#2F6B4F", "#A8324A", "#2E5F7C", "#9AA5AE"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _metric_block(y, p, g, seed) -> dict:
    out = {}
    for m in ("accuracy", "macro_f1", "weighted_f1", "ua"):
        r = cluster_bootstrap_ci(y, p, g, metric=m, seed=seed)
        out[m] = {"point": r["point"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]}
    perm = permutation_test_predictions(y, p, "macro_f1", seed=seed)
    out["permutation_p_macro_f1"] = perm["p_value"]
    return out


def per_class_ci(y, p, g, n_classes, n_boot=1000, seed=42):
    """Per-class F1 with speaker-cluster bootstrap CIs (resample speakers, not rows)."""
    rng = np.random.RandomState(seed)
    groups = np.unique(g)
    by_g = {k: np.where(g == k)[0] for k in groups}
    point = precision_recall_fscore_support(y, p, labels=range(n_classes), zero_division=0)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([by_g[k] for k in rng.choice(groups, len(groups), replace=True)])
        boots.append(f1_score(y[idx], p[idx], labels=range(n_classes), average=None, zero_division=0))
    boots = np.array(boots)
    return {"precision": point[0], "recall": point[1], "f1": point[2], "support": point[3],
            "f1_lo": np.percentile(boots, 2.5, axis=0), "f1_hi": np.percentile(boots, 97.5, axis=0)}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y, p, names, path, title):
    cm = confusion_matrix(y, p, labels=range(len(names)))
    rows = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rows, out=np.zeros_like(cm, dtype=float), where=rows > 0)
    fig, axes = plt.subplots(1, 2, figsize=(4.2 * 2 + 1, 4.2))
    for ax, mat, fmt, sub in ((axes[0], cm, "d", "counts"), (axes[1], norm, ".2f", "row-normalised (recall)")):
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=(mat.max() or 1))
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(sub, fontsize=10)
        hi = (mat.max() or 1) * 0.6
        for i in range(len(names)):
            for j in range(len(names)):
                v = mat[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8,
                        color="white" if v > hi else "black")
    fig.suptitle(title, fontsize=11)
    _save(fig, path)


def plot_outcomes(y, p, names, path, title):
    """TP / FP / FN / TN per class (one-vs-rest)."""
    tp, fp, fn, tn = [], [], [], []
    for c in range(len(names)):
        t, q = (y == c), (p == c)
        tp.append(int((t & q).sum())); fp.append(int((~t & q).sum()))
        fn.append(int((t & ~q).sum())); tn.append(int((~t & ~q).sum()))
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 3, 4))
    ax.bar(x, tp, color=C_OK, label="TP  true=c, pred=c")
    ax.bar(x, fn, bottom=tp, color=C_BAD, label="FN  true=c, pred!=c")
    ax.bar(x, fp, bottom=np.array(tp) + np.array(fn), color="#D6A863", label="FP  true!=c, pred=c")
    for i in x:
        ax.text(i, tp[i] + fn[i] + fp[i], f"TN={tn[i]}", ha="center", va="bottom", fontsize=7, color="#4A5862")
    ax.set_xticks(x, names, rotation=30, ha="right"); ax.set_ylabel("sessions")
    ax.set_title(title, fontsize=11); ax.legend(fontsize=8, frameon=False, loc="upper right")
    _save(fig, path)
    return {n: {"TP": tp[i], "FP": fp[i], "FN": fn[i], "TN": tn[i]} for i, n in enumerate(names)}


def plot_per_class(pc, names, path, title):
    x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(1.2 * len(names) + 3, 4))
    ax.bar(x - w, pc["precision"], w, color=C_ALT, label="precision")
    ax.bar(x, pc["recall"], w, color="#D6A863", label="recall")
    err = np.vstack([pc["f1"] - pc["f1_lo"], pc["f1_hi"] - pc["f1"]])
    ax.bar(x + w, pc["f1"], w, color=C_OK, label="F1 (95% CI)", yerr=err, capsize=3)
    for i in x:
        ax.text(i, -0.08, f"n={int(pc['support'][i])}", ha="center", fontsize=7, color="#4A5862")
    ax.set_xticks(x, names, rotation=30, ha="right"); ax.set_ylim(-0.12, 1.05)
    ax.axhline(0, color="#999", lw=0.5); ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, frameon=False, ncol=3, loc="upper right")
    _save(fig, path)


def plot_per_speaker(y, p, g, path, title):
    sp = sorted(np.unique(g), key=lambda s: -np.mean(y[g == s] == p[g == s]))
    acc = [np.mean(y[g == s] == p[g == s]) for s in sp]
    n = [int((g == s).sum()) for s in sp]
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(sp) + 2), 4))
    ax.bar(range(len(sp)), acc, color=[C_OK if a >= np.mean(acc) else C_BAD for a in acc])
    ax.axhline(np.mean(y == p), color="#333", ls="--", lw=1, label=f"pooled acc {np.mean(y == p):.3f}")
    ax.set_xticks(range(len(sp)), [f"{str(s)[:8]}\n(n={k})" for s, k in zip(sp, n)], fontsize=7, rotation=45, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("accuracy"); ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    _save(fig, path)


def plot_distribution(y, p, names, path, title):
    x = np.arange(len(names)); w = 0.38
    t = np.bincount(y, minlength=len(names)); q = np.bincount(p, minlength=len(names))
    fig, ax = plt.subplots(figsize=(1.1 * len(names) + 3, 3.6))
    ax.bar(x - w / 2, t, w, color=C_GREY, label="true"); ax.bar(x + w / 2, q, w, color=C_ALT, label="predicted")
    ax.set_xticks(x, names, rotation=30, ha="right"); ax.set_ylabel("sessions")
    ax.set_title(title, fontsize=11); ax.legend(fontsize=8, frameon=False)
    _save(fig, path)


def plot_confidence(conf, correct, path, title):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.linspace(0, 1, 21)
    ax.hist(conf[correct], bins=bins, alpha=0.7, color=C_OK, label=f"correct (n={int(correct.sum())})")
    ax.hist(conf[~correct], bins=bins, alpha=0.7, color=C_BAD, label=f"wrong (n={int((~correct).sum())})")
    ax.set_xlabel("confidence of predicted class"); ax.set_ylabel("sessions")
    ax.set_title(title, fontsize=11); ax.legend(fontsize=8, frameon=False)
    _save(fig, path)


def plot_calibration(conf, correct, path, title, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    ece, xs, ys, ns = 0.0, [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            xs.append(conf[m].mean()); ys.append(correct[m].mean()); ns.append(int(m.sum()))
            ece += m.mean() * abs(conf[m].mean() - correct[m].mean())
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1, label="perfect calibration")
    ax.plot(xs, ys, "o-", color=C_ALT, label=f"model (ECE={ece:.3f})")
    for x_, y_, n_ in zip(xs, ys, ns):
        ax.annotate(str(n_), (x_, y_), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("mean confidence"); ax.set_ylabel("observed accuracy")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _save(fig, path)
    return float(ece)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def write_report(out_dir: str | Path, y_true, y_pred, proba: Optional[np.ndarray],
                 speakers, keys, title: str, extra_cols: Optional[dict] = None,
                 config: Optional[dict] = None, seed: int = 42, verbose: bool = True) -> dict:
    """
    y_true, y_pred : 7-class indices (EMOTIONS order), one per session
    proba          : (n, 7) probabilities or None
    speakers, keys : speaker id and session key per row
    extra_cols     : e.g. {"transcript": [...]} -> added to predictions/errors CSVs
    config         : the run configuration, saved next to the results
    """
    out = Path(out_dir); plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    y, p = np.asarray(y_true, int), np.asarray(y_pred, int)
    g = np.asarray(speakers).astype(str); keys = list(keys)
    y3, p3 = TO_V_IDX[y], TO_V_IDX[p]

    metrics = {"title": title, "n_sessions": int(len(y)), "n_speakers": int(len(np.unique(g))),
               "emotion7": _metric_block(y, p, g, seed), "valence3": _metric_block(y3, p3, g, seed)}

    pc7 = per_class_ci(y, p, g, len(EMOTIONS), seed=seed)
    metrics["emotion7"]["per_class"] = {
        e: {"precision": float(pc7["precision"][i]), "recall": float(pc7["recall"][i]),
            "f1": float(pc7["f1"][i]), "f1_ci": [float(pc7["f1_lo"][i]), float(pc7["f1_hi"][i])],
            "support": int(pc7["support"][i])} for i, e in enumerate(EMOTIONS)}

    plot_confusion(y, p, EMOTIONS, plots / "confusion_7.png", f"{title} - 7 emotions")
    plot_confusion(y3, p3, VALENCE, plots / "confusion_3.png", f"{title} - 3 valence classes")
    metrics["emotion7"]["outcomes"] = plot_outcomes(y, p, EMOTIONS, plots / "tp_fp_fn_tn_7.png",
                                                    f"{title} - TP/FP/FN/TN, 7 emotions")
    metrics["valence3"]["outcomes"] = plot_outcomes(y3, p3, VALENCE, plots / "tp_fp_fn_tn_3.png",
                                                    f"{title} - TP/FP/FN/TN, 3 classes")
    plot_per_class(pc7, EMOTIONS, plots / "per_class_f1_7.png", f"{title} - per-class metrics")
    plot_per_speaker(y, p, g, plots / "per_speaker.png", f"{title} - accuracy per speaker")
    plot_distribution(y, p, EMOTIONS, plots / "distribution.png", f"{title} - true vs predicted")

    conf = None
    if proba is not None:
        proba = np.asarray(proba, float)
        conf = proba.max(axis=1)
        correct = (y == p)
        plot_confidence(conf, correct, plots / "confidence.png", f"{title} - confidence")
        metrics["emotion7"]["ece"] = plot_calibration(conf, correct, plots / "calibration.png",
                                                      f"{title} - calibration")

    # CSVs (utf-8-sig so Excel on Windows shows accents correctly)
    extra_cols = extra_cols or {}
    header = ["key", "speaker", "true", "pred", "correct", "true3", "pred3", "confidence"] \
        + ([f"p_{e}" for e in EMOTIONS] if proba is not None else []) + list(extra_cols)
    rows = []
    for i in range(len(y)):
        r = [keys[i], g[i], EMOTIONS[y[i]], EMOTIONS[p[i]], int(y[i] == p[i]),
             VALENCE[y3[i]], VALENCE[p3[i]], "" if conf is None else round(float(conf[i]), 4)]
        if proba is not None:
            r += [round(float(v), 4) for v in proba[i]]
        r += [extra_cols[c][i] for c in extra_cols]
        rows.append(r)
    with open(out / "predictions_oof.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    wrong = [r for r in rows if r[4] == 0]
    wrong.sort(key=lambda r: -(r[7] if r[7] != "" else 0))
    with open(out / "errors.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(wrong)

    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if config is not None:
        with open(out / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, default=str)

    if verbose:
        e7, v3 = metrics["emotion7"], metrics["valence3"]
        print(f"  {title}")
        for name, blk in (("7 emotions", e7), ("3 valence ", v3)):
            print(f"    {name}  acc {blk['accuracy']['point']:.3f} "
                  f"[{blk['accuracy']['ci_low']:.3f},{blk['accuracy']['ci_high']:.3f}]  "
                  f"macroF1 {blk['macro_f1']['point']:.3f} "
                  f"[{blk['macro_f1']['ci_low']:.3f},{blk['macro_f1']['ci_high']:.3f}]  "
                  f"perm p={blk['permutation_p_macro_f1']:.4f}")
        print(f"    -> {out}")
    return metrics


def plot_agreement(y, pred_a, pred_b, name_a, name_b, path, title):
    """
    2x2 agreement between two branches (e.g. audio vs text), for 7 and 3 classes.
    'A right, B wrong' is the only region where fusion can beat B alone.
    """
    y, a, b = map(lambda v: np.asarray(v, int), (y, pred_a, pred_b))
    res = {}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (lvl, yy, aa, bb) in zip(axes, (("7 emotions", y, a, b),
                                            ("3 valence", TO_V_IDX[y], TO_V_IDX[a], TO_V_IDX[b]))):
        ra, rb = aa == yy, bb == yy
        m = np.array([[np.sum(ra & rb), np.sum(ra & ~rb)], [np.sum(~ra & rb), np.sum(~ra & ~rb)]])
        ax.imshow(m, cmap="Greens")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{m[i, j]}\n({m[i, j] / len(yy):.0%})", ha="center", va="center", fontsize=10)
        ax.set_xticks([0, 1], [f"{name_b} right", f"{name_b} wrong"])
        ax.set_yticks([0, 1], [f"{name_a} right", f"{name_a} wrong"])
        ax.set_title(lvl, fontsize=10)
        res[lvl] = {"both_right": int(m[0, 0]), f"{name_a}_only": int(m[0, 1]),
                    f"{name_b}_only": int(m[1, 0]), "both_wrong": int(m[1, 1])}
    fig.suptitle(title, fontsize=11)
    _save(fig, path)
    return res
