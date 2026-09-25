"""
stats.py — Statistical rigour for publication.

Point estimates from a single CV run are not publishable. With 15 speakers and
minority classes holding 20-40 examples, a 3-point difference is well inside
noise. Reviewers at ICASSP / Interspeech / TAFFC will ask for confidence
intervals and a significance test on every claimed improvement.

Three tools here:

  cluster_bootstrap_ci   Confidence interval on a pooled metric.
  paired_bootstrap_test  Is system A better than system B? (the ablation test)
  permutation_test       Is the model better than chance at all?

WHY CLUSTER BOOTSTRAP AND NOT ORDINARY BOOTSTRAP
─────────────────────────────────────────────────────────────────────────────
Sessions are nested inside speakers, and sessions from one speaker are not
independent of each other -- they share a voice, a microphone, and a personal
baseline. Resampling individual sessions treats 300 correlated observations as
300 independent ones and produces intervals that are far too narrow.

The correct unit of resampling is the SPEAKER. We resample speakers with
replacement, take all of that speaker's sessions, and recompute the metric.
The resulting interval is wider and honest: it reflects that the real sample
size is 15 people, not 300 rows.

Report as: macro F1 = 0.312, 95% CI [0.241, 0.388].
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

DEFAULT_N_BOOT = 2000
DEFAULT_ALPHA = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def unweighted_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    UA / UAR: mean per-class recall. This is what most SER papers report
    alongside WA, so compute it if you want numbers comparable to the
    literature.
    """
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() > 0:
            recalls.append(float((y_pred[mask] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


METRICS: dict[str, Callable] = {
    "macro_f1": macro_f1,
    "weighted_f1": weighted_f1,
    "accuracy": accuracy,
    "ua": unweighted_accuracy,
}


# ─────────────────────────────────────────────────────────────────────────────
# Cluster bootstrap confidence interval
# ─────────────────────────────────────────────────────────────────────────────
def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    metric: str | Callable = "macro_f1",
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 42,
) -> dict:
    """
    Percentile bootstrap CI, resampling at the speaker level.

    groups : speaker id per sample. This is the clustering variable.
    """
    fn = METRICS[metric] if isinstance(metric, str) else metric
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))

    point = fn(y_true, y_pred)

    unique_groups = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in unique_groups}
    rng = np.random.RandomState(seed)

    boots = []
    for _ in range(n_boot):
        picked = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in picked])
        # A resample can miss a class entirely; zero_division=0 handles it.
        boots.append(fn(y_true[idx], y_pred[idx]))

    boots = np.array(boots)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))

    return {
        "metric": metric if isinstance(metric, str) else "custom",
        "point": point,
        "ci_low": lo,
        "ci_high": hi,
        "ci_width": hi - lo,
        "boot_mean": float(boots.mean()),
        "boot_std": float(boots.std()),
        "n_boot": n_boot,
        "n_clusters": int(len(unique_groups)),
        "resample_unit": "speaker",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Paired bootstrap: is system A better than system B?
# ─────────────────────────────────────────────────────────────────────────────
def paired_bootstrap_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    groups: np.ndarray,
    metric: str | Callable = "macro_f1",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 42,
) -> dict:
    """
    Test whether system A beats system B on the SAME sessions.

    This is the test for every ablation claim in the paper: with vs without the
    passage reference, with vs without ASR augmentation, LLM fusion vs late
    fusion. Both systems must have been evaluated on identical sessions in
    identical order.

    Returns a one-sided p-value: the fraction of speaker-level resamples in
    which A does NOT beat B. Small p means the improvement survives
    resampling the speaker pool.
    """
    fn = METRICS[metric] if isinstance(metric, str) else metric
    y_true, pred_a, pred_b, groups = map(
        np.asarray, (y_true, pred_a, pred_b, groups))

    if not (len(y_true) == len(pred_a) == len(pred_b) == len(groups)):
        raise ValueError("Systems must be evaluated on identical sessions.")

    obs_a, obs_b = fn(y_true, pred_a), fn(y_true, pred_b)
    obs_delta = obs_a - obs_b

    unique_groups = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in unique_groups}
    rng = np.random.RandomState(seed)

    deltas = []
    for _ in range(n_boot):
        picked = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in picked])
        deltas.append(fn(y_true[idx], pred_a[idx]) - fn(y_true[idx], pred_b[idx]))

    deltas = np.array(deltas)
    p_one_sided = float((deltas <= 0).mean())

    return {
        "metric": metric if isinstance(metric, str) else "custom",
        "score_a": obs_a,
        "score_b": obs_b,
        "delta": obs_delta,
        "delta_ci_low": float(np.percentile(deltas, 2.5)),
        "delta_ci_high": float(np.percentile(deltas, 97.5)),
        "p_value": p_one_sided,
        "significant_05": p_one_sided < 0.05,
        "n_boot": n_boot,
        "n_clusters": int(len(unique_groups)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Permutation test against a chance null
# ─────────────────────────────────────────────────────────────────────────────
def permutation_test_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str | Callable = "macro_f1",
    n_perm: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Approximate permutation test: shuffle the true labels and recompute.

    IMPORTANT CAVEAT for the write-up. The gold-standard permutation test
    re-runs the entire cross-validation on shuffled labels, which also captures
    variance from model fitting. This version permutes only at the prediction
    stage, so it is cheaper but tests a narrower null: "could this prediction
    vector match these labels by chance given the class distribution?"

    Use it as a sanity screen. If a result does not clear this bar it is
    definitely not real. Clearing it is necessary, not sufficient -- report the
    cluster bootstrap CI as the headline uncertainty.
    """
    fn = METRICS[metric] if isinstance(metric, str) else metric
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    obs = fn(y_true, y_pred)
    rng = np.random.RandomState(seed)

    null = np.empty(n_perm)
    shuffled = y_true.copy()
    for i in range(n_perm):
        rng.shuffle(shuffled)
        null[i] = fn(shuffled, y_pred)

    p = float(((null >= obs).sum() + 1) / (n_perm + 1))   # add-one correction

    return {
        "metric": metric if isinstance(metric, str) else "custom",
        "observed": obs,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": p,
        "significant_05": p < 0.05,
        "n_perm": n_perm,
        "caveat": "prediction-level permutation; not a full CV re-run",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed aggregation
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_seeds(runs: list[dict], keys: Optional[list[str]] = None) -> dict:
    """
    Summarise the same experiment repeated over several random seeds.

    A single seed tells you nothing about run-to-run stability. Report
    mean +/- std over at least 5 seeds; papers that report one seed on a small
    corpus are routinely asked to redo the experiment.
    """
    if not runs:
        return {}
    if keys is None:
        keys = [k for k, v in runs[0].get("pooled", runs[0]).items()
                if isinstance(v, (int, float))]

    out = {}
    for k in keys:
        vals = []
        for r in runs:
            src = r.get("pooled", r)
            if k in src:
                vals.append(float(src[k]))
        if vals:
            out[k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "n_seeds": len(vals),
            }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def format_ci(res: dict, digits: int = 3) -> str:
    return (f"{res['point']:.{digits}f} "
            f"[{res['ci_low']:.{digits}f}, {res['ci_high']:.{digits}f}]")


def print_stats_block(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    title: str = "Results",
    metrics: tuple = ("accuracy", "macro_f1", "weighted_f1", "ua"),
    seed: int = 42,
) -> dict:
    """Print the table that goes into the paper, and return it as a dict."""
    print(f"\n  {title}")
    print(f"  {'-' * 62}")
    print(f"  {'metric':<14} {'point':>8}   {'95% CI (speaker bootstrap)':<26}")
    print(f"  {'-' * 62}")

    out = {}
    for m in metrics:
        res = cluster_bootstrap_ci(y_true, y_pred, groups, metric=m, seed=seed)
        out[m] = res
        print(f"  {m:<14} {res['point']:>8.3f}   "
              f"[{res['ci_low']:.3f}, {res['ci_high']:.3f}]  "
              f"width={res['ci_width']:.3f}")

    perm = permutation_test_predictions(y_true, y_pred, "macro_f1", seed=seed)
    out["permutation"] = perm
    print(f"\n  Permutation vs chance (macro F1): observed {perm['observed']:.3f}, "
          f"null {perm['null_mean']:.3f}+/-{perm['null_std']:.3f}, p={perm['p_value']:.4f}")

    n_clusters = out[metrics[0]]["n_clusters"]
    print(f"  Resampling unit: speaker (n={n_clusters}). "
          f"Intervals are wide because that is the real sample size.")
    return out
