"""
c1_compare_datasets.py - One table and two figures across every evaluation set.

    python c1_compare_datasets.py --protocol grouped
    python c1_compare_datasets.py --protocol louo --eval_sets own,ravdess,iemocap

Collects every metrics.json written by the audio, text and fusion branches for the
chosen protocol, for each evaluation set (own, ravdess, cremad, iemocap), and
averages over seeds. It also runs the key ablation of this work on every dataset:
the same encoder WITH vs WITHOUT the neutral reference (delta), as a paired
speaker-bootstrap test on identical folds.

Output: comparison/
    summary_<protocol>.csv       every run: dataset, branch, run, mean +/- std over seeds
    summary_<protocol>.md        best run per dataset and branch, readable
    delta_<protocol>.csv         with vs without the neutral reference, per dataset
    macro_f1_<protocol>.png      frozen audio configurations per dataset, with chance level
    delta_<protocol>.png         gain from the neutral reference, with 95% CI

Reading the numbers against the literature: public benchmarks usually report
accuracy (WA) and unweighted accuracy (UA, mean per-class recall); both are here.
IEMOCAP 4-class with leave-one-session-out ('louo' here) is the standard setup.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import EVAL_SETS, V2_DIR, eval_paths, outputs, set_eval
from stats import paired_bootstrap_test

METRICS = [("emotion7", "accuracy", "acc"), ("emotion7", "ua", "ua"), ("emotion7", "macro_f1", "f1"),
           ("valence3", "accuracy", "acc3"), ("valence3", "macro_f1", "f1_3")]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]        # validated categorical order
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def collect(protocol: str) -> dict:
    """{(dataset, branch, run): {seed: metrics_json_path}}"""
    found = defaultdict(dict)
    for ds in EVAL_SETS:
        set_eval(ds)
        o = outputs()
        for branch in ("audio_result", "text_result", "fusion_result"):
            root = o[branch]
            if not root.exists():
                continue
            for m in root.rglob("metrics.json"):
                parts = m.relative_to(root).parts
                if protocol not in parts:
                    continue
                i = parts.index(protocol)
                seed_part = parts[i + 1] if i + 1 < len(parts) else ""
                if not seed_part.startswith("seed"):
                    continue
                run = "/".join(parts[:i] + parts[i + 2:-1])        # fusion keeps its strategy name
                if branch == "fusion_result" and parts[-2] in ("audio", "text"):
                    continue                                        # single-branch baselines, not fusions
                found[(ds, branch.replace("_result", ""), run)][int(seed_part[4:])] = m
    return found


def chance(ds: str) -> dict:
    p = eval_paths(ds)["splits"]
    if not p.exists():
        return {}
    ev = json.loads(p.read_text(encoding="utf-8"))["evaluation_set"]
    c = Counter(s["y7"] for s in ev)
    return {"n": len(ev), "n_speakers": len({s["speaker_id"] for s in ev}), "k": len(c),
            "majority_acc": max(c.values()) / len(ev), "chance_ua": 1 / len(c)}


def oof_pred(run_dir: Path, protocol: str, seed: int):
    p = run_dir / protocol / f"seed{seed}" / "oof.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["keys"], np.array(d["y7"]), np.argmax(d["test_proba"], 1)


def main():
    ap = argparse.ArgumentParser(description="Compare results across datasets")
    ap.add_argument("--protocol", default="grouped")
    ap.add_argument("--eval_sets", default=",".join(EVAL_SETS))
    a = ap.parse_args()
    sets = [s for s in a.eval_sets.split(",") if s]
    out = V2_DIR / "comparison"
    out.mkdir(exist_ok=True)

    found = collect(a.protocol)
    base = {ds: chance(ds) for ds in sets}
    rows = []
    for (ds, branch, run), by_seed in sorted(found.items()):
        if ds not in sets:
            continue
        vals = defaultdict(list)
        first = json.loads(by_seed[min(by_seed)].read_text(encoding="utf-8"))
        for seed, p in sorted(by_seed.items()):
            m = json.loads(p.read_text(encoding="utf-8"))
            for lvl, key, short in METRICS:
                vals[short].append(m[lvl][key]["point"])
        r = {"dataset": ds, "branch": branch, "run": run, "n_seeds": len(by_seed),
             "n_sessions": first["n_sessions"], "n_speakers": first["n_speakers"]}
        for _, _, short in METRICS:
            r[f"{short}_mean"] = round(float(np.mean(vals[short])), 4)
            r[f"{short}_std"] = round(float(np.std(vals[short])), 4)
        ci = first["emotion7"]["macro_f1"]
        r["f1_ci_seed0"] = f"[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]"
        r["perm_p_seed0"] = round(first["emotion7"]["permutation_p_macro_f1"], 4)
        r["majority_acc"] = round(base.get(ds, {}).get("majority_acc", np.nan), 4)
        r["chance_ua"] = round(base.get(ds, {}).get("chance_ua", np.nan), 4)
        rows.append(r)
    if not rows:
        raise SystemExit(f"[ERROR] no results for protocol '{a.protocol}'. Run the branches first.")

    with open(out / f"summary_{a.protocol}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ── delta ablation: <enc> vs <enc>_delta (frozen) and *_nodelta vs * (fine-tuned) ──
    delta_rows = []
    for ds in sets:
        set_eval(ds)
        ar = outputs()["audio_result"]
        pairs = []
        for d in (ar / "frozen").glob("*_delta") if (ar / "frozen").exists() else []:
            if not d.name.startswith("acoustic"):
                pairs.append((d.name[:-6], d, ar / "frozen" / d.name[:-6]))
        for d in (ar / "finetune").glob("*_nodelta") if (ar / "finetune").exists() else []:
            pairs.append((f"finetune:{d.name[:-8]}", ar / "finetune" / d.name[:-8], d))
        for name, with_d, without_d in pairs:
            seeds = sorted({int(p.parent.name[4:]) for p in with_d.glob(f"{a.protocol}/seed*/oof.json")})
            for seed in seeds[:1]:
                A, B = oof_pred(with_d, a.protocol, seed), oof_pred(without_d, a.protocol, seed)
                if not A or not B or A[0] != B[0]:
                    continue
                keys, y, pw = A
                ev = json.loads(eval_paths(ds)["splits"].read_text(encoding="utf-8"))["evaluation_set"]
                spk = {s["key"]: s["speaker_id"] for s in ev}
                t = paired_bootstrap_test(y, pw, B[2], np.array([spk[k] for k in keys]), "macro_f1", seed=seed)
                delta_rows.append({"dataset": ds, "encoder": name, "seed": seed,
                                   "f1_without": round(t["score_b"], 4), "f1_with": round(t["score_a"], 4),
                                   "delta": round(t["delta"], 4), "ci_low": round(t["delta_ci_low"], 4),
                                   "ci_high": round(t["delta_ci_high"], 4), "p": round(t["p_value"], 4)})
    if delta_rows:
        with open(out / f"delta_{a.protocol}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(delta_rows[0])); w.writeheader(); w.writerows(delta_rows)

    # ── readable summary: best run per dataset and branch ────────────────────
    md = [f"# Results across datasets - protocol: {a.protocol}", "",
          "Mean over seeds. WA = accuracy, UA = mean per-class recall. "
          "`perm p` tests the first seed's macro F1 against chance.", "",
          "| Dataset | Sessions | Speakers | Majority acc | Chance UA |", "|---|---|---|---|---|"]
    for ds in sets:
        b = base.get(ds)
        if b:
            md.append(f"| {ds} | {b['n']} | {b['n_speakers']} | {b['majority_acc']:.3f} | {b['chance_ua']:.3f} |")
    md += ["", "| Dataset | Branch | Best run | Seeds | WA | UA | Macro F1 | 95% CI (seed 1) | perm p | "
               "Valence acc | Valence F1 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    best = {}
    for r in rows:
        k = (r["dataset"], r["branch"])
        if k not in best or r["f1_mean"] > best[k]["f1_mean"]:
            best[k] = r
    for (ds, br), r in sorted(best.items(), key=lambda t: (sets.index(t[0][0]), t[0][1])):
        sd = lambda x: f"{r[x + '_mean']:.3f} ± {r[x + '_std']:.3f}" if r["n_seeds"] > 1 else f"{r[x + '_mean']:.3f}"
        md.append(f"| {ds} | {br} | {r['run']} | {r['n_seeds']} | {sd('acc')} | {sd('ua')} | {sd('f1')} | "
                  f"{r['f1_ci_seed0']} | {r['perm_p_seed0']:.4f} | {sd('acc3')} | {sd('f1_3')} |")
    md += ["", "Best run is picked on these test results, so it is optimistic; use it to read the "
           "table, not as a claim. Claims need the paired tests below or in each branch's summary.", ""]
    if delta_rows:
        md += ["## Neutral reference (delta): with vs without", "",
               "| Dataset | Encoder | F1 without | F1 with | Delta | 95% CI | p |", "|---|---|---|---|---|---|---|"]
        for d in delta_rows:
            star = " *" if d["p"] < 0.05 else ""
            md.append(f"| {d['dataset']} | {d['encoder']} | {d['f1_without']:.3f} | {d['f1_with']:.3f} | "
                      f"{d['delta']:+.3f}{star} | [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] | {d['p']:.4f} |")
        md += ["", "* p < 0.05 (one-sided paired speaker bootstrap). A CI crossing zero = no detected effect."]
    (out / f"summary_{a.protocol}.md").write_text("\n".join(md), encoding="utf-8")

    # ── figure 1: frozen audio configurations per dataset ────────────────────
    cfg_order = ["acoustic", "wavlm_base_plus", "wavlm_base_plus_delta", "acoustic_wavlm_base_plus_delta"]
    labels = ["acoustic", "WavLM", "WavLM + delta", "acoustic + WavLM + delta"]
    ds_have = [ds for ds in sets if any(r["dataset"] == ds and r["run"] == f"frozen/{c}"
                                        for r in rows for c in cfg_order)]
    if ds_have:
        fig, ax = plt.subplots(figsize=(1.9 * len(ds_have) + 2.5, 4.2))
        w = 0.19
        for j, (c, lab) in enumerate(zip(cfg_order, labels)):
            xs, ys, es = [], [], []
            for i, ds in enumerate(ds_have):
                r = next((r for r in rows if r["dataset"] == ds and r["run"] == f"frozen/{c}"), None)
                if r:
                    xs.append(i + (j - 1.5) * (w + 0.01)); ys.append(r["f1_mean"]); es.append(r["f1_std"])
            if xs:
                bars = ax.bar(xs, ys, w, yerr=es if any(es) else None, color=SERIES[j], label=lab,
                              edgecolor="white", linewidth=1, capsize=2, error_kw={"ecolor": MUTED, "lw": 1})
                for b, v in zip(bars, ys):
                    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                            fontsize=7, color=INK)
        for i, ds in enumerate(ds_have):
            k = base[ds]["k"]
            ax.plot([i - 0.45, i + 0.45], [1 / k, 1 / k], ls="--", color=MUTED, lw=1)
        ax.plot([], [], ls="--", color=MUTED, lw=1, label="chance (1 / classes)")
        ax.set_xticks(range(len(ds_have)), [f"{ds}\n({base[ds]['k']} classes)" for ds in ds_have], color=INK)
        ax.set_ylabel("macro F1", color=INK)
        ax.set_ylim(0, max(1.0, ax.get_ylim()[1]))
        ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"Frozen audio configurations across datasets ({a.protocol})", fontsize=11, color=INK)
        ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
        fig.tight_layout(); fig.savefig(out / f"macro_f1_{a.protocol}.png", dpi=160, bbox_inches="tight"); plt.close(fig)

    # ── figure 2: gain from the neutral reference ────────────────────────────
    if delta_rows:
        fig, ax = plt.subplots(figsize=(6.5, 0.5 * len(delta_rows) + 1.6))
        for i, d in enumerate(delta_rows):
            ax.plot([d["ci_low"], d["ci_high"]], [i, i], color=SERIES[0], lw=2)
            ax.plot(d["delta"], i, "o", color=SERIES[0], ms=8, mec="white", mew=1.5)
            ax.text(d["ci_high"] + 0.01, i, f"{d['delta']:+.3f}  p={d['p']:.3f}", va="center", fontsize=8, color=INK)
        ax.axvline(0, color=MUTED, lw=1)
        ax.invert_yaxis()                                   # same top-to-bottom order as the table
        ax.set_yticks(range(len(delta_rows)), [f"{d['dataset']} | {d['encoder']}" for d in delta_rows], fontsize=8)
        ax.set_xlabel("macro F1 with delta - without (95% CI, paired speaker bootstrap)", color=INK)
        ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_title(f"Effect of the within-speaker neutral reference ({a.protocol})", fontsize=11, color=INK)
        lo = min(d["ci_low"] for d in delta_rows); hi = max(d["ci_high"] for d in delta_rows)
        ax.set_xlim(min(lo, 0) - 0.05, hi + 0.25)
        fig.tight_layout(); fig.savefig(out / f"delta_{a.protocol}.png", dpi=160); plt.close(fig)

    print((out / f"summary_{a.protocol}.md").read_text(encoding="utf-8"))
    print(f"\n[INFO] -> {out}")


if __name__ == "__main__":
    main()
