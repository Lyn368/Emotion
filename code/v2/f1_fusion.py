"""
f1_fusion.py - Fuse the audio and text branches.

Reads the out-of-fold probabilities (oof.json) of one audio run and one text run,
computed on the SAME shared folds, and combines them four ways:

  audio, text   each branch alone (the baselines fusion has to beat)
  late          p = a * p_audio + (1 - a) * p_text; 'a' is chosen per fold on that
                fold's VALIDATION sessions, never on its test sessions
  stack         logistic regression on [log p_audio, log p_text], trained per fold
                on the validation sessions
  llm           an LLM reads the transcript, the speaker-relative acoustic changes
                (freestyle vs passage) and both branches' probabilities, and scores
                the seven labels (no training; frozen prompt with SHA-1)

    python f1_fusion.py --audio best --text best --protocol grouped
    python f1_fusion.py --audio frozen/acoustic_wavlm_base_plus_delta --text emo_distilroberta_sentence_frozen
    python f1_fusion.py --audio best --text best --strategies late,stack,llm

"best" = highest macro F1 on the VALIDATION sessions among every saved run.
Output: fusion_result/<audio>__<text>/<protocol>/seed<k>/<strategy>/ + agreement plot + summary.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from audio_common import OOF, N7, load_evaluation, macro_f1, read_json, summarise
from config import ensure_dirs, get_device
from labels import EMOTIONS
from report import plot_agreement, write_report
from stats import paired_bootstrap_test

LLM_SYSTEM = "You are an expert in recognising emotion from speech and language."
LLM_USER = (
    "A person recorded two clips in one session: reading a fixed neutral text (the reference) and "
    "then talking freely about their day. They are not actors; their voice is often calm even when "
    "they feel something.\n\n"
    'What they said:\n"""{text}"""\n\n'
    "How their voice in the free talk differs from their own neutral reading:\n{acoustic}\n\n"
    "Two trained classifiers gave these probabilities:\n  voice-based: {audio}\n  text-based:  {textp}\n\n"
    "Use only the evidence above; do not invent context. Which emotion did the speaker most likely "
    "feel? Choose exactly one of: angry, disgust, fear, happy, neutral, sad, surprise.\n"
    "Answer with the single word only.")
LLM_SHA1 = hashlib.sha1((LLM_SYSTEM + "\n" + LLM_USER).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def resolve_run(branch_dir: Path, spec: str, protocol: str, seed: int = 42) -> Path:
    """
    'best' = the run with the highest macro F1 on its VALIDATION sessions (pooled over
    folds), searched over every run saved under the branch folder. Choosing on the
    test sessions would be model selection on the test set.
    """
    if spec != "best":
        return branch_dir / spec
    ranking = []
    for oof_path in branch_dir.rglob(f"{protocol}/seed{seed}/oof.json"):
        d = read_json(oof_path)
        pos = {k: i for i, k in enumerate(d["keys"])}
        yv, pv = [], []
        for v in d["val"]:
            yv += [d["y7"][pos[k]] for k in v["keys"]]
            pv += list(np.argmax(v["proba"], 1))
        if yv:
            ranking.append((macro_f1(np.array(yv), np.array(pv)), oof_path.parents[2]))
    if not ranking:
        raise SystemExit(f"[ERROR] no {protocol}/seed{seed} results under {branch_dir}; run that branch first.")
    ranking.sort(key=lambda t: -t[0])
    print(f"[INFO] best in {branch_dir.name} by validation macro F1:")
    for f1v, run in ranking[:5]:
        print(f"       {f1v:.3f}  {run.relative_to(branch_dir).as_posix()}")
    return ranking[0][1]


def load_oof(run_dir: Path, protocol: str, seed: int) -> dict:
    p = run_dir / protocol / f"seed{seed}" / "oof.json"
    if not p.exists():
        raise SystemExit(f"[ERROR] {p} not found. Run that branch with --protocols {protocol} --seeds {seed}.")
    d = read_json(p)
    d["test_proba"] = np.array(d["test_proba"], float)
    return d


def acoustic_text(ev_item, feats) -> str:
    """Speaker-relative voice changes, freestyle minus passage, in plain numbers."""
    f, p = feats.get(ev_item["freestyle_clip"]), feats.get(ev_item.get("passage_clip") or "")
    if not f or not p:
        return "  (no passage reference available)"
    lines = []
    for key, label, unit in (("F0semitoneFrom27.5Hz_sma3nz_amean", "pitch", "semitones"),
                             ("loudness_sma3_amean", "loudness", "(relative units)"),
                             ("speaking_rate_wps", "speaking rate", "words/s"),
                             ("pause_per_min", "pauses", "per minute")):
        if key in f and key in p:
            lines.append(f"  {label}: {f[key] - p[key]:+.2f} {unit}")
    return "\n".join(lines) or "  (not available)"


def fmt(p) -> str:
    top = np.argsort(p)[::-1][:3]
    return ", ".join(f"{EMOTIONS[i]} {p[i]:.2f}" for i in top)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
def fuse_late(A, T, y, fold_of, val_a, val_t, pos):
    out, alphas = np.zeros_like(A), []
    for fi, (va, vt) in enumerate(zip(val_a, val_t)):
        idx = np.array([pos[k] for k in va["keys"]])
        pa, pt = np.array(va["proba"]), np.array(vt["proba"])
        grid = np.linspace(0, 1, 21)
        scores = [macro_f1(y[idx], (a * pa + (1 - a) * pt).argmax(1)) for a in grid]
        best = max(scores)
        a = float(np.mean([g for g, s in zip(grid, scores) if s == best]))   # middle of the ties
        alphas.append(a)
        te = fold_of == fi
        out[te] = a * A[te] + (1 - a) * T[te]
    return out, alphas


def fuse_stack(A, T, y, fold_of, val_a, val_t, pos, seed):
    out = np.zeros_like(A)
    for fi, (va, vt) in enumerate(zip(val_a, val_t)):
        idx = np.array([pos[k] for k in va["keys"]])
        Xv = np.hstack([np.log(np.clip(va["proba"], 1e-6, 1)), np.log(np.clip(vt["proba"], 1e-6, 1))])
        te = fold_of == fi
        Xt = np.hstack([np.log(np.clip(A[te], 1e-6, 1)), np.log(np.clip(T[te], 1e-6, 1))])
        if len(np.unique(y[idx])) < 2:
            out[te] = (A[te] + T[te]) / 2
            continue
        lr = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000, random_state=seed)
        lr.fit(Xv, y[idx])
        pr = np.zeros((te.sum(), N7))
        pr[:, lr.classes_] = lr.predict_proba(Xt)
        out[te] = pr
    return out


def fuse_llm(A, T, ev, texts, feats, cache_path, model_name, load_4bit):
    from t3_text_llm import load_llm, score_labels
    cached = {}
    if cache_path.exists():
        for l in cache_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            cached[r["id"]] = np.array(r["proba"])
    out, tok, model = np.zeros_like(A), None, None
    device = get_device(verbose=False)
    with open(cache_path, "a", encoding="utf-8") as f:
        for i, s in enumerate(ev):
            msg = LLM_USER.format(text=texts[i], acoustic=acoustic_text(s, feats), audio=fmt(A[i]), textp=fmt(T[i]))
            cid = hashlib.sha1((model_name + LLM_SYSTEM + msg).encode("utf-8")).hexdigest()
            if cid not in cached:
                if model is None:
                    tok, model = load_llm(model_name, device, load_4bit)
                cached[cid] = score_labels(tok, model, [{"role": "system", "content": LLM_SYSTEM},
                                                        {"role": "user", "content": msg}], device)
                f.write(json.dumps({"id": cid, "proba": cached[cid].round(6).tolist()}) + "\n"); f.flush()
            out[i] = cached[cid]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Audio + text fusion")
    ap.add_argument("--audio", default="best")
    ap.add_argument("--text", default="best")
    ap.add_argument("--protocol", default="grouped")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--strategies", default="late,stack", help="late, stack, llm")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--load_4bit", action="store_true")
    a = ap.parse_args()

    dirs = ensure_dirs()
    ev, _ = load_evaluation()
    keys = [s["key"] for s in ev]
    pos = {k: i for i, k in enumerate(keys)}
    spk = np.array([s["speaker_id"] for s in ev])
    seed0 = int(a.seeds.split(",")[0])
    arun = resolve_run(dirs["audio_result"], a.audio, a.protocol, seed0)
    trun = resolve_run(dirs["text_result"], a.text, a.protocol, seed0)
    tag = f"{arun.relative_to(dirs['audio_result']).as_posix().replace('/', '-')}__{trun.name}"
    print(f"[INFO] audio: {arun}\n[INFO] text:  {trun}")

    from text_common import load_sessions
    texts = [s["text"] for s in load_sessions()]
    fpath = dirs["prep"] / "features" / "acoustic_own.json"
    feats = read_json(fpath) if fpath.exists() else {}

    rows = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        oa, ot = load_oof(arun, a.protocol, seed), load_oof(trun, a.protocol, seed)
        if oa["keys"] != keys or ot["keys"] != keys:
            raise SystemExit("[ERROR] branch results were computed on different sessions or order. "
                             "Re-run both branches after the same p5_make_splits.py.")
        y = np.array(oa["y7"])
        fold_of = np.array(oa["fold_of"])
        if not np.array_equal(fold_of, np.array(ot["fold_of"])):
            raise SystemExit("[ERROR] audio and text used different folds; fusion would leak.")
        A, T = oa["test_proba"], ot["test_proba"]
        base = dirs["fusion_result"] / tag / a.protocol / f"seed{seed}"
        base.mkdir(parents=True, exist_ok=True)
        agree = plot_agreement(y, A.argmax(1), T.argmax(1), "audio", "text",
                               base / "agreement.png", f"audio vs text | {a.protocol} | seed {seed}")
        with open(base / "agreement.json", "w", encoding="utf-8") as f:
            json.dump(agree, f, indent=2)

        fused = {"audio": A, "text": T}
        info = {}
        strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
        if "late" in strategies:
            fused["late"], info["late_alphas"] = fuse_late(A, T, y, fold_of, oa["val"], ot["val"], pos)
        if "stack" in strategies:
            fused["stack"] = fuse_stack(A, T, y, fold_of, oa["val"], ot["val"], pos, seed)
        if "llm" in strategies:
            fused["llm"] = fuse_llm(A, T, ev, texts, feats, dirs["fusion_result"] / "llm_cache.jsonl",
                                    a.llm, a.load_4bit)

        print(f"\n== fusion | {a.protocol} | seed {seed} | "
              f"audio-only-right {agree['7 emotions']['audio_only']}, text-only-right {agree['7 emotions']['text_only']}")
        for name, P in fused.items():
            pred = P.argmax(1)
            out = base / name
            oof = OOF(keys)
            oof.test[:] = P; oof.fold_of[:] = fold_of
            out.mkdir(parents=True, exist_ok=True)
            oof.save(out / "oof.json", y)                         # read by the multimodal stage
            write_report(out, y, pred, P, spk, keys, title=f"fusion {name} | {a.protocol} | seed {seed}",
                         extra_cols={"transcript": texts},
                         config={"audio_run": str(arun), "text_run": str(trun), "strategy": name,
                                 "protocol": a.protocol, "seed": seed, **info,
                                 **({"llm": a.llm, "llm_prompt_sha1": LLM_SHA1} if name == "llm" else {})},
                         seed=seed, verbose=False)
            m = summarise(y, pred)
            txt = ""
            if name != "text":
                t = paired_bootstrap_test(y, pred, T.argmax(1), spk, "macro_f1", seed=seed)
                txt = f"vs text {t['delta']:+.3f} p={t['p_value']:.3f}{' *' if t['significant_05'] else ''}"
            print(f"   {name:<7} acc7 {m['acc7']:.3f}  F1_7 {m['f1_7']:.3f}  acc3 {m['acc3']:.3f}  F1_3 {m['f1_3']:.3f}   {txt}")
            rows.append({"seed": seed, "strategy": name, **m})
        if "late_alphas" in info:
            print(f"   late: audio weight per fold {np.round(info['late_alphas'], 2).tolist()}")

    with open(dirs["fusion_result"] / tag / f"summary_{a.protocol}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n[INFO] -> {dirs['fusion_result'] / tag}")


if __name__ == "__main__":
    main()
