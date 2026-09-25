"""
06_session_fusion.py
═══════════════════════════════════════════════════════════════════════════════
Combine the audio branch (04) and the text branch (05) into a final
session-level emotion prediction.

Three fusion strategies are run and compared. Reporting all three is the
point -- if the LLM only matches a weighted average of the two branches, it is
not earning its cost, and a reviewer will ask.

  1. LATE FUSION      weighted average of the two branches' probabilities.
                      The alpha weight is fitted on the training folds only.
                      This is the baseline the LLM has to beat.

  2. JOINT MLP        both feature sets concatenated, one model trained on the
                      combination. Learns cross-modal interactions that
                      probability averaging cannot.

  3. LLM FUSION       Qwen reasons over the transcript, the discretised
                      speaking-style attributes, and BOTH branches' calibrated
                      probabilities, then commits to a label.

    python 06_session_fusion.py --cv louo
    python 06_session_fusion.py --cv all --strategies late,joint
    python 06_session_fusion.py --cv louo --strategies llm

NOTES ON THE LLM STAGE
─────────────────────────────────────────────────────────────────────────────
C2SER's finding is that audio-language models hallucinate: they invent
ungrounded context ("maybe he is preparing for an exam") and then justify a
wrong label with it. Their fix is a chain of thought that walks content ->
speaking style -> inference, grounded in discretised acoustic attributes.
With a 1.5B model the hallucination risk is higher, so the prompt here is
constrained: fixed label set, structured output, and the reasoning must cite
the attributes it was given.

MASKSER's complementary finding is that one modality tends to dominate the
fusion. In this corpus the speakers are neutral-toned and the transcripts are
content-rich, so the expected failure is the LLM ignoring prosody and copying
the text branch. The prompt therefore shows BOTH branches' probabilities and
explicitly asks the model to state when it overrides one. The
`llm_agreement_report` at the end quantifies how often it just followed text.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from common import (
    EMOTION_LABELS, N_EMOTIONS, THREE_CLASS_LABELS, IDX7_TO_IDX3, DEVICE,
    load_video_index, load_features, resolve_stem,
    dominant_emotion, emotions_to_multilabel,
    make_splits, train_fold, evaluate_fold, run_cv, save_results,
)

import importlib
audio_mod = importlib.import_module("04_audio_emotion")
text_mod = importlib.import_module("05_text_emotion")

ROOT = Path(r"C:\Users\2022g\Downloads\RAVDESS_IEMOCAP_CREMAD")
QWEN_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


# ─────────────────────────────────────────────────────────────────────────────
# Aligned dataset: audio features and text features for the SAME sessions
# ─────────────────────────────────────────────────────────────────────────────
def build_aligned(video_index: dict, features: dict, use_delta: bool = True,
                  encoder: str = "emotion2vec", hc_blocks: str = ""):
    """
    Build audio and text matrices over the intersection of sessions that have
    both, so every fusion strategy sees exactly the same sessions.
    """
    a_metas, Xa, y7a, yma, ga = audio_mod.build_dataset(
        video_index, features, use_delta=use_delta,
        encoder=encoder, hc_blocks=hc_blocks)
    t_metas, Xt, y7t, ymt, gt, texts, extra = text_mod.build_dataset(
        video_index, features)

    a_key = {(m["user_id"], m["session_id"]): i for i, m in enumerate(a_metas)}
    t_key = {(m["user_id"], m["session_id"]): i for i, m in enumerate(t_metas)}
    shared = sorted(set(a_key) & set(t_key))

    if not shared:
        raise SystemExit("[ERROR] No sessions have both audio and text features.")

    ai = [a_key[k] for k in shared]
    ti = [t_key[k] for k in shared]

    dropped = len(a_key) - len(shared)
    if dropped > 0:
        print(f"[WARN] {dropped} session(s) dropped: present in the audio set "
              f"but with no usable transcript.")

    metas = [a_metas[i] for i in ai]
    return {
        "metas": metas,
        "Xa": Xa[ai], "Xt": Xt[ti],
        "y7": y7a[ai], "ym": yma[ai],
        "groups": ga[ai],
        "texts": [texts[i] for i in ti],
        "extra": extra[ti],
        "keys": shared,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1 — Late fusion
# ─────────────────────────────────────────────────────────────────────────────
def fit_alpha(pa_tr: np.ndarray, pt_tr: np.ndarray, y_tr: np.ndarray) -> float:
    """
    Grid-search the audio/text mixing weight on TRAINING folds only.
    alpha = 1.0 means audio only, 0.0 means text only.
    """
    best_a, best_f1 = 0.5, -1.0
    for a in np.linspace(0.0, 1.0, 21):
        pred = (a * pa_tr + (1 - a) * pt_tr).argmax(axis=1)
        f1 = f1_score(y_tr, pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_a = f1, float(a)
    return best_a


def run_late_fusion(data: dict, method: str, n_folds: int, seed: int) -> dict:
    """
    For each fold: train both branches on the train split, produce probabilities
    for both train and test, fit alpha on train, apply to test.
    """
    Xa, Xt = data["Xa"], data["Xt"]
    y7, ym, groups = data["y7"], data["ym"], data["groups"]

    all_true, all_pred, alphas = [], [], []

    for tr, te, name in make_splits(y7, groups, method, n_folds, seed):
        if len(np.unique(y7[tr])) < 2:
            continue

        m_a, n_a = train_fold(Xa[tr], y7[tr], ym[tr], Xa[te], y7[te], ym[te])
        m_t, n_t = train_fold(Xt[tr], y7[tr], ym[tr], Xt[te], y7[te], ym[te])

        # Train-split probabilities for fitting alpha (in-sample, but alpha is
        # a single scalar so the overfitting risk is negligible).
        pa_tr = np.array(evaluate_fold(m_a, Xa[tr], y7[tr], ym[tr], n_a)["proba7"])
        pt_tr = np.array(evaluate_fold(m_t, Xt[tr], y7[tr], ym[tr], n_t)["proba7"])
        alpha = fit_alpha(pa_tr, pt_tr, y7[tr])
        alphas.append(alpha)

        pa_te = np.array(evaluate_fold(m_a, Xa[te], y7[te], ym[te], n_a)["proba7"])
        pt_te = np.array(evaluate_fold(m_t, Xt[te], y7[te], ym[te], n_t)["proba7"])
        pred = (alpha * pa_te + (1 - alpha) * pt_te).argmax(axis=1)

        all_true.append(y7[te])
        all_pred.append(pred)
        print(f"  {name:<16} n={len(te):<4} alpha={alpha:.2f} "
              f"(alpha=1 audio-only, 0 text-only)  "
              f"acc={accuracy_score(y7[te], pred):.3f}")

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    t3, p3 = IDX7_TO_IDX3[y_true], IDX7_TO_IDX3[y_pred]

    print(f"\n  Mean alpha across folds: {np.mean(alphas):.3f}")
    print(f"  (>0.5 means audio carries more weight than text)")

    return {
        "strategy": "late_fusion",
        "mean_alpha": float(np.mean(alphas)),
        "alphas": alphas,
        "pooled": {
            "acc7": float(accuracy_score(y_true, y_pred)),
            "f1_macro7": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted7": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "acc3_post": float(accuracy_score(t3, p3)),
            "f1_macro3_post": float(f1_score(t3, p3, average="macro", zero_division=0)),
        },
        "y7_true": y_true.tolist(),
        "y7_pred": y_pred.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2 — Joint MLP
# ─────────────────────────────────────────────────────────────────────────────
def run_joint(data: dict, method: str, n_folds: int, seed: int) -> dict:
    X = np.hstack([data["Xa"], data["Xt"]]).astype(np.float32)
    print(f"  Joint feature dim: {X.shape[1]} "
          f"(audio {data['Xa'].shape[1]} + text {data['Xt'].shape[1]})")
    res = run_cv(X, data["y7"], data["ym"], data["groups"],
                 method, n_folds=n_folds, seed=seed, tag="fusion_joint")
    res["strategy"] = "joint_mlp"
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3 — LLM fusion
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a speech emotion recognition specialist working with recordings "
    "of ordinary people describing their day. The speakers are NOT actors: "
    "their delivery is usually calm and understated even when what they "
    "describe is emotionally significant.\n\n"
    "Two independent classifiers have already analysed this session. You are "
    "given both of their outputs, the transcript, and measured speaking-style "
    "attributes.\n\n"
    "Rules:\n"
    "- A calm delivery does NOT imply neutral. Judge content and prosody together.\n"
    "- Use 'neutral' only when the content AND the speaking style are both "
    "unremarkable.\n"
    "- Speaking-style levels are relative to THIS speaker's own baseline, not "
    "to the population. 'High pitch' means high for this person.\n"
    "- Ground your reasoning ONLY in the transcript and attributes provided. "
    "Do not invent context, backstory, or motivations that are not stated.\n"
    "- If you disagree with a classifier, say which one and why in one clause.\n"
    "- Be concise: at most three sentences of reasoning."
)


def build_prompt(transcript: str, attrs: dict, delta: dict,
                 audio_probs: np.ndarray, text_probs: np.ndarray) -> str:
    def top3(p):
        idx = np.argsort(p)[::-1][:3]
        return ", ".join(f"{EMOTION_LABELS[i]} {p[i]:.2f}" for i in idx)

    delta_str = ""
    if delta:
        delta_str = (
            f"\nCompared with the same speaker reading a fixed passage in this "
            f"same session:\n"
            f"  - pitch difference:  {delta.get('d_pitch_hz', 0):+.1f} Hz\n"
            f"  - energy difference: {delta.get('d_energy_lufs', 0):+.1f} LUFS\n"
            f"  - rate difference:   {delta.get('d_rate_wps', 0):+.2f} words/sec\n"
            f"  (This cancels the speaker's own baseline and the recording "
            f"conditions, so these differences reflect state, not identity.)\n"
        )

    return (
        f'Transcript of spontaneous speech:\n"{transcript}"\n\n'
        f"Speaking style (relative to this speaker's baseline):\n"
        f"  - Pitch:         {attrs.get('pitch_level', 'Medium')} "
        f"({attrs.get('pitch_hz', 0):.0f} Hz)\n"
        f"  - Energy:        {attrs.get('energy_level', 'Medium')} "
        f"({attrs.get('energy_lufs', 0):.1f} LUFS)\n"
        f"  - Speaking rate: {attrs.get('speaking_rate_level', 'Medium')} "
        f"({attrs.get('speaking_rate_wps', 0):.2f} words/sec)\n"
        f"{delta_str}\n"
        f"Acoustic classifier (top 3): {top3(audio_probs)}\n"
        f"Text classifier    (top 3): {top3(text_probs)}\n\n"
        f"Reason in at most three sentences, then output exactly one line:\n"
        f"EMOTION: <label>\n\n"
        f"Valid labels: {', '.join(EMOTION_LABELS)}"
    )


def load_qwen(model_id: str = QWEN_MODEL_ID):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if DEVICE != "cpu" else torch.float32,
        device_map="auto", trust_remote_code=True)
    mdl.eval()
    print(f"[INFO] {model_id} loaded")
    return tok, mdl


@torch.no_grad()
def query_qwen(prompt: str, tok, mdl, max_new_tokens: int = 160) -> tuple[str, str]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(mdl.device)
    out = mdl.generate(**inputs, max_new_tokens=max_new_tokens,
                       temperature=0.1, do_sample=True,
                       pad_token_id=tok.eos_token_id)
    resp = tok.decode(out[0][inputs["input_ids"].shape[-1]:],
                      skip_special_tokens=True).strip()
    return resp, parse_emotion(resp)


def parse_emotion(response: str) -> str:
    m = re.search(r"EMOTION:\s*([a-zA-Z]+)", response, re.IGNORECASE)
    if m:
        raw = m.group(1).lower().strip()
        if raw in EMOTION_LABELS:
            return raw
    low = response.lower()
    for e in EMOTION_LABELS:
        if e in low:
            return e
    return "neutral"


def run_llm_fusion(data: dict, features: dict, method: str,
                   n_folds: int, seed: int) -> dict:
    Xa, Xt = data["Xa"], data["Xt"]
    y7, ym, groups = data["y7"], data["ym"], data["groups"]
    metas, texts = data["metas"], data["texts"]

    # Recover the freestyle feature record per session for attrs / deltas.
    rec_by_key = {}
    for rec in features.values():
        if rec.get("video_type") == "freestyle":
            rec_by_key[(rec.get("user_id"), rec.get("session_id"))] = rec

    tok, mdl = load_qwen()

    all_true, all_pred, audio_pred, text_pred, traces = [], [], [], [], []

    for tr, te, name in make_splits(y7, groups, method, n_folds, seed):
        if len(np.unique(y7[tr])) < 2:
            continue
        print(f"\n  --- {name} ({len(te)} sessions) ---")

        m_a, n_a = train_fold(Xa[tr], y7[tr], ym[tr], Xa[te], y7[te], ym[te])
        m_t, n_t = train_fold(Xt[tr], y7[tr], ym[tr], Xt[te], y7[te], ym[te])
        pa = np.array(evaluate_fold(m_a, Xa[te], y7[te], ym[te], n_a)["proba7"])
        pt = np.array(evaluate_fold(m_t, Xt[te], y7[te], ym[te], n_t)["proba7"])

        for j, i in enumerate(te):
            key = (metas[i]["user_id"], metas[i]["session_id"])
            rec = rec_by_key.get(key, {})
            prompt = build_prompt(
                texts[i], rec.get("acoustic_attrs", {}),
                rec.get("delta_vs_passage", {}), pa[j], pt[j])
            try:
                reasoning, label = query_qwen(prompt, tok, mdl)
            except Exception as exc:
                print(f"    [WARN] Qwen failed on {key}: {exc}")
                reasoning, label = f"[ERROR] {exc}", EMOTION_LABELS[pa[j].argmax()]

            all_true.append(int(y7[i]))
            all_pred.append(EMOTION_LABELS.index(label))
            audio_pred.append(int(pa[j].argmax()))
            text_pred.append(int(pt[j].argmax()))
            traces.append({
                "user_id": key[0], "session_id": key[1],
                "true": EMOTION_LABELS[y7[i]],
                "audio": EMOTION_LABELS[pa[j].argmax()],
                "text": EMOTION_LABELS[pt[j].argmax()],
                "llm": label,
                "reasoning": reasoning[:400],
            })
            mark = "OK " if all_pred[-1] == all_true[-1] else "-- "
            print(f"    {mark}{key[1][:16]:<18} true={EMOTION_LABELS[y7[i]]:<9} "
                  f"audio={traces[-1]['audio']:<9} text={traces[-1]['text']:<9} "
                  f"llm={label}")

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    ap = np.array(audio_pred)
    tp = np.array(text_pred)
    t3, p3 = IDX7_TO_IDX3[y_true], IDX7_TO_IDX3[y_pred]

    # How much is the LLM actually contributing?
    follows_text = float(np.mean(y_pred == tp))
    follows_audio = float(np.mean(y_pred == ap))
    independent = float(np.mean((y_pred != tp) & (y_pred != ap)))

    print(f"\n  {'-' * 70}")
    print(f"  LLM agreement analysis")
    print(f"  {'-' * 70}")
    print(f"    matches the text branch    {follows_text:.1%}")
    print(f"    matches the audio branch   {follows_audio:.1%}")
    print(f"    disagrees with both        {independent:.1%}")
    print("  If the LLM matches text almost always, it is not fusing -- it is")
    print("  copying the stronger branch, which is the failure mode MASKSER's")
    print("  KL alignment term exists to prevent. Compare against late fusion.")

    return {
        "strategy": "llm_fusion",
        "pooled": {
            "acc7": float(accuracy_score(y_true, y_pred)),
            "f1_macro7": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_weighted7": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "acc3_post": float(accuracy_score(t3, p3)),
            "f1_macro3_post": float(f1_score(t3, p3, average="macro", zero_division=0)),
        },
        "agreement": {
            "follows_text": follows_text,
            "follows_audio": follows_audio,
            "independent": independent,
        },
        "y7_true": y_true.tolist(),
        "y7_pred": y_pred.tolist(),
        "traces": traces,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Session-level fusion")
    p.add_argument("--cv", default="louo", choices=["kfold", "grouped", "louo", "all"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strategies", default="late,joint",
                   help="Comma-separated: late,joint,llm")
    p.add_argument("--features", type=Path, default=ROOT / "code/data/features.jsonl")
    p.add_argument("--video_index", type=Path,
                   default=ROOT / "firebase_data" / "video_index.json")
    p.add_argument("--out_dir", type=Path, default=ROOT / "code/data/results")
    p.add_argument("--no_delta", action="store_true")
    p.add_argument("--encoder", default="emotion2vec",
                   choices=["emotion2vec", "custom", "both", "none"],
                   help="Audio representation the audio branch classifies from")
    p.add_argument("--hc_blocks", default="",
                   help="Handcrafted families for the audio branch")
    args = p.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    methods = ["kfold", "grouped", "louo"] if args.cv == "all" else [args.cv]

    print(f"[INFO] Device: {DEVICE}")
    video_index = load_video_index(args.video_index)
    features = load_features(args.features)

    data = build_aligned(video_index, features, use_delta=not args.no_delta,
                         encoder=args.encoder, hc_blocks=args.hc_blocks)
    print(f"[INFO] {len(data['y7'])} aligned sessions | "
          f"{len(set(data['groups']))} users")
    print(f"[INFO] audio dim={data['Xa'].shape[1]}  text dim={data['Xt'].shape[1]}")

    summary = defaultdict(dict)

    for method in methods:
        print(f"\n{'#' * 74}")
        print(f"#  CV = {method}")
        print(f"{'#' * 74}")

        if "late" in strategies:
            print(f"\n>> LATE FUSION")
            r = run_late_fusion(data, method, args.folds, args.seed)
            save_results(r, args.out_dir / f"fusion_late_{method}.json")
            summary[method]["late"] = r["pooled"]

        if "joint" in strategies:
            print(f"\n>> JOINT MLP")
            r = run_joint(data, method, args.folds, args.seed)
            save_results(r, args.out_dir / f"fusion_joint_{method}.json")
            summary[method]["joint"] = r["pooled"]

        if "llm" in strategies:
            print(f"\n>> LLM FUSION (Qwen)")
            r = run_llm_fusion(data, features, method, args.folds, args.seed)
            save_results(r, args.out_dir / f"fusion_llm_{method}.json")
            summary[method]["llm"] = r["pooled"]

    # ── Final table ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 74}")
    print(f"  FUSION SUMMARY")
    print(f"{'=' * 74}")
    print(f"  {'CV':<10} {'strategy':<12} {'acc7':>7} {'mF1_7':>7} "
          f"{'wF1_7':>7} {'acc3':>7} {'mF1_3':>7}")
    print(f"  {'-' * 62}")
    for method in methods:
        for strat, q in summary[method].items():
            print(f"  {method:<10} {strat:<12} {q['acc7']:>7.4f} "
                  f"{q['f1_macro7']:>7.4f} {q.get('f1_weighted7', 0):>7.4f} "
                  f"{q.get('acc3_post', q.get('acc3_head', 0)):>7.4f} "
                  f"{q.get('f1_macro3_post', q.get('f1_macro3_head', 0)):>7.4f}")

    print(f"\n[INFO] Results -> {args.out_dir}")


if __name__ == "__main__":
    main()
