"""
12_multimodal_fusion.py
═══════════════════════════════════════════════════════════════════════════════
Session-level emotion from ALL modalities: audio (yours), video, wearable, and
the metadata logs. Two fusion strategies, same inputs, so the comparison
isolates the LLM rather than the architecture.

    # 1. ALWAYS run this first. It tells you what actually aligns.
    python 12_multimodal_fusion.py --audit

    # 2. Stacking (no LLM)
    python 12_multimodal_fusion.py --cv louo --fusion stack

    # 3. Both, plus per-modality ablation
    python 12_multimodal_fusion.py --cv louo --fusion both --ablate

UNIT OF ANALYSIS
─────────────────────────────────────────────────────────────────────────────
The session. A session can hold several logs with different self-reported
emotions, so the target is MULTI-LABEL, exactly as the audio branch already
treats it: a 7-position vector of what was reported, plus a single dominant
label resolved by the shared priority order in common.py. Wearable and
metadata arrive per log and are averaged up to the session.

WHY STACKING AND NOT CROSS-ATTENTION
─────────────────────────────────────────────────────────────────────────────
Cross-attention fuses sequences of representations and needs the data to
support its parameter count. Here each modality contributes a probability
vector of length 3 or 7 over a few hundred sessions. Attention over that has
far more capacity than evidence and would fit noise. Stacking a concatenated
probability vector is the standard, appropriate estimator at this scale, and
it has the property you want for the write-up: the learned weights say how
much each modality contributed, which is the actual research question.

Attention becomes the right tool when you fuse per-frame EMBEDDINGS from each
modality and have an order of magnitude more sessions. Say that in the paper
rather than leaving it unaddressed — it is the first thing a reviewer asks.

AVAILABILITY MASKS
─────────────────────────────────────────────────────────────────────────────
Video covers a minority of sessions. Zero-filling a missing modality tells the
model "all classes equally unlikely", which is a confident and false claim.
Every modality block therefore ships with a present/absent bit, so the model
can learn to route around a gap instead of reading it as evidence.

NO LLM OUTSIDE THE FUSION STAGE
─────────────────────────────────────────────────────────────────────────────
Unchanged from 06: the unimodal branches are discriminative models. The only
generative model in the pipeline sits here, in --fusion llm.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import (
    EMOTION_LABELS, N_EMOTIONS, THREE_CLASS_LABELS, N_THREE, IDX7_TO_IDX3,
    EMOTION_TO_3CLASS, DEVICE,
    load_video_index, load_features, resolve_stem,
    dominant_emotion, emotions_to_multilabel,
    make_splits, make_inner_split, train_fold_explicit, evaluate_fold,
    unweighted_accuracy, save_results,
)
from stats import cluster_bootstrap_ci, paired_bootstrap_test

ROOT = Path("/Users/ytinglin/Downloads/ProjectCE903/25-26_CE903-SP_team01/MLmodel/data")

# Teammate label vocabularies -> ours
EMO_ALIAS = {
    "anger": "angry", "angry": "angry",
    "disgust": "disgust",
    "fear": "fear", "fearful": "fear",
    "happiness": "happy", "happy": "happy", "joy": "happy",
    "neutral": "neutral",
    "sadness": "sad", "sad": "sad",
    "surprise": "surprise", "surprised": "surprise",
}
VAL_ALIAS = {"negative": "negative", "neutral": "neutral", "positive": "positive"}

MODALITIES = ["audio", "text", "wearable", "video", "metadata"]


def read_json(path: Path):
    """utf-8-sig: the wearable exports carry a BOM and CRLF line endings."""
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Teammate sources
# ─────────────────────────────────────────────────────────────────────────────
def load_wearable(path: Path, alias: dict, classes_out: list) -> dict:
    """{log_id: prob vector in OUR class order}. Keyed by log, not session."""
    d = read_json(path)
    classes = [alias.get(c.strip().lower()) for c in d["classes"]]
    out = {}
    for r in d["records"]:
        v = np.zeros(len(classes_out), dtype=np.float32)
        probs = r.get("probs", {})
        for src_name, canon in zip(d["classes"], classes):
            if canon in classes_out:
                v[classes_out.index(canon)] = float(probs.get(src_name, 0.0))
        s = v.sum()
        out[r["log_id"]] = v / s if s > 0 else v
    return out


def load_video(path: Path) -> dict:
    """{video_id: 3-class prob vector}."""
    d = read_json(path)
    out = {}
    for r in d["predictions"]:
        v = np.zeros(N_THREE, dtype=np.float32)
        for name, p in r.get("probabilities", {}).items():
            canon = VAL_ALIAS.get(name.strip().lower())
            if canon in THREE_CLASS_LABELS:
                v[THREE_CLASS_LABELS.index(canon)] = float(p)
        s = v.sum()
        out[r["video_id"]] = v / s if s > 0 else v
    return out


def load_metadata(path: Path) -> tuple[dict, dict]:
    """({session_id: [log records]}, {log_id: session_id})."""
    rows = read_json(path)
    by_session = defaultdict(list)
    log_to_session = {}
    for r in rows:
        by_session[r["session_id"]].append(r)
        log_to_session[r["log_id"]] = r["session_id"]
    return dict(by_session), log_to_session


# ─────────────────────────────────────────────────────────────────────────────
# Metadata -> features
# ─────────────────────────────────────────────────────────────────────────────
ACTIVITIES = ["sitting", "walking", "eating", "standing", "lying", "exercising"]


def metadata_vec(logs: list[dict], include_clinical: bool = False) -> np.ndarray:
    """
    Session-level metadata features, averaged over the session's logs.

    gad7 and phq9 are OFF by default. They are participant traits, near-constant
    for several participants across all their logs. Under speaker-disjoint CV
    they cannot help generalise to a new person; under any weaker split they act
    as a participant fingerprint. Turn them on only as an explicit ablation, and
    report it as one.
    """
    if not logs:
        return np.zeros(len(ACTIVITIES) + 7, dtype=np.float32)

    act = np.zeros(len(ACTIVITIES), dtype=np.float32)
    for r in logs:
        a = str(r.get("activity") or "").strip().lower()
        if a in ACTIVITIES:
            act[ACTIVITIES.index(a)] += 1.0
    if act.sum():
        act /= act.sum()

    def mean_of(key, default=0.0):
        vals = [float(r[key]) for r in logs
                if r.get(key) is not None and not isinstance(r.get(key), (list, dict))]
        return float(np.mean(vals)) if vals else default

    hours = []
    for r in logs:
        ts = r.get("log_timestamp_ms")
        if ts:
            hours.append(((ts / 1000) % 86400) / 3600.0)
    hour = float(np.mean(hours)) if hours else 12.0

    extra = [
        mean_of("temperature", 15.0) / 30.0,
        mean_of("weather_code", 0.0) / 10.0,
        mean_of("wentOut"),
        mean_of("didSocialize"),
        float(any(r.get("food") for r in logs)),
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
    ]
    v = np.concatenate([act, np.array(extra, dtype=np.float32)])

    if include_clinical:
        v = np.concatenate([v, np.array(
            [mean_of("gad7", 0.0) / 21.0, mean_of("phq9", 0.0) / 27.0],
            dtype=np.float32)])
    return np.nan_to_num(v).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Session assembly
# ─────────────────────────────────────────────────────────────────────────────
def audio_block(f_rec: dict | None, p_rec: dict | None) -> tuple[np.ndarray, float]:
    """
    Audio as a probability vector, so every modality enters on equal footing.

    Prefers the custom encoder's heads (from 09). Falls back to emotion2vec's
    own label as a one-hot, which is weaker but keeps the block present.
    """
    if not f_rec:
        return np.zeros(N_EMOTIONS + N_THREE, dtype=np.float32), 0.0

    p7 = np.asarray(f_rec.get("custom_probs7", []), dtype=np.float32)
    p3 = np.asarray(f_rec.get("custom_probs3", []), dtype=np.float32)
    if p7.size == N_EMOTIONS:
        if p3.size != N_THREE:
            p3 = np.zeros(N_THREE, dtype=np.float32)
            for i, v in enumerate(p7):
                p3[IDX7_TO_IDX3[i]] += v
        return np.concatenate([p7, p3]).astype(np.float32), 1.0

    lbl = (f_rec.get("emotion2vec_label") or "").strip().lower()
    v7 = np.zeros(N_EMOTIONS, dtype=np.float32)
    if lbl in EMOTION_LABELS:
        v7[EMOTION_LABELS.index(lbl)] = 1.0
    v3 = np.zeros(N_THREE, dtype=np.float32)
    for i, v in enumerate(v7):
        v3[IDX7_TO_IDX3[i]] += v
    return np.concatenate([v7, v3]).astype(np.float32), float(v7.sum() > 0)


def build_sessions(video_index: dict, features: dict, meta_by_session: dict,
                   wear7: dict, wear3: dict, video: dict,
                   include_clinical: bool = False) -> list[dict]:
    by_session = defaultdict(dict)
    for (uid, sid, vtype), info in video_index.items():
        by_session[(uid, sid)][vtype] = info

    rows = []
    for (uid, sid), tasks in sorted(by_session.items()):
        logs = meta_by_session.get(sid, [])

        # Multi-label target: every emotion reported in this session's logs.
        # The metadata logs are the authoritative source; the video index is
        # the fallback for sessions with no log record.
        reported = [EMO_ALIAS.get(str(r.get("emotion", "")).strip().lower())
                    for r in logs]
        reported = [e for e in reported if e in EMOTION_LABELS]
        if not reported:
            reported = (tasks.get("freestyle", {}).get("reported_emotions")
                        or tasks.get("passage", {}).get("reported_emotions") or [])
            reported = [e for e in reported if e in EMOTION_LABELS]
        if not reported:
            continue

        f_rec = resolve_stem(tasks.get("freestyle", {}), features)
        p_rec = resolve_stem(tasks.get("passage", {}), features)
        a_vec, a_ok = audio_block(f_rec, p_rec)

        w7 = [wear7[r["log_id"]] for r in logs if r["log_id"] in wear7]
        w3 = [wear3[r["log_id"]] for r in logs if r["log_id"] in wear3]
        w7v = np.mean(w7, axis=0) if w7 else np.zeros(N_EMOTIONS, dtype=np.float32)
        w3v = np.mean(w3, axis=0) if w3 else np.zeros(N_THREE, dtype=np.float32)

        vids = [info.get("video_id") for info in tasks.values() if info.get("video_id")]
        vprobs = [video[v] for v in vids if v in video]
        vvec = np.mean(vprobs, axis=0) if vprobs else np.zeros(N_THREE, dtype=np.float32)

        rows.append({
            "user_id": uid, "session_id": sid,
            "reported": sorted(set(reported)),
            "n_logs": len(logs),
            "audio": a_vec, "audio_ok": a_ok,
            "text_rec": f_rec,
            "wearable": np.concatenate([w7v, w3v]),
            "wearable_ok": float(bool(w7 or w3)),
            "video": vvec, "video_ok": float(bool(vprobs)),
            "metadata": metadata_vec(logs, include_clinical),
            "metadata_ok": float(bool(logs)),
        })
    return rows


def text_matrix(rows: list[dict]) -> np.ndarray:
    """Sentence embeddings of the freestyle transcript, encoded once."""
    texts = [(r["text_rec"] or {}).get("transcript", "") or "" for r in rows]
    if not any(t.strip() for t in texts):
        print("  [WARN] No transcripts found; the text block will be empty.")
        return np.zeros((len(rows), 0), dtype=np.float32)
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = enc.encode(texts, batch_size=32, convert_to_numpy=True,
                     show_progress_bar=False).astype(np.float32)
    ok = np.array([[float(bool(t.strip()))] for t in texts], dtype=np.float32)
    return np.hstack([emb, ok])


def assemble(rows: list[dict], modalities: list[str],
             text_X: np.ndarray | None) -> tuple[np.ndarray, list[str]]:
    blocks, names = [], []
    if "audio" in modalities:
        blocks.append(np.stack([r["audio"] for r in rows]))
        blocks.append(np.array([[r["audio_ok"]] for r in rows], dtype=np.float32))
        names += [f"audio_{l}" for l in EMOTION_LABELS] \
            + [f"audio3_{l}" for l in THREE_CLASS_LABELS] + ["audio_present"]
    if "text" in modalities and text_X is not None and text_X.shape[1]:
        blocks.append(text_X)
        names += [f"text{i}" for i in range(text_X.shape[1] - 1)] + ["text_present"]
    if "wearable" in modalities:
        blocks.append(np.stack([r["wearable"] for r in rows]))
        blocks.append(np.array([[r["wearable_ok"]] for r in rows], dtype=np.float32))
        names += [f"wear_{l}" for l in EMOTION_LABELS] \
            + [f"wear3_{l}" for l in THREE_CLASS_LABELS] + ["wear_present"]
    if "video" in modalities:
        blocks.append(np.stack([r["video"] for r in rows]))
        blocks.append(np.array([[r["video_ok"]] for r in rows], dtype=np.float32))
        names += [f"video3_{l}" for l in THREE_CLASS_LABELS] + ["video_present"]
    if "metadata" in modalities:
        blocks.append(np.stack([r["metadata"] for r in rows]))
        blocks.append(np.array([[r["metadata_ok"]] for r in rows], dtype=np.float32))
        names += [f"meta{i}" for i in range(rows[0]["metadata"].size)] + ["meta_present"]
    if not blocks:
        raise SystemExit("[ERROR] No modalities selected.")
    return np.hstack(blocks).astype(np.float32), names


# ─────────────────────────────────────────────────────────────────────────────
# Audit
# ─────────────────────────────────────────────────────────────────────────────
def audit(rows: list[dict]) -> None:
    n = len(rows)
    print(f"\n{'=' * 74}\n  ALIGNMENT AUDIT — {n} sessions, "
          f"{len({r['user_id'] for r in rows})} participants\n{'=' * 74}")

    print(f"\n  {'modality':<12}{'sessions':>10}{'coverage':>11}")
    for m in ("audio", "wearable", "video", "metadata"):
        k = sum(1 for r in rows if r[f"{m}_ok"] > 0)
        print(f"  {m:<12}{k:>10}{k / max(n, 1):>10.1%}")
    txt = sum(1 for r in rows if (r["text_rec"] or {}).get("transcript", "").strip())
    print(f"  {'text':<12}{txt:>10}{txt / max(n, 1):>10.1%}")

    combos = Counter(
        tuple(m for m in ("audio", "wearable", "video", "metadata") if r[f"{m}_ok"] > 0)
        for r in rows)
    print(f"\n  Modality combinations present:")
    for c, k in combos.most_common(8):
        print(f"    {k:5d}  {'+'.join(c) if c else '(none)'}")

    n_multi = sum(1 for r in rows if len(r["reported"]) > 1)
    print(f"\n  Sessions with more than one reported emotion: {n_multi} "
          f"({n_multi / max(n, 1):.1%})")
    print(f"  Logs per session: {dict(Counter(r['n_logs'] for r in rows))}")

    dom = Counter(dominant_emotion(r["reported"]) for r in rows)
    print(f"\n  Dominant-label distribution:")
    for l in EMOTION_LABELS:
        c = dom.get(l, 0)
        print(f"    {l:<10}{c:5d}  {'#' * int(40 * c / max(max(dom.values()), 1))}")
    small = [l for l in EMOTION_LABELS if dom.get(l, 0) < 20]
    if small:
        print(f"\n  [!] Under 20 sessions: {small}. Per-class F1 for these is noise.")
        print(f"      Report the 3-class result as the headline and show the")
        print(f"      7-class per-class counts next to its table.")

    per_user = Counter(r["user_id"] for r in rows)
    print(f"\n  Sessions per participant: "
          f"min {min(per_user.values())} max {max(per_user.values())}")


# ─────────────────────────────────────────────────────────────────────────────
# Stacking fusion
# ─────────────────────────────────────────────────────────────────────────────
def run_stack(X, y7, ym, groups, method, n_folds, seed, tag=""):
    preds, trues, gs, order = [], [], [], []
    for tr, te, _ in make_splits(y7, groups, method, n_folds, seed):
        if len(np.unique(y7[tr])) < 2:
            continue
        in_tr, in_val = make_inner_split(y7[tr], groups[tr], method, seed=seed)
        model, norm = train_fold_explicit(
            X[tr[in_tr]], y7[tr[in_tr]], ym[tr[in_tr]],
            X[tr[in_val]], y7[tr[in_val]], ym[tr[in_val]],
            device=DEVICE, seed=seed)
        res = evaluate_fold(model, X[te], y7[te], ym[te], norm, device=DEVICE)
        preds.append(np.array(res["y7_pred"]))
        trues.append(y7[te])
        gs.append(groups[te])
        order.extend(te.tolist())

    p = np.concatenate(preds); t = np.concatenate(trues); g = np.concatenate(gs)
    srt = np.argsort(np.array(order))
    return p[srt], t[srt], g[srt]


# ─────────────────────────────────────────────────────────────────────────────
# LLM fusion
# ─────────────────────────────────────────────────────────────────────────────
def llm_prompt(r: dict) -> str:
    def fmt(vec, labels):
        pairs = sorted(zip(labels, vec), key=lambda kv: -kv[1])[:4]
        return ", ".join(f"{l} {v:.2f}" for l, v in pairs if v > 0)

    lines = ["A person recorded a short session. Independent models each looked",
             "at one signal. Some signals are missing. Decide the emotion.", ""]

    if r["audio_ok"]:
        lines.append(f"VOICE  (7-class): {fmt(r['audio'][:N_EMOTIONS], EMOTION_LABELS)}")
    else:
        lines.append("VOICE  : not available")

    if r["wearable_ok"]:
        lines.append(f"WEARABLE (7-class): {fmt(r['wearable'][:N_EMOTIONS], EMOTION_LABELS)}")
        lines.append(f"WEARABLE (valence): {fmt(r['wearable'][N_EMOTIONS:], THREE_CLASS_LABELS)}")
    else:
        lines.append("WEARABLE: not available")

    if r["video_ok"]:
        lines.append(f"FACE   (valence only): {fmt(r['video'], THREE_CLASS_LABELS)}")
    else:
        lines.append("FACE   : not available")

    logs = r.get("_logs", [])
    if logs:
        acts = ", ".join(sorted({str(x.get('activity')) for x in logs if x.get('activity')}))
        w = logs[0]
        lines.append(f"CONTEXT: activity {acts or 'unknown'}; "
                     f"temperature {w.get('temperature')}C; "
                     f"went out {w.get('wentOut')}; socialised {w.get('didSocialize')}")

    t = (r["text_rec"] or {}).get("transcript", "").strip()
    if t:
        lines.append(f"WHAT THEY SAID: \"{t[:400]}\"")

    lines += [
        "",
        "The signals disagree sometimes. Weigh them; do not just copy the most",
        "confident one. A missing signal is not evidence of anything.",
        f"Answer with one word from: {', '.join(EMOTION_LABELS)}.",
        "Format: REASON: <one sentence> | EMOTION: <word>",
    ]
    return "\n".join(lines)


def run_llm(rows, y7, groups, seed):
    fusion_mod = importlib.import_module("06_session_fusion")
    tok, mdl = fusion_mod.load_qwen()

    preds, copied = [], Counter()
    for i, r in enumerate(rows, 1):
        try:
            _, label = fusion_mod.query_qwen(llm_prompt(r), tok, mdl)
        except Exception as exc:
            print(f"    [WARN] Qwen failed on {r['session_id']}: {exc}")
            label = "neutral"
        idx = EMOTION_LABELS.index(label) if label in EMOTION_LABELS \
            else EMOTION_LABELS.index("neutral")
        preds.append(idx)

        # Which modality did it agree with? This is the number that says
        # whether the LLM fused anything or just echoed the loudest input.
        if r["audio_ok"] and int(np.argmax(r["audio"][:N_EMOTIONS])) == idx:
            copied["audio"] += 1
        if r["wearable_ok"] and int(np.argmax(r["wearable"][:N_EMOTIONS])) == idx:
            copied["wearable"] += 1
        if i % 25 == 0:
            print(f"    [{i}/{len(rows)}]", flush=True)

    n = len(rows)
    print(f"\n  LLM agreement: audio {copied['audio'] / n:.1%}, "
          f"wearable {copied['wearable'] / n:.1%} of sessions")
    print("  If one of these is near 100%, the LLM is echoing that modality,")
    print("  not fusing. That is the finding, and it belongs in the paper.")
    return np.array(preds, dtype=np.int64)


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────
def score(y, p) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": float(accuracy_score(y, p)),
        "ua": unweighted_accuracy(y, p),
        "f1_macro": float(f1_score(y, p, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y, p, average="micro", zero_division=0)),
        "f1_weighted": float(f1_score(y, p, average="weighted", zero_division=0)),
    }


def report(name, y, p, g, seed) -> dict:
    s7 = score(y, p)
    s3 = score(IDX7_TO_IDX3[y], IDX7_TO_IDX3[p])
    ci = cluster_bootstrap_ci(y, p, g, metric="macro_f1", seed=seed)
    print(f"\n  {name}")
    print(f"    7-class  acc {s7['accuracy']:.4f}  UA {s7['ua']:.4f}  "
          f"macroF1 {s7['f1_macro']:.4f} [{ci['ci_low']:.3f},{ci['ci_high']:.3f}]  "
          f"microF1 {s7['f1_micro']:.4f}  wF1 {s7['f1_weighted']:.4f}")
    print(f"    3-class  acc {s3['accuracy']:.4f}  UA {s3['ua']:.4f}  "
          f"macroF1 {s3['f1_macro']:.4f}  wF1 {s3['f1_weighted']:.4f}")
    return {"7class": s7, "3class": s3, "ci_macro_f1": ci}


def main():
    p = argparse.ArgumentParser(description="Multimodal session fusion")
    p.add_argument("--data", type=Path, default=ROOT)
    p.add_argument("--features", type=Path, default=None)
    p.add_argument("--video_index", type=Path, default=None)
    p.add_argument("--wearable_e7", type=Path, default=None)
    p.add_argument("--wearable_v3", type=Path, default=None)
    p.add_argument("--video_preds", type=Path, default=None)
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--cv", default="louo", choices=["kfold", "grouped", "louo"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--modalities", default="audio,wearable,video,metadata")
    p.add_argument("--fusion", default="stack", choices=["stack", "llm", "both"])
    p.add_argument("--ablate", action="store_true",
                   help="Also run the full set minus each modality")
    p.add_argument("--include_clinical", action="store_true",
                   help="Ablation only: add gad7/phq9 as features")
    p.add_argument("--audit", action="store_true", help="Report alignment and exit")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    d = args.data
    video_index = load_video_index(args.video_index or d / "firebase_data" / "video_index.json")
    features = load_features(args.features or d / "features.jsonl")
    meta_by_session, _ = load_metadata(args.metadata or d / "metadata_records.json")
    wear7 = load_wearable(args.wearable_e7 or d / "predictions_emotion7",
                          EMO_ALIAS, EMOTION_LABELS)
    wear3 = load_wearable(args.wearable_v3 or d / "predictions_valence3",
                          VAL_ALIAS, THREE_CLASS_LABELS)
    video = load_video(args.video_preds or d / "predictions_per_video.json")

    print(f"[INFO] {len(video_index)} index entries | {len(features)} features | "
          f"{len(meta_by_session)} metadata sessions | {len(wear7)} wearable logs | "
          f"{len(video)} video predictions")

    rows = build_sessions(video_index, features, meta_by_session,
                          wear7, wear3, video, args.include_clinical)
    if not rows:
        raise SystemExit("[ERROR] No sessions assembled. Run --audit on the inputs.")
    for r, logs in zip(rows, (meta_by_session.get(r["session_id"], []) for r in rows)):
        r["_logs"] = logs

    audit(rows)
    if args.audit:
        return

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    text_X = text_matrix(rows) if "text" in modalities else None

    y7 = np.array([EMOTION_LABELS.index(dominant_emotion(r["reported"])) for r in rows])
    ym = np.stack([emotions_to_multilabel(r["reported"]) for r in rows]).astype(np.float32)
    groups = np.array([r["user_id"] for r in rows])

    results, preds = {}, {}

    if args.fusion in ("stack", "both"):
        X, names = assemble(rows, modalities, text_X)
        print(f"\n[INFO] Stacking input: {X.shape[1]} dims over {X.shape[0]} sessions")
        pred, y_s, g_s = run_stack(X, y7, ym, groups, args.cv, args.folds, args.seed)
        results["stack"] = report(f"STACK ({'+'.join(modalities)})", y_s, pred, g_s, args.seed)
        preds["stack"] = pred
        y_ref, g_ref = y_s, g_s

        if args.ablate:
            for drop in modalities:
                rest = [m for m in modalities if m != drop]
                if not rest:
                    continue
                Xa, _ = assemble(rows, rest, text_X)
                pa, ya, ga = run_stack(Xa, y7, ym, groups, args.cv, args.folds, args.seed)
                results[f"stack-no-{drop}"] = report(f"STACK minus {drop}", ya, pa, ga, args.seed)
                preds[f"stack-no-{drop}"] = pa
                t = paired_bootstrap_test(y_ref, preds["stack"], pa, g_ref,
                                          metric="macro_f1", seed=args.seed)
                print(f"      dropping {drop}: Δ macro F1 {t['delta']:+.4f} "
                      f"[{t['delta_ci_low']:+.3f},{t['delta_ci_high']:+.3f}] "
                      f"p={t['p_value']:.4f}")

    if args.fusion in ("llm", "both"):
        print(f"\n{'=' * 74}\n  LLM FUSION (Qwen)\n{'=' * 74}")
        pred_llm = run_llm(rows, y7, groups, args.seed)
        results["llm"] = report("LLM", y7, pred_llm, groups, args.seed)
        preds["llm"] = pred_llm

        if "stack" in preds and len(preds["stack"]) == len(pred_llm):
            t = paired_bootstrap_test(y7, pred_llm, preds["stack"], groups,
                                      metric="macro_f1", seed=args.seed)
            print(f"\n  LLM vs stacking: Δ macro F1 {t['delta']:+.4f} "
                  f"[{t['delta_ci_low']:+.3f},{t['delta_ci_high']:+.3f}] "
                  f"p={t['p_value']:.4f}")
            print("  The LLM has to beat stacking on the SAME inputs to have")
            print("  earned its cost. Matching it is a negative result worth "
                  "reporting.")

    out = args.out or d / "results" / f"multimodal_{args.cv}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "cv": args.cv, "modalities": modalities,
        "n_sessions": len(rows), "n_participants": int(len(set(groups))),
        "results": results,
    }, indent=2, default=float))
    print(f"\n[INFO] Results -> {out}")


if __name__ == "__main__":
    main()
