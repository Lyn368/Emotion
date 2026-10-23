"""
a6_audio_clinical.py - Elevated anxiety (GAD-7 >= 10) and depression (PHQ-9 >= 10) from the voice.

Audio counterpart of the clinical block in the metadata branch, so both tables can
be read side by side. Labels come from complete_firebase_data.json, the same source
the metadata branch uses. Same design:

  complete-case  keep only sessions where the TARGET questionnaire was answered.
                 An absent questionnaire is never counted as "Not elevated".
  label          Elevated if score >= 10, else Not elevated.
  config A       audio only. No clinical score enters the model.
  config B       audio + the OTHER clinical scale (anxiety sees PHQ-9, depression
                 sees GAD-7), with its missing flag, as in the metadata branch.
  protocol       Leave-One-User-Out; SVM and MLP; majority-class and
                 stratified-random baselines.

Added for audio (the metadata block does not have these):

  ref: other scale only   the other scale with no audio. Config B only means
                          something next to this row: B minus this row is what
                          the voice adds on top of the other questionnaire.
                          The table reports it as "delta vs ref" with a paired
                          participant-bootstrap CI (A rows are compared with
                          the gender-only row). p_perm on a B row tests B
                          against chance, not the audio's contribution.
  ref: gender only        when Firebase has background_information.gender. Both scales
                          tend to run higher in women and the voice encodes sex,
                          so audio can beat chance just by detecting sex. Audio
                          has to beat this row as well.
  feature sets            <src>_free        freestyle clip only (absolute voice)
                          <src>_delta_only  freestyle minus passage (within-session reference)
                          <src>_delta       both; the vector used for emotion
                          The reference cancels the speaker's baseline. GAD-7 and
                          PHQ-9 are close to constant per participant, so the
                          reference may remove the very signal wanted here.
                          free vs delta_only answers that directly.
  participant level       the labels behave like traits, so the effective sample
                          size is the number of participants, not of sessions.
                          Reported: sessions per participant, share of the label
                          explained by participant identity, a participant-level
                          vote, a participant-cluster bootstrap CI, and a
                          permutation test that shuffles labels BETWEEN
                          participants (block permutation).

AUROC is computed on the pooled out-of-fold scores (SVM decision values, MLP
probabilities). Many LOUO test folds contain a single class, so per-fold AUROC
and F1 are undefined; every metric here is pooled over all test folds.

Run after a1_acoustic_features.py (and a2_encoder_embeddings.py for encoder sets):

    python a6_audio_clinical.py
    python a6_audio_clinical.py --firebase ..\\..\\firebase_data\\complete_firebase_data.json
    python a6_audio_clinical.py --feature_sets acoustic_free,wavlm_base_plus_free,wavlm_base_plus_delta_only
    python a6_audio_clinical.py --n_perm 0             (quick run, no permutation test)

Output (audio_result/clinical/):
    clinical_summary.csv         one row per scale x config x model (aggregates only)
    clinical_results.json        settings, diagnostics and all rows (aggregates only)
    predictions_clinical.json    per-session predictions. Contains participant ids and
                                 clinical labels: keep it out of git (the existing
                                 'predictions_*.json' ignore rule covers it).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from audio_common import load_evaluation
from config import ensure_dirs

# ---------------------------------------------------------------------------
# Settings. Keep the model settings identical to the metadata branch's run_task,
# otherwise the two tables are not comparable.
# ---------------------------------------------------------------------------
CUTOFF = 10                                    # GAD-7 >= 10, PHQ-9 >= 10 (moderate)
MAX_SCORE = {"gad7": 21, "phq9": 27}
N_ITEMS = {"gad7": 7, "phq9": 9}
SCALES = [("Anxiety (GAD-7>=10)", "gad7", "phq9"),
          ("Depression (PHQ-9>=10)", "phq9", "gad7")]
SVM_PARAMS = dict(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
LINSVM_PARAMS = dict(C=1.0, class_weight="balanced", max_iter=10000)    # --models linsvm (MODMA baseline)
MLP_PARAMS = dict(hidden_layer_sizes=(64,), alpha=1e-2, max_iter=500)
MLP_BALANCE = True        # MLPClassifier has no class_weight: duplicate minority rows per training fold
PCA_VAR = 0.95            # encoder sets only (hundreds of dims per clip)
N_RANDOM = 1000           # draws averaged for the stratified-random baseline
N_BOOT = 2000             # participant-cluster bootstrap resamples
NAN = float("nan")


# ---------------------------------------------------------------------------
# Clinical labels, straight from complete_firebase_data.json
# (the same source the metadata branch builds its records from)
# ---------------------------------------------------------------------------
def _num(v):
    if isinstance(v, bool):
        return float(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _item(x):
    """One questionnaire answer: a number, a numeric string, or a dict holding one."""
    if isinstance(x, dict):
        for k in ("value", "score", "answer", "response"):
            if k in x:
                return _num(x[k])
        return None
    return _num(x)


def find_firebase(arg, dirs) -> Path | None:
    """--firebase, else complete*firebase*.json in the usual folders around v2."""
    if arg:
        p = Path(arg)
        if not p.exists():
            raise SystemExit(f"[ERROR] {p} not found")
        return p
    here = Path.cwd()
    for base in (here, here.parent, here.parent.parent, dirs["prep"], dirs["prep"].parent):
        for sub in ("", "firebase_data", "data"):
            hits = sorted((base / sub).glob("complete*firebase*.json")) if (base / sub).is_dir() else []
            if hits:
                return hits[0]
    return None


GENDER_KEY = re.compile(r"^(gender|sex|genero|género|sexo)$")


def gender_of(bg) -> str | None:
    if not isinstance(bg, dict):
        return None
    for k, v in bg.items():
        if GENDER_KEY.match(str(k).strip().lower()) and isinstance(v, (str, int, float)):
            g = str(v).strip().lower()
            return g or None
    return None


def firebase_clinical(path: Path):
    """
    GAD-7 and PHQ-9 per session:
      uid -> sessions -> session_id -> questionnaires -> anxiety_disorder / depression_symptoms -> responses

    A score exists only if the questionnaire is there with the expected number of
    numeric answers. Otherwise the session is MISSING for that scale (never 0).
    A PHQ-9 with a 10th answer is accepted and the 10th is ignored: it is the
    functional-difficulty question, which is not part of the score.
    Gender comes from background_information.
    """
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    users = data.get("users", data) if isinstance(data, dict) else {}
    clin, gender = {}, {}
    rep = {k: {"questionnaires": 0, "valid": 0, "rejected": 0, "answers_per_questionnaire": Counter(),
               "item_values": Counter()} for k in N_ITEMS}
    for uid, u in users.items():
        if not isinstance(u, dict):
            continue
        bg = u.get("background_information") or next(
            (v for k, v in u.items() if "background" in str(k).lower() and isinstance(v, dict)), None)
        g = gender_of(bg)
        if g:
            gender[str(uid)] = g
        for sid, s in (u.get("sessions") or {}).items():
            if not isinstance(s, dict):
                continue
            q = s.get("questionnaires") or {}
            row = {"uid": str(uid)}
            for k, pat in (("gad7", "anxi"), ("phq9", "depress")):
                row[k] = None
                block = next((b for name, b in q.items()
                              if pat in str(name).lower() and isinstance(b, dict)), None)
                resp = block.get("responses") if block else None
                if isinstance(resp, dict):                      # Firebase sometimes stores arrays as {"0":..}
                    resp = [resp[i] for i in sorted(resp, key=lambda x: (len(str(x)), str(x)))]
                if not isinstance(resp, list) or not resp:
                    continue
                r = rep[k]
                r["questionnaires"] += 1
                r["answers_per_questionnaire"][len(resp)] += 1
                items = [_item(x) for x in resp]
                n = N_ITEMS[k]
                if any(v is None for v in items) or not (len(items) == n or (k == "phq9" and len(items) == n + 1)):
                    r["rejected"] += 1
                    continue
                r["item_values"].update(int(v) for v in items[:n])
                score = sum(items[:n])
                if not 0 <= score <= MAX_SCORE[k]:
                    r["rejected"] += 1
                    continue
                row[k] = float(score)
                r["valid"] += 1
            clin[str(sid)] = row
    return clin, gender, rep, len(users)


# ---------------------------------------------------------------------------
# Audio features
# ---------------------------------------------------------------------------
def acoustic_blocks(ev, dirs):
    """Raw a1 features per clip (no per-speaker normalisation, which would erase traits)."""
    p = dirs["prep"] / "features" / "acoustic_own.json"
    if not p.exists():
        raise SystemExit(f"[ERROR] {p} not found. Run: python a1_acoustic_features.py")
    feats = json.loads(p.read_text(encoding="utf-8"))
    names = sorted({n for d in feats.values() if isinstance(d, dict) for n, v in d.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)})

    def vec(clip):
        d = feats.get(clip) if clip else None
        if not isinstance(d, dict):
            return None
        return np.array([_num(d.get(n)) if _num(d.get(n)) is not None else np.nan for n in names])

    F = np.full((len(ev), len(names)), np.nan)
    P = np.full_like(F, np.nan)
    for i, s in enumerate(ev):
        f, pv = vec(s["freestyle_clip"]), vec(s.get("passage_clip"))
        if f is not None:
            F[i] = f
        if pv is not None:
            P[i] = pv
    return F, P


def encoder_blocks(ev, dirs, model):
    """a2 segment embeddings, averaged over segments and layers -> one vector per clip."""
    d = dirs["embeddings"] / model
    arr_p, idx_p = d / "own.npy", d / "own_index.jsonl"
    if not arr_p.exists():
        raise SystemExit(f"[ERROR] No embeddings for {model}. Run: python a2_encoder_embeddings.py --model {model}")
    arr = np.load(arr_p, mmap_mode="r")
    rows = {}
    for line in idx_p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["clip_id"]] = (r["start"], r["n"])
    D = arr.shape[-1]

    def vec(clip):
        if not clip or clip not in rows:
            return None
        s, n = rows[clip]
        return np.asarray(arr[s:s + n], dtype=np.float32).reshape(-1, D).mean(0)

    F = np.full((len(ev), D), np.nan)
    P = np.full_like(F, np.nan)
    for i, s in enumerate(ev):
        f, pv = vec(s["freestyle_clip"]), vec(s.get("passage_clip"))
        if f is not None:
            F[i] = f
        if pv is not None:
            P[i] = pv
    return F, P


def parse_set(name: str):
    for suf in ("_delta_only", "_delta", "_free"):
        if name.endswith(suf):
            return name[: -len(suf)], suf[1:]
    return name, "free"


def build_sets(names, ev, dirs):
    cache, sets = {}, {}
    for name in names:
        src, mode = parse_set(name)
        if src not in cache:
            cache[src] = acoustic_blocks(ev, dirs) if src == "acoustic" else encoder_blocks(ev, dirs, src)
        F, P = cache[src]
        X = F if mode == "free" else (F - P if mode == "delta_only" else np.hstack([F, F - P]))
        sets[name] = {"X": X, "pca": src != "acoustic", "uses_passage": mode != "free"}
    has_free = np.all([~np.isnan(F).all(1) for F, _ in cache.values()], axis=0)
    has_pass = np.all([~np.isnan(P).all(1) for _, P in cache.values()], axis=0)
    return sets, has_free, has_pass


# ---------------------------------------------------------------------------
# Models and evaluation
# ---------------------------------------------------------------------------
def make_model(kind, use_pca, n_audio, n_total, seed):
    clf = (SVC(**SVM_PARAMS) if kind == "svm" else LinearSVC(**LINSVM_PARAMS) if kind == "linsvm"
           else MLPClassifier(**MLP_PARAMS, random_state=seed))
    if not use_pca:
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)
    audio = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                          PCA(n_components=PCA_VAR, svd_solver="full", random_state=seed))
    if n_audio == n_total:
        return make_pipeline(audio, clf)
    # Config B with an encoder: PCA on the audio block only, so the clinical column survives.
    pre = ColumnTransformer([("audio", audio, list(range(n_audio))),
                             ("other", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                              list(range(n_audio, n_total)))])
    return make_pipeline(pre, clf)


def oversample(X, y, rng):
    c = np.bincount(y, minlength=2)
    if c.min() == 0 or c[0] == c[1]:
        return X, y
    small = int(np.argmin(c))
    extra = rng.choice(np.where(y == small)[0], c.max() - c.min(), replace=True)
    idx = np.concatenate([np.arange(len(y)), extra])
    return X[idx], y[idx]


def louo(kind, X, y, groups, use_pca, n_audio, seed):
    """Out-of-fold predictions and scores, one participant held out per fold."""
    pred, score = np.zeros(len(y), int), np.zeros(len(y))
    rng = np.random.RandomState(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        for tr, te in LeaveOneGroupOut().split(X, y, groups):
            ytr = y[tr]
            if len(np.unique(ytr)) < 2:                      # nothing to learn in this fold
                pred[te], score[te] = ytr[0], 0.0
                continue
            Xtr = X[tr]
            if kind == "mlp" and MLP_BALANCE:
                Xtr, ytr = oversample(Xtr, ytr, rng)
            m = make_model(kind, use_pca, n_audio, X.shape[1], seed).fit(Xtr, ytr)
            pred[te] = m.predict(X[te])
            score[te] = m.decision_function(X[te]) if kind != "mlp" else m.predict_proba(X[te])[:, 1]
    return pred, score


def _bal_acc(y, pred) -> float:
    """Balanced accuracy in numpy (same definition as sklearn), for the resampling loops."""
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def metrics(y, pred, score=None) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = {"acc": accuracy_score(y, pred),
             "bal_acc": balanced_accuracy_score(y, pred),
             "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0),
             "f1_elevated": f1_score(y, pred, pos_label=1, zero_division=0),
             "auroc": roc_auc_score(y, score) if score is not None and len(np.unique(y)) == 2 else NAN}
    return {k: float(v) for k, v in m.items()}


def participant_vote(y, pred, groups) -> dict:
    """Each participant: majority true label vs majority predicted label."""
    t, p = [], []
    for g in np.unique(groups):
        i = groups == g
        t.append(int(y[i].mean() >= 0.5))
        p.append(int(pred[i].mean() >= 0.5))
    t, p = np.array(t), np.array(p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ba = balanced_accuracy_score(t, p) if len(np.unique(t)) == 2 else NAN
    return {"participant_bal_acc": float(ba), "participants_correct": int((t == p).sum()),
            "participants": int(len(t))}


def cluster_ci(y, pred, groups, rng):
    """95% CI of balanced accuracy, resampling PARTICIPANTS with replacement."""
    parts = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in parts}
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([by[g] for g in rng.choice(parts, len(parts), replace=True)])
        if len(np.unique(y[idx])) == 2:
            vals.append(_bal_acc(y[idx], pred[idx]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (NAN, NAN)


def paired_delta(y, p1, p2, groups, rng):
    """bal_acc(p1) - bal_acc(p2) on the same sessions, participant-cluster bootstrap CI."""
    parts = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in parts}
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([by[g] for g in rng.choice(parts, len(parts), replace=True)])
        if len(np.unique(y[idx])) == 2:
            vals.append(_bal_acc(y[idx], p1[idx]) - _bal_acc(y[idx], p2[idx]))
    lo, hi = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (NAN, NAN)
    return _bal_acc(y, p1) - _bal_acc(y, p2), lo, hi


def block_permute(y, groups, rng):
    """Each participant receives another participant's labels (resampled to its length)."""
    parts = np.unique(groups)
    out = np.empty_like(y)
    for g, donor in zip(parts, rng.permutation(parts)):
        i = np.where(groups == g)[0]
        yd = y[groups == donor]
        out[i] = yd if len(yd) == len(i) else rng.choice(yd, len(i), replace=True)
    return out


def _perm_once(kind, X, y, groups, use_pca, n_audio, seed, k):
    yp = block_permute(y, groups, np.random.RandomState(seed * 100003 + k))
    if len(np.unique(yp)) < 2:
        return NAN
    pred, _ = louo(kind, X, yp, groups, use_pca, n_audio, seed)
    return _bal_acc(yp, pred)


def perm_test(kind, X, y, groups, use_pca, n_audio, seed, observed, n_perm, jobs):
    null = Parallel(n_jobs=jobs)(delayed(_perm_once)(kind, X, y, groups, use_pca, n_audio, seed, k)
                                 for k in range(n_perm))
    null = np.array([v for v in null if not math.isnan(v)])
    if not len(null):
        return {"p_perm": NAN, "null_mean": NAN, "null_95": NAN}
    return {"p_perm": float((1 + (null >= observed - 1e-12).sum()) / (1 + len(null))),
            "null_mean": float(null.mean()), "null_95": float(np.percentile(null, 95))}


def baselines(y, groups, rng) -> dict:
    maj, prev = np.zeros(len(y), int), np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(y, y, groups):
        maj[te] = int(np.bincount(y[tr], minlength=2).argmax())
        prev[te] = y[tr].mean()
    rows = {"Majority class": metrics(y, maj) | participant_vote(y, maj, groups)}
    # Expected value of a classifier that guesses "Elevated" with the training prevalence.
    draws = (rng.random_sample((N_RANDOM, len(y))) < prev).astype(int)
    ms = [metrics(y, d) | participant_vote(y, d, groups) for d in draws]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rows["Stratified random"] = {k: float(np.nanmean([m[k] for m in ms])) for k in ms[0]}
    return rows


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def between_share(v, groups):
    """Share of the variance explained by participant identity (1 - within / total)."""
    v = v.astype(float)
    tot = ((v - v.mean()) ** 2).sum()
    if tot == 0:
        return NAN
    within = sum(((v[groups == g] - v[groups == g].mean()) ** 2).sum() for g in np.unique(groups))
    return float(1 - within / tot)


def describe(scale, y, raw, groups, gender_by_part) -> dict:
    parts = np.unique(groups)
    rate = {g: float(y[groups == g].mean()) for g in parts}
    mostly = sum(r >= 0.5 for r in rate.values())
    d = {"sessions": int(len(y)), "participants": int(len(parts)),
         "sessions_per_participant": len(y) / len(parts),
         "elevated_sessions": int(y.sum()),
         "participants_mostly_elevated": int(mostly),
         "participants_ever_elevated": int(sum(r > 0 for r in rate.values())),
         "participants_switching": int(sum(0 < r < 1 for r in rate.values())),
         "label_share_participant": between_share(y, groups),
         "score_share_participant": between_share(raw, groups)}
    print(f"\n[{scale}] complete-case records: {d['sessions']} | distinct sessions: {d['sessions']} "
          f"| inflation factor: 1.00x  (one audio row per session)")
    print(f"  participants: {d['participants']} | sessions per participant: {d['sessions_per_participant']:.1f}")
    print(f"  Elevated: {d['elevated_sessions']}/{d['sessions']} sessions "
          f"({100 * d['elevated_sessions'] / d['sessions']:.1f}%) | participants mostly Elevated: "
          f"{mostly}/{d['participants']} | ever Elevated: {d['participants_ever_elevated']} "
          f"| switch label: {d['participants_switching']}")
    print(f"  variance explained by participant identity: label {d['label_share_participant']:.2f} "
          f"| raw score {d['score_share_participant']:.2f}")
    if gender_by_part:
        tab = defaultdict(lambda: [0, 0])
        for g in parts:
            tab[gender_by_part.get(g) or "unknown"][int(rate[g] >= 0.5)] += 1
        d["gender_by_label"] = {k: {"elevated": v[1], "not_elevated": v[0]} for k, v in tab.items()}
        print("  participants by gender (majority label): "
              + " | ".join(f"{k} {v[1]}E/{v[0]}N" for k, v in sorted(tab.items())))
    if mostly < 2 or len(parts) - mostly < 2:
        print("  [WARN] fewer than 2 participants on one side at participant level: "
              "results are descriptive, not evidence.")
    if d["label_share_participant"] > 0.6:
        print(f"  [NOTE] the label is mostly a participant trait: each fold learns from "
              f"{len(parts) - 1} people, not {d['sessions']} independent sessions.")
    return d


def fmt(v, nd=3):
    return "  -  " if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def print_table(scale, rows):
    print(f"\n{'=' * 130}\n{scale}\n{'=' * 130}")
    print(f"{'config':<44}{'model':<6}{'acc':>7}{'bal_acc':>9}{'95% CI':>16}{'mF1':>7}{'F1_E':>7}"
          f"{'AUROC':>7}{'people':>8}{'p_perm':>8}   delta bal_acc vs ref [95% CI]")
    for r in rows:
        ci = f"[{fmt(r.get('bal_acc_lo'), 2)},{fmt(r.get('bal_acc_hi'), 2)}]" if r.get("bal_acc_lo") is not None else ""
        ppl = f"{r['participants_correct']:.0f}/{r['participants']:.0f}"
        dl = (f"{r['delta_ref']:+.3f} [{r['delta_lo']:+.2f},{r['delta_hi']:+.2f}] vs {r['vs_ref'].split(' (')[0][5:]}"
              if r.get("vs_ref") else "")
        print(f"{r['config'][:43]:<44}{r['model']:<6}{fmt(r['acc']):>7}{fmt(r['bal_acc']):>9}{ci:>16}"
              f"{fmt(r['macro_f1']):>7}{fmt(r['f1_elevated']):>7}{fmt(r.get('auroc')):>7}{ppl:>8}"
              f"{fmt(r.get('p_perm')):>8}   {dl}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="GAD-7 / PHQ-9 >= 10 from audio (LOUO)")
    ap.add_argument("--firebase", default=None,
                    help="complete_firebase_data.json (found automatically in firebase_data/ if omitted)")
    ap.add_argument("--feature_sets", default="acoustic_free,acoustic_delta_only,acoustic_delta",
                    help="acoustic_* and/or <encoder>_free|_delta|_delta_only (e.g. wavlm_base_plus_free)")
    ap.add_argument("--models", default="svm,mlp", help="svm (RBF), mlp, linsvm (linear SVM)")
    ap.add_argument("--n_perm", type=int, default=200, help="block permutations (0 = skip)")
    ap.add_argument("--perm_models", default="svm", help="models that get the permutation test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers for the permutation test")
    a = ap.parse_args()

    t0 = time.time()
    dirs = ensure_dirs()
    out_dir = dirs["audio_result"] / "clinical"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(a.seed)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    perm_models = {m.strip() for m in a.perm_models.split(",") if m.strip()}

    # ---- labels ---------------------------------------------------------------
    fb_path = find_firebase(a.firebase, dirs)
    if fb_path is None:
        raise SystemExit("[ERROR] complete_firebase_data.json not found. Pass --firebase <path>.")
    clin, gender_by_uid, rep, n_users = firebase_clinical(fb_path)
    print(f"[INFO] firebase: {fb_path} | {n_users} participants | {len(clin)} sessions")
    for k, r in rep.items():
        vals = sorted(r["item_values"])
        print(f"  {k}: {r['questionnaires']} questionnaires, {r['valid']} valid, {r['rejected']} rejected "
              f"| answers per questionnaire {dict(sorted(r['answers_per_questionnaire'].items()))} "
              f"| answer values {vals[0] if vals else '-'}..{vals[-1] if vals else '-'}")
        if vals and (vals[-1] > 3 or vals[0] > 0):
            print(f"  [WARN] {k} answers are not coded 0-3. The >=10 cut-off assumes 0-3 per item; "
                  f"check how the app stores them before trusting the labels.")
        if r["questionnaires"] and r["rejected"] > 0.2 * r["questionnaires"]:
            print(f"  [WARN] {k}: many questionnaires rejected. Look at 'answers per questionnaire'.")

    ev, _ = load_evaluation()
    sid_of = [str(s.get("session_id") or s["key"].split("|")[-1]) for s in ev]
    in_fb = sum(sid in clin for sid in sid_of)
    print(f"[INFO] audio sessions: {len(ev)} | found in firebase: {in_fb} "
          f"| with GAD-7: {sum(clin.get(x, {}).get('gad7') is not None for x in sid_of)} "
          f"| with PHQ-9: {sum(clin.get(x, {}).get('phq9') is not None for x in sid_of)}")
    if in_fb == 0:
        raise SystemExit(f"[ERROR] no audio session id is in firebase. "
                         f"audio e.g. {sid_of[:2]} | firebase e.g. {list(clin)[:2]}")
    wrong_uid = sum(1 for s, sid in zip(ev, sid_of) if sid in clin and clin[sid]["uid"] != str(s["speaker_id"]))
    if wrong_uid:
        print(f"  [WARN] {wrong_uid} audio sessions belong to a different user in firebase than in the "
              f"audio index. LOUO groups use the audio speaker_id.")

    # ---- audio ----------------------------------------------------------------
    set_names = [s.strip() for s in a.feature_sets.split(",") if s.strip()]
    sets, has_free, has_pass = build_sets(set_names, ev, dirs)
    print(f"[INFO] sessions with freestyle features: {int(has_free.sum())}/{len(ev)} "
          f"| without a passage reference: {int((has_free & ~has_pass).sum())} (delta imputed in-fold)")

    spk = np.array([s["speaker_id"] for s in ev])
    gender_by_part = {}
    for sp, sid in zip(spk, sid_of):
        g = gender_by_uid.get(str(sp)) or gender_by_uid.get(clin.get(sid, {}).get("uid"))
        if g:
            gender_by_part.setdefault(sp, g)

    # ---- tasks ----------------------------------------------------------------
    all_rows, diagnostics, predictions = [], {}, {}
    for scale, key, other in SCALES:
        score = np.array([_num(clin.get(sid, {}).get(key)) if clin.get(sid, {}).get(key) is not None
                          else np.nan for sid in sid_of])
        cc = has_free & ~np.isnan(score)
        idx = np.where(cc)[0]
        if len(idx) == 0:
            print(f"\n[{scale}] no complete-case sessions; skipped.")
            continue
        y = (score[idx] >= CUTOFF).astype(int)
        groups = spk[idx]
        if len(np.unique(y)) < 2:
            print(f"\n[{scale}] no variation in complete-case data; skipped.")
            continue
        diagnostics[scale] = describe(scale, y, score[idx], groups, gender_by_part)

        oth = np.array([clin.get(sid_of[i], {}).get(other) for i in idx], dtype=float)
        other_cols = np.column_stack([oth, np.isnan(oth).astype(float)])   # value + missing flag

        tasks = []      # (config label, feature set, X, use_pca, n_audio)
        gvals = np.array([gender_by_part.get(g) or "unknown" for g in groups])
        if len(set(gvals) - {"unknown"}) >= 2:
            cats = sorted(set(gvals))
            tasks.append(("ref: gender only", "-", np.column_stack([gvals == c for c in cats]).astype(float), False, 0))
        ref_other = f"ref: {other.upper()} only (no audio)"
        tasks.append((ref_other, "-", other_cols, False, 0))
        for name in set_names:
            Xa = sets[name]["X"][idx]
            tasks.append((f"A: audio only | {name}", name, Xa, sets[name]["pca"], Xa.shape[1]))
            tasks.append((f"B: audio + {other.upper()} | {name}", name, np.hstack([Xa, other_cols]),
                          sets[name]["pca"], Xa.shape[1]))

        rows, preds_here = [], {}
        for bname, m in baselines(y, groups, rng).items():
            rows.append({"config": bname, "model": "-", "feature_set": "-"} | m)
        for label, fset, X, use_pca, n_audio in tasks:
            for kind in models:
                pred, sc = louo(kind, X, y, groups, use_pca, n_audio or X.shape[1], a.seed)
                r = {"config": label, "model": kind.upper(), "feature_set": fset,
                     "n_features": int(X.shape[1])} | metrics(y, pred, sc) | participant_vote(y, pred, groups)
                r["bal_acc_lo"], r["bal_acc_hi"] = cluster_ci(y, pred, groups, rng)
                if a.n_perm > 0 and kind in perm_models:
                    t1 = time.time()
                    r |= perm_test(kind, X, y, groups, use_pca, n_audio or X.shape[1], a.seed,
                                   r["bal_acc"], a.n_perm, a.jobs)
                    print(f"  perm {label} {kind}: p={r['p_perm']:.3f} ({time.time() - t1:.0f}s)")
                rows.append(r)
                preds_here[(label, r["model"])] = pred
                predictions.setdefault(scale, {})[f"{label} | {kind}"] = {
                    "keys": [ev[i]["key"] for i in idx], "y": y.tolist(),
                    "pred": pred.tolist(), "score": np.round(sc, 5).tolist()}

        # What the voice adds: A vs gender only (is it more than sex?), B vs other scale only.
        # p_perm on a B row tests B against chance, NOT the audio's contribution; this does.
        for r in rows:
            ref = (ref_other if r["config"].startswith("B:")
                   else "ref: gender only" if r["config"].startswith("A:") else None)
            if ref and (ref, r["model"]) in preds_here:
                dlt, lo, hi = paired_delta(y, preds_here[(r["config"], r["model"])],
                                           preds_here[(ref, r["model"])], groups, rng)
                r |= {"vs_ref": ref, "delta_ref": dlt, "delta_lo": lo, "delta_hi": hi}
        print_table(scale, rows)
        for r in rows:
            all_rows.append({"scale": scale, "n_sessions": int(len(y)),
                             "n_participants": int(len(np.unique(groups)))} | r)

    # ---- save -----------------------------------------------------------------
    cols = ["scale", "config", "feature_set", "model", "n_sessions", "n_participants", "n_features",
            "acc", "bal_acc", "bal_acc_lo", "bal_acc_hi", "macro_f1", "f1_elevated", "auroc",
            "participants_correct", "participants", "participant_bal_acc", "p_perm", "null_mean", "null_95",
            "vs_ref", "delta_ref", "delta_lo", "delta_hi"]
    with open(out_dir / "clinical_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    settings = {"cutoff": CUTOFF, "svm": SVM_PARAMS, "linsvm": LINSVM_PARAMS, "mlp": MLP_PARAMS, "mlp_balance": MLP_BALANCE,
                "pca_var_encoders": PCA_VAR, "n_random": N_RANDOM, "n_boot": N_BOOT,
                "protocol": "Leave-One-User-Out"} | vars(a)
    with open(out_dir / "clinical_results.json", "w", encoding="utf-8") as f:
        json.dump({"settings": settings, "firebase_file": str(fb_path),
                   "diagnostics": diagnostics, "rows": all_rows}, f, indent=2, default=str)
    with open(out_dir / "predictions_clinical.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"\n[DONE] {time.time() - t0:.0f}s -> {out_dir}")
    print("  predictions_clinical.json has participant ids + clinical labels: do not commit it.")


if __name__ == "__main__":
    main()