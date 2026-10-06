"""
m1_multimodal.py - Add teammates' modalities (video, physiological, metadata) to audio + text.

Step 1, inspect their files and write a draft configuration you then CHECK:
    python m1_multimodal.py --inspect
    -> prints what each file contains and writes multimodal_sources.json

Step 2, run the fusion with that configuration:
    python m1_multimodal.py --protocol grouped
    python m1_multimodal.py --audio best --text best --protocol louo --seeds 42,43,44,45,46

What it reports, for 7 emotions and for 3 valence classes:
    audio+text                (our baseline)
    audio+text+<each source>  (does that modality help?)
    all                       (every source together)
Each combination is a logistic-regression stack trained per fold on the fold's
VALIDATION sessions and applied to its test sessions, like f1_fusion.py.

LEAKAGE - read before trusting any number:
  * Teammates' predictions must be OUT-OF-FOLD for speaker-disjoint splits. If their
    model saw a session during training, its prediction for that session is too good
    and every fused number is inflated. Share preprocessing/splits.json with them.
  * Metadata fields that look like the label (emotion, label, reported, valence,
    mood, feel) are dropped automatically.
  * Any single source or field that predicts the label with >= 95% accuracy is
    flagged: that almost always means leakage, not a great model.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from audio_common import N7, load_evaluation, macro_f1, read_json, summarise
from config import V2_DIR, ensure_dirs, project_root, sources
from f1_fusion import load_oof, resolve_run
from labels import EMOTIONS, TO_VALENCE, VALENCE
from report import write_report
from stats import paired_bootstrap_test

CONFIG = V2_DIR / "multimodal_sources.json"
TEAM_FILES = ["metadata_records.json", "predictions_per_video.json", "predictions_emotion7", "predictions_valence3"]
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
LEAKY = re.compile(r"emotion|label|reported|valence|mood|feel|target|class", re.I)
# identifiers are not features: user ids let the model recognise speakers (defeats LOUO)
ID_LIKE = re.compile(r"(^|_)(id|uid|key)$|user|participant|subject|session|video|path|url|file|timestamp|date|time", re.I)
NAMES7 = {"anger": "angry", "angry": "angry", "disgust": "disgust", "disgusted": "disgust", "fear": "fear",
          "fearful": "fear", "joy": "happy", "happiness": "happy", "happy": "happy", "neutral": "neutral",
          "sadness": "sad", "sad": "sad", "surprise": "surprise", "surprised": "surprise"}
NAMES3 = {"negative": "negative", "neg": "negative", "neutral": "neutral", "neu": "neutral",
          "positive": "positive", "pos": "positive"}
V_IDX = np.array([VALENCE.index(TO_VALENCE[e]) for e in EMOTIONS])


# ---------------------------------------------------------------------------
# Reading any file format
# ---------------------------------------------------------------------------
def read_any(path: Path) -> list[dict]:
    """JSON (list / dict of records / nested) or CSV / TSV, with or without extension."""
    txt = path.read_text(encoding="utf-8-sig", errors="ignore").strip()
    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        dialect = csv.Sniffer().sniff(txt[:4096], delimiters=",\t;")
        return list(csv.DictReader(io.StringIO(txt), dialect=dialect))
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict):
        vals = list(data.values())
        if vals and all(isinstance(v, dict) for v in vals):             # {id: {...}}
            return [{"_key": k, **v} for k, v in data.items()]
        for v in vals:                                                   # {"records": [...]}
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Mapping teammates' ids to our sessions
# ---------------------------------------------------------------------------
def session_lookup(ev) -> dict[str, tuple[str, str]]:
    """Every id form we know -> (session key, clip type)."""
    from p1_index_datasets import find_video_index
    src = sources(project_root())
    _, entries = find_video_index(src["own_index_dir"])
    clip2sess = {}
    for s in ev:
        clip2sess[s["freestyle_clip"]] = (s["key"], "freestyle")
        if s.get("passage_clip"):
            clip2sess[s["passage_clip"]] = (s["key"], "passage")
    look = {}
    for e in entries:
        stem = re.split(r"[/\\]", e.get("local_path", "") or "")[-1].rsplit(".", 1)[0]
        hit = clip2sess.get(f"own__{stem}")
        if not hit:
            continue
        vid = (e.get("video_id") or "").lower()
        for k in {stem.lower(), vid, vid.replace("-webm", "").replace(".webm", ""),
                  (UUID_RE.search(vid).group(0) if UUID_RE.search(vid) else "")}:
            if k:
                look[k] = hit
    for s in ev:                                                         # session ids too
        look[s["session_id"].lower()] = (s["key"], "session")
        look[s["key"].lower()] = (s["key"], "session")
    return look


def norm_id(v) -> list[str]:
    v = str(v).strip().lower()
    out = [v, v.replace("-webm", "").replace(".webm", ""), re.split(r"[/\\]", v)[-1].rsplit(".", 1)[0]]
    m = UUID_RE.search(v)
    if m:
        out.append(m.group(0))
    return out


def match_key(rec, key_field, look):
    for v in ([rec.get(key_field)] if key_field else rec.values()):
        if isinstance(v, (str, int)):
            for k in norm_id(v):
                if k in look:
                    return look[k]
    return None


# ---------------------------------------------------------------------------
# Inspect -> draft configuration
# ---------------------------------------------------------------------------
def detect(recs, look) -> dict:
    """Guess key field, probability fields or label field, and the label space."""
    cfg = {"key_field": None, "kind": None, "fields": None, "label_field": None, "space": None}
    best = (0, None)
    for f in recs[0].keys():
        n = sum(1 for r in recs[:300] if any(k in look for k in norm_id(r.get(f, ""))))
        if n > best[0]:
            best = (n, f)
    cfg["key_field"] = best[1]
    cols = {c.lower().strip(): c for c in recs[0].keys()}
    p7 = {NAMES7[c]: cols[c] for c in cols if c in NAMES7 and to_float(recs[0][cols[c]]) is not None}
    p3 = {NAMES3[c]: cols[c] for c in cols if c in NAMES3 and to_float(recs[0][cols[c]]) is not None}
    if len(p7) >= 5:
        cfg.update(kind="proba", fields={e: p7.get(e) for e in EMOTIONS}, space="emotion7")
    elif len(p3) == 3:
        cfg.update(kind="proba", fields={v: p3[v] for v in VALENCE}, space="valence3")
    else:
        for c, orig in cols.items():                                     # list of numbers
            v = recs[0][orig]
            if isinstance(v, list) and len(v) in (3, 7) and all(to_float(x) is not None for x in v):
                cfg.update(kind="proba_list", fields=orig, space="emotion7" if len(v) == 7 else "valence3")
                return cfg
        for c, orig in cols.items():                                     # predicted label
            vals = Counter(str(r.get(orig, "")).strip().lower() for r in recs[:300])
            if sum(n for v, n in vals.items() if v in NAMES7) > 0.8 * sum(vals.values()) and \
                    re.search(r"pred|output|result", c):
                cfg.update(kind="label", label_field=orig, space="emotion7")
                return cfg
            if sum(n for v, n in vals.items() if v in NAMES3) > 0.8 * sum(vals.values()) and \
                    re.search(r"pred|output|result", c):
                cfg.update(kind="label", label_field=orig, space="valence3")
                return cfg
        cfg.update(kind="features")
    return cfg


def inspect(ev):
    root = project_root()
    look = session_lookup(ev)
    draft = []
    for name in TEAM_FILES:
        p = root / name
        if not p.exists():
            cands = list(root.glob(name + ".*"))
            p = cands[0] if cands else p
        print(f"\n== {p.name}")
        if not p.exists():
            print("   not found"); continue
        recs = read_any(p)
        if not recs:
            print("   could not read any records"); continue
        print(f"   {len(recs)} records | fields: {', '.join(list(recs[0].keys())[:20])}")
        print(f"   first record: {json.dumps(recs[0], ensure_ascii=False)[:300]}")
        cfg = detect(recs, look)
        matched = sum(1 for r in recs if match_key(r, cfg["key_field"], look))
        print(f"   key field: {cfg['key_field']}  -> {matched}/{len(recs)} records map to our sessions")
        print(f"   detected: {cfg['kind']}" + (f" ({cfg['space']})" if cfg["space"] else ""))
        draft.append({"name": p.stem.replace("predictions_", ""), "file": p.name, **cfg, "use": True,
                      "clip": "any"})
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump(draft, f, indent=2)
    print(f"\n[INFO] draft configuration -> {CONFIG}")
    print("       CHECK IT before running: key_field, kind (proba / proba_list / label / features),")
    print("       fields, space (emotion7 / valence3). Set \"use\": false to skip a source.")


# ---------------------------------------------------------------------------
# Loading configured sources as per-session matrices
# ---------------------------------------------------------------------------
def load_source(cfg, ev, look):
    """-> (matrix n_sessions x d, space, present mask). Several rows per session are averaged."""
    recs = read_any(project_root() / cfg["file"])
    keys = [s["key"] for s in ev]
    pos = {k: i for i, k in enumerate(keys)}
    rows = defaultdict(list)
    for r in recs:
        hit = match_key(r, cfg.get("key_field"), look)
        if not hit or (cfg.get("clip", "any") not in ("any", hit[1]) and hit[1] != "session"):
            continue
        if cfg["kind"] == "proba":
            v = [to_float(r.get(cfg["fields"][c])) if cfg["fields"].get(c) else 0.0
                 for c in (EMOTIONS if cfg["space"] == "emotion7" else VALENCE)]
        elif cfg["kind"] == "proba_list":
            v = [to_float(x) for x in r[cfg["fields"]]]
        elif cfg["kind"] == "label":
            names = NAMES7 if cfg["space"] == "emotion7" else NAMES3
            space = EMOTIONS if cfg["space"] == "emotion7" else VALENCE
            lab = names.get(str(r.get(cfg["label_field"], "")).strip().lower())
            if lab is None:
                continue
            v = [1.0 if c == lab else 0.0 for c in space]
        else:
            continue
        if any(x is None for x in v):
            continue
        rows[pos[hit[0]]].append(np.array(v, float))
    d = 7 if cfg["space"] == "emotion7" else 3
    M = np.full((len(ev), d), 1.0 / d)
    present = np.zeros(len(ev), bool)
    for i, vs in rows.items():
        m = np.mean(vs, 0)
        M[i] = m / m.sum() if m.sum() > 0 else 1.0 / d
        present[i] = True
    return M, cfg["space"], present


def load_features(cfg, ev, look):
    """Numeric / categorical metadata per session, with label-like fields removed."""
    recs = read_any(project_root() / cfg["file"])
    pos = {s["key"]: i for i, s in enumerate(ev)}
    per = defaultdict(dict)
    for r in recs:
        hit = match_key(r, cfg.get("key_field"), look)
        if hit:
            per[pos[hit[0]]].update(r)
    dropped, ids, num, cat = [], [], [], []
    for f in (cfg.get("fields") or sorted({k for d in per.values() for k in d})):
        if f == cfg.get("key_field") or f == "_key":
            continue
        if LEAKY.search(f):
            dropped.append(f); continue
        if ID_LIKE.search(f):
            ids.append(f); continue
        vals = [per[i].get(f) for i in per]
        if all(to_float(v) is not None for v in vals if v not in (None, "")):
            num.append(f)
        elif 1 < len({str(v) for v in vals}) <= 12:
            cat.append(f)
    cols, X = [], []
    for i in range(len(ev)):
        d, row = per.get(i, {}), []
        row += [to_float(d.get(f)) if to_float(d.get(f)) is not None else np.nan for f in num]
        row += [1.0 if str(d.get(f)) == c else 0.0 for f in cat for c in sorted({str(per[j].get(f)) for j in per})]
        X.append(row)
    cols = num + [f"{f}={c}" for f in cat for c in sorted({str(per[j].get(f)) for j in per})]
    X = np.array(X, float)
    X = np.where(np.isnan(X), np.nanmean(X, 0) if X.size else 0, X)
    if dropped:
        print(f"   [{cfg['name']}] dropped label-like fields: {', '.join(dropped)}")
    if ids:
        print(f"   [{cfg['name']}] ignored identifier fields: {', '.join(ids)}")
    return np.nan_to_num(X), cols, np.array([i in per for i in range(len(ev))])


def logp(P):
    return np.log(np.clip(P, 1e-6, 1))


def main():
    ap = argparse.ArgumentParser(description="Multimodal fusion with teammates' outputs")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--audio", default="best")
    ap.add_argument("--text", default="best")
    ap.add_argument("--protocol", default="grouped")
    ap.add_argument("--seeds", default="42")
    a = ap.parse_args()

    dirs = ensure_dirs()
    ev, _ = load_evaluation()
    if a.inspect:
        inspect(ev); return
    if not CONFIG.exists():
        raise SystemExit("[ERROR] run  python m1_multimodal.py --inspect  first, then check multimodal_sources.json")

    look = session_lookup(ev)
    keys = [s["key"] for s in ev]
    pos = {k: i for i, k in enumerate(keys)}
    spk = np.array([s["speaker_id"] for s in ev])
    seed0 = int(a.seeds.split(",")[0])
    arun = resolve_run(dirs["audio_result"], a.audio, a.protocol, seed0)
    trun = resolve_run(dirs["text_result"], a.text, a.protocol, seed0)

    team = {}
    for cfg in read_json(CONFIG):
        if not cfg.get("use", True):
            continue
        if cfg["kind"] == "features":
            X, cols, present = load_features(cfg, ev, look)
            if X.shape[1]:
                team[cfg["name"]] = ("features", X, present)
                print(f"   [{cfg['name']}] {X.shape[1]} metadata features, {present.sum()}/{len(ev)} sessions")
        else:
            M, space, present = load_source(cfg, ev, look)
            team[cfg["name"]] = (space, M, present)
            print(f"   [{cfg['name']}] {space} predictions for {present.sum()}/{len(ev)} sessions")

    rows = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        oa, ot = load_oof(arun, a.protocol, seed), load_oof(trun, a.protocol, seed)
        if oa["keys"] != keys or ot["keys"] != keys:
            raise SystemExit("[ERROR] branch results do not match the current splits.json")
        y7 = np.array(oa["y7"]); y3 = V_IDX[y7]
        fold_of = np.array(oa["fold_of"])
        val_sets = [np.array([pos[k] for k in v["keys"]]) for v in oa["val"]]
        # audio / text probabilities for EVERY session: out-of-fold test probs, and for the
        # validation rows of each fold, that fold's own validation probs
        A, T = oa["test_proba"].copy(), ot["test_proba"].copy()

        # leakage screen on each teammate source alone
        for name, (kind, M, present) in team.items():
            if kind == "features":
                continue
            yy = y7 if kind == "emotion7" else y3
            acc = (M[present].argmax(1) == yy[present]).mean() if present.any() else 0
            flag = "   <- CHECK: this is suspiciously high (leakage or in-sample predictions?)" if acc >= 0.95 else ""
            print(f"   {name} alone: accuracy {acc:.3f} on its {present.sum()} sessions{flag}")

        for level, y, n_cls in (("emotion7", y7, 7), ("valence3", y3, 3)):
            def conv(P):                                               # 7-class probs -> 3 if needed
                return P if n_cls == 7 else np.stack([P[:, V_IDX == v].sum(1) for v in range(3)], 1)

            usable = {n: v for n, v in team.items()
                      if v[0] == "features" or v[0] == level or (level == "valence3" and v[0] == "emotion7")}
            combos = {"audio+text": []}
            for n in usable:
                combos[f"audio+text+{n}"] = [n]
            if len(usable) > 1:
                combos["all"] = list(usable)

            results = {}
            for cname, extra in combos.items():
                out = np.zeros((len(y), n_cls))
                for fi, va in enumerate(val_sets):
                    te = fold_of == fi
                    # validation rows use this fold's validation probabilities
                    va_a = np.array(oa["val"][fi]["proba"]); va_t = np.array(ot["val"][fi]["proba"])
                    blocks_va = [logp(conv(va_a)), logp(conv(va_t))]
                    blocks_te = [logp(conv(A[te])), logp(conv(T[te]))]
                    for n in extra:
                        kind, M, present = usable[n]
                        if kind == "features":
                            blocks_va.append(M[va]); blocks_te.append(M[te])
                        else:
                            Mc = conv(M) if (kind == "emotion7") else M
                            blocks_va += [logp(Mc[va]), present[va, None].astype(float)]
                            blocks_te += [logp(Mc[te]), present[te, None].astype(float)]
                    Xv, Xt = np.hstack(blocks_va), np.hstack(blocks_te)
                    mu, sd = Xv.mean(0), Xv.std(0); sd = np.where(sd > 1e-8, sd, 1)
                    if len(np.unique(y[va])) < 2:
                        out[te] = 1.0 / n_cls; continue
                    lr = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000, random_state=seed)
                    lr.fit((Xv - mu) / sd, y[va])
                    out[np.ix_(np.where(te)[0], lr.classes_)] = lr.predict_proba((Xt - mu) / sd)
                results[cname] = out

            base = results["audio+text"].argmax(1)
            print(f"\n== {level} | {a.protocol} | seed {seed}")
            for cname, P in results.items():
                pred = P.argmax(1)
                acc = float((pred == y).mean()); f1 = macro_f1(y, pred)
                txt = ""
                if cname != "audio+text":
                    t = paired_bootstrap_test(y, pred, base, spk, "macro_f1", seed=seed)
                    txt = f"vs audio+text {t['delta']:+.3f} p={t['p_value']:.3f}{' *' if t['significant_05'] else ''}"
                print(f"   {cname:<34} acc {acc:.3f}  macro F1 {f1:.3f}   {txt}")
                rows.append({"seed": seed, "level": level, "combination": cname, "acc": acc, "macro_f1": f1})
                if level == "emotion7":
                    exp = dirs["multimodal_result"] / a.protocol / f"seed{seed}" / cname.replace("+", "_")
                    write_report(exp, y, pred, P, spk, keys, title=f"multimodal {cname} | {a.protocol} | seed {seed}",
                                 config={"audio_run": str(arun), "text_run": str(trun), "sources": extra,
                                         "protocol": a.protocol, "seed": seed}, seed=seed, verbose=False)

    out = dirs["multimodal_result"] / f"summary_{a.protocol}.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\n[INFO] -> {out}")
    print("[INFO] Only trust gains from teammates' sources if their predictions are out-of-fold")
    print("       on the same speaker-disjoint splits (share preprocessing/splits.json with them).")


if __name__ == "__main__":
    main()
