"""
p6_public_sessions.py - Turn RAVDESS, CREMA-D and IEMOCAP into "sessions" like the in-house data.

In-house sessions pair a freestyle clip (the target) with a passage clip read in a
neutral voice by the same person (the reference). Public corpora have no passage,
so each utterance is paired with a NEUTRAL utterance by the same speaker:

  RAVDESS / CREMA-D   same speaker AND same sentence if possible (the words are then
                      identical, so the difference is purely how it was said), else
                      same speaker.
  IEMOCAP             same speaker AND same dialogue if possible (same recording
                      session and microphone), else same speaker.

The reference is chosen WITHOUT looking at the target's label (a hash of the target's
clip id picks among the candidates) and is never the target itself. Neutral targets
get another neutral clip as reference.

    python p6_public_sessions.py                              # all three
    python p6_public_sessions.py --datasets ravdess --seeds 42,43,44,45,46
    python p6_public_sessions.py --iemocap_classes all        # keep fear/disgust/surprise too

Class sets (defaults follow the usual benchmarks):
  RAVDESS  7 classes; 'calm' is DROPPED (it has no slot; --ravdess_calm neutral merges it)
  CREMA-D  6 classes (no surprise in the corpus)
  IEMOCAP  4 classes: angry, happy (+excited), neutral, sad (--iemocap_classes all = 7)

Folds are by speaker, like the in-house data. For IEMOCAP the group is the recording
SESSION (Session1..5, two speakers each), so 'louo' is leave-one-session-out, the
standard IEMOCAP protocol, and no dialogue is ever split across train and test.

Output: preprocessing/public/<dataset>/splits.json   (same schema as preprocessing/splits.json)
Then run the branches with --eval <dataset>.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from config import V2_DIR, ensure_dirs
from cv import METHODS, inner_split, outer_splits
from labels import EMOTIONS, resolve

IEMOCAP_4 = {"angry", "happy", "neutral", "sad"}


def pick(cands: list[str], target_id: str) -> str:
    """Deterministic, label-independent choice among candidate references."""
    cands = sorted(cands)
    return cands[int(hashlib.md5(target_id.encode("utf-8")).hexdigest(), 16) % len(cands)]


def build(dataset: str, recs: list[dict], a) -> tuple[list[dict], Counter, Counter]:
    excluded, match = Counter(), Counter()

    usable = []
    for r in recs:
        if not r.get("encoder_path") or not r.get("measure_path"):
            excluded["not preprocessed (run p2)"] += 1; continue
        raw = r.get("label_raw")
        if dataset == "ravdess" and raw == "calm" and a.ravdess_calm == "drop":
            excluded["RAVDESS calm (dropped)"] += 1; continue
        lab = resolve(dataset, raw)
        if lab["y7"] is None:
            excluded[f"label '{raw}' has no class"] += 1; continue
        if dataset == "iemocap" and a.iemocap_classes == "4" and lab["dominant"] not in IEMOCAP_4:
            excluded[f"IEMOCAP {lab['dominant']} (4-class setup)"] += 1; continue
        usable.append((r, lab))

    # neutral reference pools; RAVDESS 'calm' never serves as a neutral reference
    ctx = "session_id" if dataset == "iemocap" else "sentence_id"
    by_spk, by_ctx = defaultdict(list), defaultdict(list)
    for r, lab in usable:
        if lab["dominant"] == "neutral" and r.get("label_raw") != "calm":
            by_spk[r["speaker_id"]].append(r["clip_id"])
            by_ctx[(r["speaker_id"], r.get(ctx))].append(r["clip_id"])

    out = []
    for r, lab in usable:
        cid, spk = r["clip_id"], r["speaker_id"]
        same_ctx = [c for c in by_ctx.get((spk, r.get(ctx)), []) if c != cid]
        same_spk = [c for c in by_spk.get(spk, []) if c != cid]
        if same_ctx:
            ref, how = pick(same_ctx, cid), f"same speaker + same {'dialogue' if ctx == 'session_id' else 'sentence'}"
        elif same_spk:
            ref, how = pick(same_spk, cid), "same speaker"
        else:
            ref, how = None, "no neutral clip for this speaker"
        match[how] += 1
        group = spk.split("_")[0] if dataset == "iemocap" else spk          # IEMOCAP: Session1..5
        out.append({"key": f"{spk}|{cid}", "speaker_id": spk, "group": group,
                    "session_id": cid, "freestyle_clip": cid, "passage_clip": ref,
                    "reference_match": how, "label_raw": r.get("label_raw"),
                    "dominant": lab["dominant"], "y7": lab["y7"], "y3": lab["y3"]})
    return out, excluded, match


def make_splits(ev: list[dict], protocols, n_folds, seeds):
    y = np.array([s["y7"] for s in ev])
    g = np.array([s["group"] for s in ev])
    keys = np.array([s["key"] for s in ev])
    group_of_clip = {s["freestyle_clip"]: s["group"] for s in ev}
    for s in ev:                                       # reference always comes from the same group
        if s["passage_clip"]:
            assert group_of_clip.get(s["passage_clip"], s["group"]) == s["group"]
    biggest = max(Counter(y.tolist()).values())
    if biggest < n_folds:
        raise SystemExit(f"[ERROR] largest class has {biggest} items < --folds {n_folds}")
    splits = {}
    for seed in seeds:
        splits[str(seed)] = {}
        for method in protocols:
            folds = []
            for tr, te, name in outer_splits(y, g, method, n_folds, seed):
                itr, ival = inner_split(y[tr], g[tr], method, seed=seed)
                trn, val = tr[itr], tr[ival]
                if method != "kfold":
                    assert not set(g[te]) & set(g[tr]), "group in train and test"
                    assert not set(g[val]) & set(g[trn]), "group in inner train and val"
                folds.append({"name": name.replace("user=", "group="), "train": keys[trn].tolist(),
                              "val": keys[val].tolist(), "test": keys[te].tolist()})
            cov = Counter(k for f in folds for k in f["test"])
            assert len(cov) == len(keys) and max(cov.values()) == 1, f"{method}: each item tested once"
            splits[str(seed)][method] = folds
    return splits


def main():
    ap = argparse.ArgumentParser(description="Public corpora -> sessions + folds")
    ap.add_argument("--datasets", default="ravdess,cremad,iemocap")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--protocols", default=",".join(METHODS))
    ap.add_argument("--ravdess_calm", default="drop", choices=["drop", "neutral"])
    ap.add_argument("--iemocap_classes", default="4", choices=["4", "all"])
    a = ap.parse_args()

    prep = ensure_dirs()["prep"]
    seeds = [int(s) for s in a.seeds.split(",")]
    protocols = [p for p in a.protocols.split(",") if p]
    for ds in [d.strip() for d in a.datasets.split(",") if d.strip()]:
        path = prep / f"{ds}.json"
        if not path.exists():
            print(f"\n== {ds}: {path.name} missing. Run p1 -> p2 -> p4 with this dataset first."); continue
        data = json.loads(path.read_text(encoding="utf-8"))
        recs = data["records"] if isinstance(data, dict) else data
        ev, excluded, match = build(ds, recs, a)
        if not ev:
            print(f"\n== {ds}: no usable items. Excluded: {dict(excluded)}"); continue
        n_groups = len({s["group"] for s in ev})
        if ds == "cremad" and "louo" in protocols:
            print(f"[NOTE] CREMA-D louo = {n_groups} folds; grouped (5 folds) is usually enough.")
        splits = make_splits(ev, protocols, a.folds, seeds)
        counts = Counter(EMOTIONS[s["y7"]] for s in ev)
        out = {"dataset": ds, "evaluation_set": ev, "excluded": dict(excluded),
               "n_sessions": len(ev), "n_speakers": len({s["speaker_id"] for s in ev}),
               "group_by": "recording session" if ds == "iemocap" else "speaker", "n_groups": n_groups,
               "class_counts": dict(sorted(counts.items())), "reference_match": dict(match),
               "class_policy": {"ravdess_calm": a.ravdess_calm, "iemocap_classes": a.iemocap_classes},
               "n_folds": a.folds, "splits": splits}
        d = prep / "public" / ds
        d.mkdir(parents=True, exist_ok=True)
        (d / "splits.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\n== {ds}: {len(ev)} items | {out['n_speakers']} speakers | {n_groups} {out['group_by']} groups")
        print(f"   classes:   {dict(sorted(counts.items()))}")
        print(f"   reference: {dict(match)}")
        if excluded:
            print(f"   excluded:  {dict(excluded)}")
        for seed, by_m in splits.items():
            print(f"   seed {seed}: " + ", ".join(f"{m}={len(f)} folds" for m, f in by_m.items()))
        print(f"   -> {d / 'splits.json'}")
        print(f"   next: python a1_acoustic_features.py --dataset {ds}")
        print(f"         python a2_encoder_embeddings.py --model wavlm_base_plus --dataset {ds}")
        print(f"         python a3_audio_frozen.py --eval {ds} --protocols grouped")


if __name__ == "__main__":
    main()
