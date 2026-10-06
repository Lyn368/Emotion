"""
p5_make_splits.py — The folds every branch must use.
═══════════════════════════════════════════════════════════════════════════════
    python p5_make_splits.py
    python p5_make_splits.py --seeds 42,43,44,45,46     # for multi-seed runs

Output:  preprocessing/splits.json

WHY THIS EXISTS
─────────────────────────────────────────────────────────────────────────────
Fusion combines the audio and text predictions of the SAME session. That is
only valid if both branches predicted that session from models that never saw
it — i.e. with identical folds. When each branch built its own folds, the
audio branch had 393 sessions and the text branch 385, so even with the same
seed the folds came out different. From now on the evaluation set and the
folds are defined once, here, and every branch reads them.

EVALUATION SET
─────────────────────────────────────────────────────────────────────────────
A session is included only if it has: a freestyle clip, a passage clip, a
non-empty freestyle transcript, and a self-report that maps to one of the seven
classes. Excluded sessions are listed with the reason, so the number can be
reported in the paper.

Each fold stores train / val / test session keys ("speaker|session"). val is
the inner split carved out of train (speaker-disjoint for grouped and louo),
used for early stopping and for fitting fusion weights. test is never used for
any decision.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from config import ensure_dirs
from cv import METHODS, inner_split, outer_splits
from labels import EMOTIONS, resolve


def main():
    ap = argparse.ArgumentParser(description="Shared evaluation set and folds")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="42")
    args = ap.parse_args()

    prep = ensure_dirs()["prep"]
    path = prep / "own_dataset.json"
    if not path.exists():
        raise SystemExit("[ERROR] Run p4_build_datasets.py first.")
    clips = json.loads(path.read_text(encoding="utf-8"))

    sess: dict[tuple, dict] = defaultdict(dict)
    for c in clips:
        sess[(c["speaker_id"], c["session_id"])][c["clip_type"]] = c

    included, excluded = [], Counter()
    for (spk, sid), d in sorted(sess.items()):
        f, p = d.get("freestyle"), d.get("passage")
        if f is None:
            excluded["no freestyle clip"] += 1; continue
        if p is None:
            excluded["no passage clip"] += 1; continue
        if not (f.get("transcript") or "").strip():
            excluded["empty freestyle transcript"] += 1; continue
        lab = resolve("own", f["label_raw"])
        if lab["y7"] is None:
            excluded["label not in the 7 classes"] += 1; continue
        included.append({"key": f"{spk}|{sid}", "speaker_id": spk, "session_id": sid,
                         "freestyle_clip": f["clip_id"], "passage_clip": p["clip_id"],
                         "label_raw": f["label_raw"], "dominant": lab["dominant"],
                         "y7": lab["y7"], "y3": lab["y3"]})

    if not included:
        raise SystemExit("[ERROR] No session passes the inclusion criteria.")

    y = np.array([s["y7"] for s in included])
    g = np.array([s["speaker_id"] for s in included])
    keys = np.array([s["key"] for s in included])

    biggest = max(Counter(y.tolist()).values())
    if biggest < args.folds:
        raise SystemExit(f"[ERROR] The largest class has {biggest} sessions, fewer than "
                         f"--folds {args.folds}. Use --folds {biggest} or fewer.")

    splits = {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        splits[str(seed)] = {}
        for method in METHODS:
            folds = []
            for tr, te, name in outer_splits(y, g, method, args.folds, seed):
                itr, ival = inner_split(y[tr], g[tr], method, seed=seed)
                trn, val = tr[itr], tr[ival]
                if method != "kfold":        # leakage guard, checked every time
                    assert not set(g[te]) & set(g[tr]), "speaker in train and test"
                    assert not set(g[val]) & set(g[trn]), "speaker in inner train and val"
                folds.append({"name": name, "train": keys[trn].tolist(),
                              "val": keys[val].tolist(), "test": keys[te].tolist()})
            covered = Counter(k for f in folds for k in f["test"])
            assert len(covered) == len(keys) and max(covered.values()) == 1, \
                f"{method}: every session must be tested exactly once"
            splits[str(seed)][method] = folds

    out = {"evaluation_set": included,
           "excluded": dict(excluded),
           "n_sessions": len(included),
           "n_speakers": int(len(set(g))),
           "class_counts": {EMOTIONS[k]: int(v) for k, v in sorted(Counter(y.tolist()).items())},
           "n_folds": args.folds,
           "splits": splits}
    (prep / "splits.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"[INFO] Evaluation set: {len(included)} sessions, {len(set(g))} speakers")
    if excluded:
        print(f"[INFO] Excluded: {dict(excluded)}")
    print(f"[INFO] Classes: {out['class_counts']}")
    for seed, by_m in splits.items():
        print(f"[INFO] seed {seed}: " + ", ".join(f"{m}={len(f)} folds" for m, f in by_m.items()))
    print(f"[INFO] -> {prep / 'splits.json'}")


if __name__ == "__main__":
    main()
