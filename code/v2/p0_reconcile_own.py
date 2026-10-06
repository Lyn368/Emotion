"""
p0_reconcile_own.py - Link every video record we have to an audio file.

firebase_data does not contain one participant, but teammates' files may.
This script:
  1. collects every video record from ALL sources: every index in
     firebase_data (root + weekly) and every .json at the project root
     (metadata_records.json, predictions_per_video.json, ...),
  2. reads user, session, clip type, emotion and video id from each record,
     whatever the field names are (user_id / participant_id / uid, ...),
  3. links each record to an audio file in our_audios by, in order:
       a) the file name in local_path            (38d5...webm -> 38d5...wav)
       b) the video id itself                    (with or without dashes)
       c) the hash rule that produced the known audio names, if one exists
          (e.g. md5 of the storage URL), learned from firebase_data,
  4. writes reconciliation.csv (every record: user, video, audio, how linked)
     and extra_video_index.json with the records that link to an audio that
     firebase_data does not label. p1_index_datasets.py merges that file.

Raw inputs are never modified. Run BEFORE p1:
    python p0_reconcile_own.py
    python p1_index_datasets.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from config import AUDIO_EXTS, ensure_dirs, project_root, sources

UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I)
FIELDS = {                                   # canonical field -> key patterns (case-insensitive)
    "user": r"^(user_?id|user|participant_?id|participant|subject_?id|subject|uid)$",
    "session": r"^(session_?id|session)$",
    "type": r"^(video_?type|clip_?type|task|type)$",
    "emotion": r"^(reported_?emotion|self_?report(ed)?(_?emotion)?|emotion|label)$",
    "video_id": r"^(video_?id|vid)$",
    "url": r"(url|download)",
    "local_path": r"^(local_?path|file_?path|path|file_?name|file)$",
}
HASHES = {"md5": hashlib.md5, "sha1": hashlib.sha1, "sha256": hashlib.sha256}


# ---------------------------------------------------------------------------
# Record extraction
# ---------------------------------------------------------------------------
def all_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from all_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from all_dicts(v)


def normalise(d: dict) -> dict | None:
    """Map a raw dict onto canonical fields. None if it is not a video record."""
    rec = {}
    for canon, pat in FIELDS.items():
        for k, v in d.items():
            if canon not in rec and re.search(pat, str(k), re.I) and isinstance(v, (str, int, list)):
                rec[canon] = ",".join(map(str, v)) if isinstance(v, list) else str(v)
    has_video = any(rec.get(k) for k in ("video_id", "url", "local_path"))
    if not has_video or not rec.get("user"):
        return None
    m = UUID_RE.search(rec.get("video_id", "")) or UUID_RE.search(rec.get("url", ""))
    rec["video_uuid"] = m.group(0).lower() if m else ""
    lp = re.split(r"[/\\]", rec.get("local_path", ""))[-1]
    rec["local_stem"] = lp.rsplit(".", 1)[0] if "." in lp else lp
    return rec


def load_records(root: Path, firebase_dir: Path) -> list[dict]:
    files = sorted(firebase_dir.rglob("*.json")) + sorted(root.glob("*.json"))
    out = []
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        n = 0
        for d in all_dicts(data):
            r = normalise(d)
            if r:
                r["source"] = str(p.relative_to(root))
                r["in_firebase"] = firebase_dir in p.parents
                out.append(r)
                n += 1
        if n:
            print(f"  {n:6d} video records in {p.relative_to(root)}")
    return out


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------
def variants(v: str) -> list[str]:
    q = v.split("?")[0]
    return [v, q, v.replace("-webm", "").replace(".webm", ""), re.split(r"[/\\]|%2F", q)[-1]]


def learn_hash_rule(recs: list[dict], stems: set):
    """Find (field, variant index, algorithm) that reproduces known audio names from firebase records."""
    known = [r for r in recs if r["in_firebase"] and r["local_stem"] in stems]
    best = None
    for field in ("url", "video_id", "session"):
        vals = [(r[field], r["local_stem"]) for r in known if r.get(field)][:200]
        if not vals:
            continue
        for vi in range(4):
            for hname, fn in HASHES.items():
                ok = sum(1 for v, stem in vals if fn(variants(v)[vi].encode()).hexdigest() in (stem,))
                rate = ok / len(vals)
                if best is None or rate > best[0]:
                    best = (rate, field, vi, hname)
    return best if best and best[0] > 0.9 else None


def link(rec: dict, stems: set, rule) -> tuple[str, str]:
    if rec["local_stem"] in stems:
        return rec["local_stem"], "local_path"
    u = rec["video_uuid"]
    for cand in (u, u.replace("-", "")):
        if cand and cand in stems:
            return cand, "video_id"
    if rule:
        _, field, vi, hname = rule
        if rec.get(field):
            h = HASHES[hname](variants(rec[field])[vi].encode()).hexdigest()
            if h in stems:
                return h, f"{hname}({field})"
    return "", ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    root = project_root()
    src = sources(root)
    prep = ensure_dirs()["prep"]
    audio = {p.stem for p in src["own_audio"].rglob("*") if p.suffix.lower() in AUDIO_EXTS}
    print(f"[INFO] {len(audio)} audio files in {src['own_audio']}")

    recs = load_records(root, src["own_index_dir"])
    if not recs:
        raise SystemExit("[ERROR] No video records found in any .json file.")

    rule = learn_hash_rule(recs, audio)
    print(f"[INFO] hash rule for audio names: "
          + (f"{rule[3]}({rule[1]}), reproduces {rule[0]:.0%} of known names" if rule else "none found"))

    # one row per distinct video; firebase records take priority over teammates' ones
    by_video: dict[str, dict] = {}
    conflicts = []
    for r in sorted(recs, key=lambda r: not r["in_firebase"]):
        r["audio"], r["how"] = link(r, audio, rule)
        # the linked audio identifies a recording best: two sources may name the same
        # video differently (video_id in one, file name in the other)
        key = r["audio"] or r["video_uuid"] or r["local_stem"]
        if not key:
            continue
        if key not in by_video:
            by_video[key] = r
        else:
            old = by_video[key]
            if not old["audio"] and r["audio"]:
                old["audio"], old["how"] = r["audio"], r["how"]
            for f in ("user", "session", "type", "emotion"):
                if not old.get(f) and r.get(f):
                    old[f] = r[f]                      # fill gaps from other sources
                elif old.get(f) and r.get(f) and old[f] != r[f] and f in ("user", "emotion"):
                    conflicts.append((key, f, old[f], r[f], r["source"]))

    rows = list(by_video.values())
    fb_audio = {r["audio"] for r in rows if r["in_firebase"] and r["audio"]}
    linked = {r["audio"] for r in rows if r["audio"]}
    orphans = sorted(audio - linked)

    # per-user summary
    per_user = defaultdict(lambda: {"videos": 0, "with_audio": 0, "firebase": 0, "new": 0})
    for r in rows:
        u = per_user[r["user"]]
        u["videos"] += 1
        u["with_audio"] += bool(r["audio"])
        u["firebase"] += bool(r["in_firebase"])
        u["new"] += bool(r["audio"] and r["audio"] not in fb_audio)
    print(f"\n  {'user':<30}{'videos':>8}{'with audio':>12}{'in firebase':>13}{'new':>6}")
    for uid, u in sorted(per_user.items(), key=lambda t: -t[1]["videos"]):
        flag = "   <- not in firebase_data" if u["firebase"] == 0 else ""
        print(f"  {uid[:28]:<30}{u['videos']:>8}{u['with_audio']:>12}{u['firebase']:>13}{u['new']:>6}{flag}")

    no_audio = [r for r in rows if not r["audio"]]
    print(f"\n[INFO] distinct videos: {len(rows)} | linked to audio: {len(linked)} "
          f"| without audio: {len(no_audio)} | audio files still unlinked: {len(orphans)}")
    methods = defaultdict(int)
    for r in rows:
        if r["audio"]:
            methods[r["how"]] += 1
    print(f"       linked by: {dict(methods)}")
    if conflicts:
        print(f"[WARN] {len(conflicts)} field conflicts between sources (firebase kept). Examples:")
        for c in conflicts[:5]:
            print(f"       {c[0][:12]} {c[1]}: '{c[2]}' vs '{c[3]}' in {c[4]}")

    # outputs
    with open(prep / "reconciliation.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["user", "session", "type", "emotion", "video_uuid", "audio", "linked_by",
                    "in_firebase", "source"])
        for r in sorted(rows, key=lambda r: (r["user"], r.get("session", ""))):
            w.writerow([r["user"], r.get("session", ""), r.get("type", ""), r.get("emotion", ""),
                        r["video_uuid"], r["audio"], r["how"], r["in_firebase"], r["source"]])
        for o in orphans:
            w.writerow(["", "", "", "", "", o, "UNLINKED", "", ""])

    extra = []
    for r in rows:
        if r["audio"] and r["audio"] not in fb_audio:
            missing = [f for f in ("session", "type", "emotion") if not r.get(f)]
            extra.append({"user_id": r["user"], "session_id": r.get("session", ""),
                          "video_type": r.get("type", "").lower(),
                          "reported_emotion": r.get("emotion", ""),
                          "video_id": r["video_uuid"], "local_path": f"{r['audio']}.webm",
                          "source": r["source"], "missing_fields": missing})
    out = prep / "extra_video_index.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(extra, f, indent=2, ensure_ascii=False)
    incomplete = sum(1 for e in extra if e["missing_fields"])
    print(f"\n[INFO] {len(extra)} recordings recovered from other sources -> {out}")
    if incomplete:
        print(f"[WARN] {incomplete} of them lack session, clip type or emotion; p1 keeps them "
              f"but they cannot form a labelled session until those fields exist.")
    print(f"[INFO] full table -> {prep / 'reconciliation.csv'}")


if __name__ == "__main__":
    main()