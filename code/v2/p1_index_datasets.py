"""
p1_index_datasets.py — Inventory of every audio clip, from every source.
═══════════════════════════════════════════════════════════════════════════════
    python p1_index_datasets.py
    python p1_index_datasets.py --datasets own,iemocap
    python p1_index_datasets.py --root "C:\\Users\\2022g\\Downloads\\RAVDESS_IEMOCAP_CREMAD"

Output:  preprocessing/index.jsonl   (one line per clip)

Nothing is decoded or modified here; this only finds files and reads labels.
It is fast, and it tells you immediately if a folder is not where the pipeline
expects it, before anything slow runs.

Labels are stored RAW ("Neutral,Happiness", "fru", "calm", "ANG"). Turning them
into model targets is labels.py's job, done later by each branch.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from config import outputs, AUDIO_EXTS, ensure_dirs, project_root, sources
from labels import RAVDESS_CODES

RAVDESS_TEXT = {"01": "Kids are talking by the door.",
                "02": "Dogs are sitting by the door."}
CREMAD_TEXT = {
    "IEO": "It's eleven o'clock.", "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.", "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.", "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.", "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.", "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.", "WSI": "We'll stop in a couple of minutes.",
}


def audio_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXTS)


# ─────────────────────────────────────────────────────────────────────────────
def _entry_key(e: dict) -> str:
    """Audio file stem an entry points to: last part of local_path without extension."""
    lp = re.split(r"[/\\]", e.get("local_path", "") or "")[-1]
    if lp:
        return lp.rsplit(".", 1)[0] if "." in lp else lp
    return (e.get("video_id") or "").replace("-webm", "").replace(".webm", "")


def find_video_index(folder: Path, override: str | None = None, merge: bool = True):
    """
    Load the in-house label index.

    firebase_data holds one complete index at its root plus weekly exports in
    sub-folders. The weekly files can contain recordings the root file lacks,
    so by default ALL index files are merged:
      - the largest file is the primary source,
      - entries from the other files are added only if their audio is new,
      - the same audio with a different user or label is a CONFLICT: the
        primary file wins and the conflict is reported.
    --video_index <file> uses one file only; --no_merge uses the largest only.
    """
    if override:
        p = Path(override)
        return p, json.loads(p.read_text(encoding="utf-8"))
    if not folder.exists():
        print(f"  [own] folder not found: {folder}")
        return None, []

    found = []
    for p in folder.rglob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list) and data and isinstance(data[0], dict) \
                and ("local_path" in data[0] or "video_id" in data[0]):
            found.append((len(data), p, data))
    if not found:
        print(f"  [own] no video index (.json list with local_path/video_id) under {folder}")
        return None, []
    found.sort(key=lambda t: (-t[0], len(str(t[1]))))
    primary = found[0][1]
    print(f"  [own] {len(found)} index files found; primary: {primary}")
    if not merge or len(found) == 1:
        return primary, found[0][2]

    merged: dict[str, dict] = {}
    added_from: dict[str, int] = {}
    conflicts = []
    for rank, (n, p, data) in enumerate(found):
        for e in data:
            if not isinstance(e, dict):
                continue
            k = _entry_key(e)
            if not k:
                continue
            if k not in merged:
                merged[k] = e
                if rank > 0:
                    added_from[str(p)] = added_from.get(str(p), 0) + 1
            else:
                old = merged[k]
                if (old.get("user_id"), old.get("reported_emotion")) != \
                        (e.get("user_id"), e.get("reported_emotion")):
                    conflicts.append((k, old.get("reported_emotion"), e.get("reported_emotion"), p.parent.name))

    n_extra = sum(added_from.values())
    print(f"  [own] merged: {found[0][0]} from primary + {n_extra} only in weekly files "
          f"= {len(merged)} unique recordings")
    for p, n in sorted(added_from.items(), key=lambda t: -t[1]):
        print(f"        +{n:4d} from {Path(p).parent.name}")
    if conflicts:
        print(f"  [own] [WARN] {len(conflicts)} recording(s) appear with a different user or label "
              f"in a weekly file; the primary file was kept. Examples:")
        for k, a, b, where in conflicts[:5]:
            print(f"        {k}: primary '{a}' vs '{b}' in {where}")
    return primary, list(merged.values())


def index_own(src: dict) -> list[dict]:
    idx_path, entries = find_video_index(src["own_index_dir"], src.get("video_index"),
                                         merge=not src.get("no_merge", False))
    if idx_path is None:
        return []

    # recordings recovered from teammates' files by p0_reconcile_own.py
    extra_path = outputs()["prep"] / "extra_video_index.json"
    if extra_path.exists() and not src.get("video_index"):
        extra = json.loads(extra_path.read_text(encoding="utf-8"))
        have = {_entry_key(e) for e in entries}
        add = [e for e in extra if _entry_key(e) not in have]
        entries = list(entries) + add
        print(f"  [own] +{len(add)} recordings from {extra_path.name} (recovered by p0_reconcile_own.py)")

    by_stem: dict[str, Path] = {}
    for p in audio_files(src["own_audio"]):
        # keep the first file per stem, preferring .wav over compressed formats
        if p.stem not in by_stem or p.suffix.lower() == ".wav":
            by_stem[p.stem] = p

    out, missing = [], 0
    for e in entries:
        vid = e.get("video_id", "")
        # The audio file is the last part of local_path with .wav instead of .webm:
        #   .../videos/38d564911ee70183a2acf08be5f29232.webm -> 38d564911ee70183a2acf08be5f29232.wav
        # Split on both / and \ so a macOS path works on Windows and vice versa.
        lp_name = re.split(r"[/\\]", e.get("local_path", "") or "")[-1]
        lp_stem = lp_name.rsplit(".", 1)[0] if "." in lp_name else lp_name
        cands = [lp_stem, vid.replace("-webm", "").replace(".webm", ""), vid,
                 vid.replace("-", "_")]
        path = next((by_stem[c] for c in cands if c and c in by_stem), None)
        if path is None:
            missing += 1
            continue
        out.append({
            "clip_id": f"own__{path.stem}", "dataset": "own",
            "audio_path": str(path), "speaker_id": e.get("user_id", ""),
            "session_id": e.get("session_id", ""),
            "clip_type": (e.get("video_type") or "").lower(),
            "label_raw": e.get("reported_emotion", ""),
            "transcript_gold": None, "timestamp": e.get("timestamp"),
        })
    print(f"  [own] {len(out)} clips matched | {missing} index entries without audio "
          f"| index: {idx_path}")

    # Diagnostics: make a partial match impossible to miss
    idx_users = {e.get("user_id") for e in entries}
    got_users = {r["speaker_id"] for r in out}
    used = {Path(r["audio_path"]).stem for r in out}
    print(f"  [own] audio files in {src['own_audio'].name}: {len(by_stem)} | "
          f"index entries: {len(entries)} | users in index: {len(idx_users)} | "
          f"users matched: {len(got_users)}")
    if missing:
        shown = 0
        for e in entries:
            lp = re.split(r"[/\\]", e.get("local_path", "") or "")[-1]
            stem = lp.rsplit(".", 1)[0] if lp else ""
            if stem and stem not in by_stem and shown < 5:
                print(f"        no audio for: {stem}.wav   (user {str(e.get('user_id'))[:10]})")
                shown += 1
    unused = [s for s in by_stem if s not in used]
    if unused:
        print(f"  [own] {len(unused)} audio files not referenced by the index, e.g. "
              f"{', '.join(unused[:3])}")
    if len(got_users) < len(idx_users):
        print(f"  [own] [WARN] only {len(got_users)} of {len(idx_users)} users matched. "
              f"Check the lines above.")
    return out


# ─────────────────────────────────────────────────────────────────────────────
_EVAL_RE = re.compile(r"^\[.*?\]\s+(\S+)\s+(\w+)\s+\[")
_TRANS_RE = re.compile(r"^(\S+)\s+\[[\d.\-]+\]:\s*(.*)$")


# IEMOCAP utterance ids encode everything: Ses01F_impro01_F000
#   session 01, dialog "impro01", speaker = the LAST F/M (who is talking)
_IEMO_ID = re.compile(r"(Ses(\d{2})[FM]_([^/\\\s]+?)_([FM])\d{3})", re.IGNORECASE)  # no path separators inside an id
_ID_COLS = ("wav_file", "file", "filename", "file_name", "path", "audio", "audio_path",
            "utterance", "utt", "utterance_id", "id", "name")
_LABEL_COLS = ("emotion", "label", "emo", "category", "class", "major_emotion")


def _labels_from_table(path: Path) -> dict[str, str]:
    """Read a csv/tsv/json/jsonl label table; auto-detect id and emotion columns."""
    import csv
    rows = []
    try:
        if path.suffix.lower() in (".json", ".jsonl"):
            txt = path.read_text(encoding="utf-8", errors="ignore").strip()
            if path.suffix.lower() == ".jsonl" or not txt.startswith("["):
                rows = [json.loads(l) for l in txt.splitlines() if l.strip().startswith("{")]
            else:
                rows = json.loads(txt)
        else:
            with open(path, encoding="utf-8-sig", errors="ignore", newline="") as f:
                sample = f.read(4096); f.seek(0)
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
                rows = list(csv.DictReader(f, dialect=dialect))
    except Exception:
        return {}
    if not rows or not isinstance(rows[0], dict):
        return {}
    cols = {c.lower().strip(): c for c in rows[0].keys()}
    lab_col = next((cols[c] for c in _LABEL_COLS if c in cols), None)
    if lab_col is None:
        return {}
    id_cols = [cols[c] for c in _ID_COLS if c in cols] or list(rows[0].keys())
    out = {}
    for r in rows:
        for c in id_cols:                       # first column that contains an utterance id wins
            m = _IEMO_ID.search(str(r.get(c, "")))
            if m:
                out[m.group(1)] = str(r.get(lab_col, "")).strip()
                break
    return out


def index_iemocap(src: dict) -> list[dict]:
    """
    Works with ANY layout: the official Session*/ tree, or all wav files in one
    flat folder (as in most re-distributed copies). Metadata comes from the file
    name; labels are searched in this order:
      1. official EmoEvaluation text files, anywhere under the folder
      2. a csv/tsv/json label table, anywhere under the folder (or --iemocap_labels)
      3. an emotion code at the end of the file name (e.g. ..._F000_neu.wav)
    """
    root = src["iemocap"]
    if not root.exists():
        print(f"  [iemocap] folder not found: {root}")
        return []

    audio = [p for p in audio_files(root)]
    utts = {}
    for p in audio:
        m = _IEMO_ID.search(p.stem)
        if m:
            utts.setdefault(m.group(1), p)
    unmatched = len(audio) - len(utts)

    labels, gold, sources = {}, {}, []
    # 1. official annotation files
    for f in root.rglob("*.txt"):
        n0 = len(labels)
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = _EVAL_RE.match(line)
            if m:
                labels.setdefault(m.group(1), m.group(2))
            t = _TRANS_RE.match(line.strip())
            if t:
                gold.setdefault(t.group(1), t.group(2).strip())
        if len(labels) > n0 and "EmoEvaluation files" not in sources:
            sources.append("EmoEvaluation files")
    # 2. label tables
    tables = [src["iemocap_labels"]] if src.get("iemocap_labels") else \
        [p for ext in ("*.csv", "*.tsv", "*.json", "*.jsonl") for p in root.rglob(ext)]
    for t in tables:
        found = _labels_from_table(Path(t))
        added = sum(1 for k in found if k not in labels)
        for k, v in found.items():
            labels.setdefault(k, v)
        if added:
            sources.append(f"table {Path(t).name} ({added})")
    # 3. emotion code in the file name
    n_name = 0
    for utt, p in utts.items():
        if utt not in labels:
            tail = p.stem[len(utt):].strip("_- ").lower()
            if tail[:3] in ("neu", "hap", "exc", "ang", "sad", "fru", "fea", "sur", "dis", "oth", "xxx"):
                labels[utt] = tail[:3]; n_name += 1
    if n_name:
        sources.append(f"file names ({n_name})")

    out, n_unlab = [], 0
    for utt, p in sorted(utts.items()):
        if utt not in labels:
            n_unlab += 1
            continue
        m = _IEMO_ID.search(utt)
        sess_no, dialog, spk = m.group(2), m.group(3), m.group(4).upper()
        out.append({
            "clip_id": f"iemocap__{utt}", "dataset": "iemocap", "audio_path": str(p),
            "speaker_id": f"Session{int(sess_no)}_{spk}",          # 10 speakers in total
            "session_id": utt.rsplit("_", 1)[0],                   # the dialog, e.g. Ses01F_impro01
            "clip_type": "scripted" if "script" in dialog.lower() else "improvised",
            "label_raw": labels[utt], "transcript_gold": gold.get(utt),
        })

    print(f"  [iemocap] {len(audio)} audio files | {len(utts)} with an IEMOCAP id "
          f"| {len(out)} labelled | {n_unlab} without label")
    print(f"  [iemocap] label source(s): {', '.join(sources) if sources else 'NONE FOUND'}")
    if unmatched:
        print(f"  [iemocap] [WARN] {unmatched} files have no IEMOCAP id in their name and were skipped")
    if utts and not out and labels:
        print("  [iemocap] [ERROR] labels were found but none match an audio file name.")
        print(f"            example label id: {next(iter(labels))}")
        print(f"            example audio id: {next(iter(utts))}")
    elif utts and not out:
        print("  [iemocap] [ERROR] audio found but no labels. IEMOCAP labels are not stored in the")
        print("            audio. Put the label file (EmoEvaluation folder, or a csv with a file")
        print("            column and an emotion column) inside the iemocap folder, or pass it")
        print("            with --iemocap_labels path\\to\\labels.csv")
    if len(gold) == 0 and out:
        print("  [iemocap] no official transcripts found: text will use Whisper transcripts only")
    return out


def index_ravdess(src: dict) -> list[dict]:
    out = []
    for p in audio_files(src["ravdess"]):
        parts = p.stem.split("-")
        if len(parts) != 7:
            continue
        modality, vocal, emo, inten, stmt, rep, actor = parts
        if modality != "03" or vocal != "01" or emo not in RAVDESS_CODES:
            continue                                  # audio-only speech files
        out.append({
            "clip_id": f"ravdess__{p.stem}", "dataset": "ravdess",
            "audio_path": str(p), "speaker_id": f"actor{int(actor):02d}",
            "session_id": f"actor{int(actor):02d}", "clip_type": "acted",
            "label_raw": RAVDESS_CODES[emo], "transcript_gold": RAVDESS_TEXT.get(stmt),
            "sentence_id": stmt, "intensity": inten,
        })
    print(f"  [ravdess] {len(out)} speech clips")
    return out


def index_cremad(src: dict) -> list[dict]:
    out = []
    pat = re.compile(r"^(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})$")
    for p in audio_files(src["cremad"]):
        m = pat.match(p.stem)
        if not m:
            continue
        actor, sent, emo, level = m.groups()
        out.append({
            "clip_id": f"cremad__{p.stem}", "dataset": "cremad",
            "audio_path": str(p), "speaker_id": f"actor{actor}",
            "session_id": f"actor{actor}", "clip_type": "acted",
            "label_raw": emo, "transcript_gold": CREMAD_TEXT.get(sent),
            "sentence_id": sent, "intensity": level,
        })
    print(f"  [cremad] {len(out)} clips")
    return out


INDEXERS = {"own": index_own, "iemocap": index_iemocap,
            "ravdess": index_ravdess, "cremad": index_cremad}


def main():
    ap = argparse.ArgumentParser(description="Index all audio sources")
    ap.add_argument("--root", default=None)
    ap.add_argument("--datasets", default="own,iemocap,ravdess,cremad")
    ap.add_argument("--video_index", default=None,
                    help="Use only this index file (default: merge every index in firebase_data)")
    ap.add_argument("--no_merge", action="store_true",
                    help="Use only the largest index file instead of merging all of them")
    ap.add_argument("--iemocap_labels", default=None,
                    help="Label table for IEMOCAP if it is not inside the iemocap folder")
    args = ap.parse_args()

    root = project_root(args.root)
    src = sources(root)
    src["iemocap_labels"] = args.iemocap_labels
    src["video_index"] = args.video_index
    src["no_merge"] = args.no_merge
    out_dir = ensure_dirs()["prep"]
    print(f"[INFO] Project root: {root}")

    records = []
    for name in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        records += INDEXERS[name](src)
    if not records:
        raise SystemExit("[ERROR] Nothing indexed. Check the folder names in config.py.")

    dup = [k for k, v in Counter(r["clip_id"] for r in records).items() if v > 1]
    if dup:
        raise SystemExit(f"[ERROR] Duplicate clip ids, e.g. {dup[:3]}")

    path = out_dir / "index.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n[INFO] {len(records)} clips -> {path}")
    for ds, n in Counter(r["dataset"] for r in records).items():
        spk = len({r["speaker_id"] for r in records if r["dataset"] == ds})
        print(f"       {ds:<8} {n:6d} clips  {spk:4d} speakers")


if __name__ == "__main__":
    main()