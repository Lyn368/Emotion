"""
check_missing_labels.py - Find the labels of audio files that no video_index.json mentions.

Some recordings (one participant) are missing from firebase_data, but teammates
may have them in their own files. This script:
  1. lists the audio files in our_audios that p1 could not match,
  2. searches every .json at the project root for those file names,
  3. prints how many were found, where, and one full example record,
so the right file and format can be wired into p1.

Run after p1_index_datasets.py:
    python check_missing_labels.py
"""
from __future__ import annotations

import json
from pathlib import Path

from config import AUDIO_EXTS, outputs, project_root, sources


def find_records(obj, stems: set, hits: list, path="$"):
    """Walk any JSON structure; collect every dict whose values mention a missing stem."""
    if isinstance(obj, dict):
        text = json.dumps(obj, ensure_ascii=False)
        found = [s for s in stems if s in text]
        # only keep the smallest dict that mentions the stem (the record, not the whole file)
        child_hit = False
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and any(s in json.dumps(v, ensure_ascii=False) for s in found):
                child_hit = True
                find_records(v, stems, hits, f"{path}.{k}")
        if found and not child_hit:
            hits.append((path, found, obj))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            find_records(v, stems, hits, f"{path}[{i}]")


def main():
    root = project_root()
    src = sources(root)
    idx = outputs()["prep"] / "index.jsonl"
    if not idx.exists():
        raise SystemExit("[ERROR] Run p1_index_datasets.py first.")

    matched = {Path(json.loads(l)["audio_path"]).stem
               for l in idx.read_text(encoding="utf-8").splitlines()
               if l.strip() and json.loads(l)["dataset"] == "own"}
    audio = {p.stem for p in src["own_audio"].rglob("*") if p.suffix.lower() in AUDIO_EXTS}
    missing = audio - matched
    print(f"[INFO] {len(audio)} audio files | {len(matched)} labelled by firebase_data "
          f"| {len(missing)} without label")
    if not missing:
        return

    candidates = sorted(p for p in root.glob("*.json"))
    for p in candidates:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"\n  {p.name}: could not read ({exc})")
            continue
        hits: list = []
        find_records(data, missing, hits)
        covered = {s for _, found, _ in hits for s in found}
        print(f"\n  {p.name}: {len(covered)} of {len(missing)} unlabelled files found")
        if hits:
            path, found, rec = hits[0]
            print(f"    example record (at {path}):")
            text = json.dumps(rec, indent=2, ensure_ascii=False)
            print("    " + "\n    ".join(text.splitlines()[:30]))
            keys = sorted({k for _, _, r in hits if isinstance(r, dict) for k in r})
            print(f"    fields seen in matching records: {', '.join(keys)}")


if __name__ == "__main__":
    main()
