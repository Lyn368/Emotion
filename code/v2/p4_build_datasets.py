"""
p4_build_datasets.py — One JSON per dataset, the single source for both branches.
═══════════════════════════════════════════════════════════════════════════════
    python p4_build_datasets.py
    python p4_build_datasets.py --no_goemotions

Output (preprocessing/):
    own_dataset.json   iemocap.json   ravdess.json   cremad.json   goemotions.json

Each audio record:
    clip_id, audio_file, dataset, speaker_id, session_id, clip_type,
    transcript (Whisper), transcript_gold, asr_avg_logprob, asr_no_speech_prob,
    label_raw, label_source, duration_raw_s, duration_speech_s,
    speech_segments, measured_lufs, snr_db,
    measure_path, encoder_path, original_path        (paths RELATIVE, see below)

DESIGN RULES
─────────────────────────────────────────────────────────────────────────────
- Labels stay RAW. labels.py turns them into targets inside each branch.
- label_source records what the label means, so self-report and annotator
  labels can never be mixed in one evaluation by accident:
      own        self_report
      iemocap    annotator_majority
      ravdess    acted_intended    (the emotion the actor was asked to portray)
      cremad     acted_intended
      goemotions annotator
- No embeddings in the JSON. They depend on the model, and with fine-tuning
  they change every fold. Frozen embeddings are cached separately as .npy.
- Paths are stored relative (preprocessed audio to code\\v2, originals to the
  project root) so moving the project folder does not break anything.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from config import V2_DIR, ensure_dirs, project_root, sources
from labels import resolve

LABEL_SOURCE = {"own": "self_report", "iemocap": "annotator_majority",
                "ravdess": "acted_intended", "cremad": "acted_intended"}
OUT_NAME = {"own": "own_dataset.json", "iemocap": "iemocap.json",
            "ravdess": "ravdess.json", "cremad": "cremad.json"}


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def rel(path: str | None, base: Path) -> str | None:
    if not path:
        return None
    try:
        return str(Path(path).resolve().relative_to(base.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def load_goemotions(local: Path) -> list[dict]:
    rows = []
    if (local / "emotions.txt").exists():
        names = (local / "emotions.txt").read_text(encoding="utf-8").split()
        for split in ("train", "dev", "test"):
            f = local / f"{split}.tsv"
            if not f.exists():
                continue
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines()):
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                ids = [int(x) for x in parts[1].split(",") if x.strip().isdigit()]
                rows.append({"text_id": f"goemo_{split}_{i}", "dataset": "goemotions",
                             "split": split, "text": parts[0],
                             "label_raw": [names[j] for j in ids], "label_source": "annotator"})
        print(f"  [goemotions] {len(rows)} texts from local copy {local}")
        return rows
    try:
        from datasets import load_dataset
        ds = load_dataset("google-research-datasets/go_emotions", "simplified")
    except Exception as exc:
        print(f"  [goemotions] skipped: no local copy and HF download failed ({exc})")
        return []
    names = ds["train"].features["labels"].feature.names
    for split in ds:
        for i, ex in enumerate(ds[split]):
            rows.append({"text_id": f"goemo_{split}_{i}", "dataset": "goemotions",
                         "split": split, "text": ex["text"],
                         "label_raw": [names[j] for j in ex["labels"]],
                         "label_source": "annotator"})
    print(f"  [goemotions] {len(rows)} texts from Hugging Face")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Build per-dataset JSON files")
    ap.add_argument("--root", default=None)
    ap.add_argument("--no_goemotions", action="store_true")
    args = ap.parse_args()

    root = project_root(args.root)
    dirs = ensure_dirs()
    prep = dirs["prep"]

    index = read_jsonl(prep / "index.jsonl")
    man = {r["clip_id"]: r for r in read_jsonl(prep / "audio_manifest.jsonl")}
    trs = {r["clip_id"]: r for r in read_jsonl(prep / "transcripts.jsonl")}
    if not index:
        raise SystemExit("[ERROR] Run p1_index_datasets.py first.")

    by_ds: dict[str, list] = {}
    no_audio = Counter()
    for r in index:
        m = man.get(r["clip_id"])
        if m is None:
            no_audio[r["dataset"]] += 1
            continue
        t = trs.get(r["clip_id"], {})
        rec = {
            "clip_id": r["clip_id"],
            "audio_file": Path(r["audio_path"]).name,
            "dataset": r["dataset"], "speaker_id": r["speaker_id"],
            "session_id": r["session_id"], "clip_type": r["clip_type"],
            "transcript": t.get("transcript"),
            "transcript_gold": r.get("transcript_gold"),
            "asr_avg_logprob": t.get("asr_avg_logprob"),
            "asr_no_speech_prob": t.get("asr_no_speech_prob"),
            "asr_model": t.get("asr_model"),
            "label_raw": r["label_raw"], "label_source": LABEL_SOURCE[r["dataset"]],
            "duration_raw_s": m["duration_raw_s"], "duration_speech_s": m["duration_speech_s"],
            "speech_segments": m["speech_segments"], "measured_lufs": m["measured_lufs"],
            "snr_db": m["snr_db"], "vad": m["vad"],
            "measure_path": rel(m["measure_path"], V2_DIR),
            "encoder_path": rel(m["encoder_path"], V2_DIR),
            "original_path": rel(r["audio_path"], root),
        }
        for k in ("sentence_id", "intensity", "timestamp"):
            if k in r:
                rec[k] = r[k]
        by_ds.setdefault(r["dataset"], []).append(rec)

    print(f"[INFO] Writing to {prep}")
    for ds, recs in by_ds.items():
        (prep / OUT_NAME[ds]).write_text(json.dumps(recs, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
        usable = sum(resolve(ds, r["label_raw"])["y7"] is not None for r in recs)
        with_tr = sum(bool(r["transcript"]) for r in recs)
        raw = Counter(r["label_raw"] for r in recs)
        print(f"\n  {OUT_NAME[ds]}: {len(recs)} clips | {usable} with a 7-class label "
              f"| {with_tr} with Whisper transcript | {no_audio[ds]} indexed but not preprocessed")
        print(f"    raw labels: {dict(raw.most_common(10))}")

    if not args.no_goemotions:
        rows = load_goemotions(sources(root)["goemotions"])
        if rows:
            (prep / "goemotions.json").write_text(json.dumps(rows, ensure_ascii=False),
                                                  encoding="utf-8")
            ok = sum(resolve("goemotions", r["label_raw"])["y7"] is not None for r in rows)
            print(f"\n  goemotions.json: {len(rows)} texts | {ok} map to a single class")


if __name__ == "__main__":
    main()
