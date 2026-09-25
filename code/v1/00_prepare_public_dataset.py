"""
00_prepare_public_dataset.py
═══════════════════════════════════════════════════════════════════════════════
Turn RAVDESS / CREMA-D / IEMOCAP / EmoDB into this pipeline's format, in ONE
pass over all of them, and carve a speaker-disjoint held-out split that the
encoder never sees.

    python 00_prepare_public_dataset.py \
        --base ~/Downloads/RAVDESS_IEMOCAP_CREMAD \
        --datasets ravdess,cremad,iemocap \
        --out data_public --holdout 0.2

Auto-detects `ravdess/`, `cremad/`, `iemocap/` under --base. Override any of
them with --ravdess_root / --cremad_root / --iemocap_root.

WHY THE HOLDOUT IS BY SPEAKER AND NOT BY FILE
─────────────────────────────────────────────────────────────────────────────
"Hold out 20% of the audio" sounds equivalent to "hold out 20% of the
speakers", and it is not. RAVDESS gives each actor 60 recordings; a random 20%
of files leaves the other 48 recordings of every held-out actor in training.
The encoder then reports on voices it has already memorised, the number comes
out several points high, and it is not comparable to any published result —
every serious SER paper splits by speaker (or, for IEMOCAP, by session).

So the split here is:
  * by SPEAKER within each corpus (default), or
  * by SESSION for IEMOCAP with --iemocap_holdout session, which reserves
    Session5 — the split the IEMOCAP literature actually uses, and the one to
    pick if you want your table to sit next to theirs.

The holdout is written once to splits.json and every later script reads it.
Never regenerate it with a different seed mid-project: the moment the holdout
changes, every number computed before it becomes incomparable.

WHAT COMES OUT
─────────────────────────────────────────────────────────────────────────────
  utterances.jsonl                flat per-utterance records, for 08
  splits.json                     train / holdout speaker lists per corpus
  firebase_data/video_index.json  paired-session view, for 02 → 03 → 04
  clip_sources.jsonl              stem → source wav
  materialise_audio.sh            symlinks the corpus into audios/

The paired-session view keeps the neutral-reference design: each emotional
utterance becomes one "session" whose passage is that speaker's neutral
utterance of the SAME sentence where the corpus allows it. No utterance is
ever its own reference, and reference selection never looks at the target's
label.

CAVEAT FOR THE PAPER
─────────────────────────────────────────────────────────────────────────────
These corpora are ACTED. The reference is acted-neutral, not read-neutral, and
the emotional speech is performed rather than spontaneous. They validate that
the normalisation mechanism works; they do not show it transfers to
naturalistic speech. Frame the public results as mechanism validation and the
in-house results as the in-the-wild demonstration — not one merged benchmark.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

CANON = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ── RAVDESS filename: 03-01-06-01-02-01-12.wav ───────────────────────────────
# modality-vocal-EMOTION-intensity-STATEMENT-repetition-ACTOR
RAVDESS_EMOTION = {
    "01": "neutral", "02": "neutral",     # 02 = "calm"; merged into neutral
    "03": "happy", "04": "sad", "05": "angry",
    "06": "fear", "07": "disgust", "08": "surprise",
}

CREMAD_EMOTION = {
    "ANG": "angry", "DIS": "disgust", "FEA": "fear",
    "HAP": "happy", "NEU": "neutral", "SAD": "sad",
}

EMODB_EMOTION = {
    "W": "angry", "L": "neutral",         # L = boredom; closest canonical slot
    "E": "disgust", "A": "fear",
    "F": "happy", "T": "sad", "N": "neutral",
}

IEMOCAP_EMOTION = {
    "ang": "angry", "hap": "happy", "exc": "happy",   # excited merged, standard
    "sad": "sad", "neu": "neutral", "fea": "fear",
    "dis": "disgust", "sur": "surprise",
    # Full words, for CSV label files that spell them out
    "anger": "angry", "angry": "angry",
    "happiness": "happy", "happy": "happy",
    "excited": "happy", "excitement": "happy",
    "sadness": "sad", "neutral": "neutral", "neutral state": "neutral",
    "fear": "fear", "fearful": "fear",
    "disgust": "disgust", "disgusted": "disgust",
    "surprise": "surprise", "surprised": "surprise",
    # Deliberately absent, so they are dropped rather than forced into a class:
    #   fru/frustration (~1800 utterances), oth/other, xxx (no annotator
    #   majority). Standard practice in the IEMOCAP literature.
}


# ─────────────────────────────────────────────────────────────────────────────
# Parsers -> list of {path, speaker, emotion, sentence_id, intensity, corpus}
# ─────────────────────────────────────────────────────────────────────────────
def parse_ravdess(root: Path) -> list[dict]:
    out = []
    for p in sorted(root.rglob("*.wav")):
        parts = p.stem.split("-")
        if len(parts) != 7:
            continue
        modality, _, emo, intensity, statement, _, actor = parts
        if modality != "03":              # audio-only speech files
            continue
        label = RAVDESS_EMOTION.get(emo)
        if label is None:
            continue
        out.append({
            "path": p, "corpus": "ravdess",
            "speaker": f"ravdess_actor{int(actor):02d}",
            "emotion": label,
            "sentence_id": statement,      # 01 / 02, lexically matched
            "intensity": intensity,
            "group": f"actor{int(actor):02d}",
        })
    return out


def parse_cremad(root: Path) -> list[dict]:
    out = []
    for p in sorted(root.rglob("*.wav")):
        parts = p.stem.split("_")
        if len(parts) < 4:
            continue
        actor, sentence, emo, level = parts[0], parts[1], parts[2], parts[3]
        label = CREMAD_EMOTION.get(emo.upper())
        if label is None:
            continue
        out.append({
            "path": p, "corpus": "cremad",
            "speaker": f"cremad_{actor}",
            "emotion": label, "sentence_id": sentence, "intensity": level,
            "group": actor,
        })
    return out


def parse_emodb(root: Path) -> list[dict]:
    # 03a01Fa.wav -> speaker 03, text a01, emotion F
    out = []
    for p in sorted(root.rglob("*.wav")):
        m = re.match(r"(\d{2})([a-z]\d{2})([A-Z])", p.stem)
        if not m:
            continue
        spk, text, emo = m.groups()
        label = EMODB_EMOTION.get(emo)
        if label is None:
            continue
        out.append({"path": p, "corpus": "emodb", "speaker": f"emodb_{spk}",
                    "emotion": label, "sentence_id": text, "intensity": "NA",
                    "group": spk})
    return out


def _iemocap_meta(utt_id: str) -> tuple[str, str, str, bool]:
    """
    (session, gender, dialog, is_script) from the utterance id alone.

        Ses01F_impro01_F000    -> Session1, F, Ses01F_impro01,      False
        Ses05M_script03_2_M042 -> Session5, M, Ses05M_script03_2,   True

    Everything needed is in the id, which is why this works on a flat folder
    of wavs with no directory structure at all.
    """
    parts = utt_id.split("_")
    m = re.match(r"[Ss]es(\d+)", utt_id)
    session = f"Session{int(m.group(1))}" if m else "Session0"
    gender = parts[-1][0].upper() if parts and parts[-1] else "U"
    dialog = "_".join(parts[:-1]) if len(parts) > 1 else utt_id
    return session, gender, dialog, ("script" in dialog.lower())


def _norm_emotion(value: str):
    return IEMOCAP_EMOTION.get(str(value).strip().lower())


def _iemocap_from_csv(root: Path, wav_index: dict, explicit: Path | None,
                      verbose: bool) -> list[dict]:
    """
    Fallback for repackaged IEMOCAP: a flat folder of wavs plus a CSV of
    labels. Detects the id column and the emotion column by their contents
    rather than by name, since every repackaging picks different headers.
    """
    import csv

    candidates = [explicit] if explicit else sorted(
        set(list(root.rglob("*.csv")) + list(root.parent.rglob("*.csv"))))
    if not candidates:
        return []

    id_re = re.compile(r"Ses\d+[FM]_", re.I)
    best: list[dict] = []

    for path in candidates:
        try:
            with open(path, newline="", encoding="utf-8", errors="ignore") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            continue
        if not rows or not rows[0]:
            continue

        cols = [c for c in rows[0].keys() if c]
        sample = rows[:200]
        id_col = next((c for c in cols if sum(
            1 for r in sample if id_re.search(str(r.get(c, "")))) > len(sample) * 0.5), None)
        emo_col = next((c for c in cols if sum(
            1 for r in sample if _norm_emotion(r.get(c, ""))) > len(sample) * 0.3), None)
        if not id_col or not emo_col or id_col == emo_col:
            continue

        items, unresolved = [], 0
        for r in rows:
            raw_id = str(r.get(id_col, "")).strip()
            utt_id = Path(raw_id).stem
            label = _norm_emotion(r.get(emo_col, ""))
            if not utt_id or not label:
                continue
            wav = wav_index.get(utt_id)
            if wav is None:
                unresolved += 1
                continue
            session, gender, dialog, is_script = _iemocap_meta(utt_id)
            items.append({
                "path": wav, "corpus": "iemocap",
                "speaker": f"iemocap_{session}_{gender}",
                "emotion": label,
                "sentence_id": dialog if is_script else "impro",
                "intensity": "NA", "group": session,
                "is_script": is_script, "lexically_matched": is_script,
            })

        if verbose:
            print(f"  labels from CSV {path.name}: id column '{id_col}', "
                  f"emotion column '{emo_col}' -> {len(items)} usable"
                  + (f", {unresolved} with no matching wav" if unresolved else ""))
        if len(items) > len(best):
            best = items

    return best


def _iemocap_session(utt_id: str, path: Path) -> str:
    """
    Session number, from the utterance id first (Ses01F_impro01_F000 -> 1) and
    from the path only as a fallback. The id is the reliable source: folder
    names differ between releases and repackagings, utterance ids do not.
    """
    m = re.match(r"[Ss]es(\d+)", utt_id)
    if m:
        return f"Session{int(m.group(1))}"
    for part in path.parts:
        m = re.fullmatch(r"[Ss]ession[ _-]?(\d+)", part)
        if m:
            return f"Session{int(m.group(1))}"
    return "Session0"


def parse_iemocap(root: Path, verbose: bool = True,
                  labels_csv: Path | None = None) -> list[dict]:
    """
    Reads the EmoEvaluation label files and matches them to wav paths.

    Layout-tolerant on purpose. The canonical release is
        Session{1..5}/dialog/EmoEvaluation/*.txt
        Session{1..5}/sentences/wav/<dialog>/<utt>.wav
    but IEMOCAP travels badly: it is often nested one level deeper inside
    IEMOCAP_full_release/, or repackaged with different capitalisation. So
    instead of walking a fixed path, this finds every EmoEvaluation directory
    anywhere under root and resolves wavs through one index of every wav file,
    keyed by stem. Utterance ids are globally unique, so the index is safe.
    """
    line_re = re.compile(r"^\[.*?\]\s+(\S+)\s+(\w+)\s+\[")

    eval_dirs = [d for d in root.rglob("EmoEvaluation") if d.is_dir()]
    wav_index: dict[str, Path] = {}
    n_wavs = 0
    for w in root.rglob("*.wav"):
        n_wavs += 1
        wav_index.setdefault(w.stem, w)

    out, n_lines, n_unknown_emo, n_unresolved = [], 0, 0, 0
    label_files = 0

    for eval_dir in sorted(eval_dirs):
        # Only the .txt files directly inside EmoEvaluation are the categorical
        # summaries. Categorical/ and Self-evaluation/ hold per-annotator files
        # in a different format.
        for txt in sorted(eval_dir.glob("*.txt")):
            if txt.name.startswith("."):
                continue
            label_files += 1
            for line in txt.read_text(errors="ignore").splitlines():
                m = line_re.match(line)
                if not m:
                    continue
                n_lines += 1
                utt_id, emo = m.groups()
                label = _norm_emotion(emo)
                if label is None:
                    n_unknown_emo += 1          # xxx, fru, oth
                    continue
                wav = wav_index.get(utt_id)
                if wav is None:
                    n_unresolved += 1
                    continue

                # Everything about the utterance comes from its id, so this
                # path and the CSV path below produce identical records. Only
                # the scripted portion is lexically matched across speakers, so
                # only it can support a same-sentence neutral reference.
                session, gender, dialog, is_script = _iemocap_meta(utt_id)
                out.append({
                    "path": wav, "corpus": "iemocap",
                    "speaker": f"iemocap_{session}_{gender}",
                    "emotion": label,
                    "sentence_id": dialog if is_script else "impro",
                    "intensity": "NA",
                    "group": session,
                    "is_script": is_script,
                    "lexically_matched": is_script,
                })

    if verbose:
        print(f"  EmoEvaluation dirs {len(eval_dirs)} | label files {label_files} "
              f"| wav files {n_wavs}")
        if eval_dirs:
            print(f"  label lines {n_lines} | kept {len(out)} | "
                  f"emotion not in our 7 classes {n_unknown_emo} | "
                  f"no matching wav {n_unresolved}")

    # Fallback: no EmoEvaluation files, which is what a flat folder of wavs
    # looks like. The filenames carry session, speaker and dialog but NOT the
    # emotion, so labels have to come from somewhere — usually a CSV shipped
    # with the repackaged download.
    if not out:
        out = _iemocap_from_csv(root, wav_index, labels_csv, verbose)

    if verbose and not out:
        print("  [!] No usable IEMOCAP labels found.")
        print("      The wav filenames encode session, speaker and dialog, but")
        print("      NOT the emotion — unlike RAVDESS and CREMA-D, where the")
        print("      label is in the filename. IEMOCAP keeps its labels in")
        print("      Session*/dialog/EmoEvaluation/*.txt.")
        print("      Either put those .txt files anywhere under this folder, or")
        print("      point --iemocap_labels at a CSV that has an utterance-id")
        print("      column and an emotion column. Run inspect_corpus.py to see")
        print("      which CSVs are present and what their headers look like.")
    return out


PARSERS = {
    "ravdess": parse_ravdess,
    "cremad": parse_cremad,
    "emodb": parse_emodb,
    "iemocap": parse_iemocap,
}


# ─────────────────────────────────────────────────────────────────────────────
# Neutral-reference pairing
# ─────────────────────────────────────────────────────────────────────────────
def build_pairs(items: list[dict], match_sentence: bool = True,
                keep_neutral_targets: bool = True, verbose: bool = True
                ) -> list[dict]:
    """
    Emit one "session" per emotional utterance:
        passage   = that speaker's neutral utterance (same sentence if possible)
        freestyle = the emotional utterance
    """
    neutrals: dict[tuple, list[dict]] = defaultdict(list)
    for it in items:
        if it["emotion"] == "neutral":
            neutrals[(it["speaker"], it["sentence_id"])].append(it)
            neutrals[(it["speaker"], "__any__")].append(it)

    # The first neutral take of a speaker+sentence is reserved as the reference
    # and never emitted as its own target. Extra takes can still be targets,
    # which is what keeps 'neutral' in the label set.
    reserved = {id(v[0]) for v in neutrals.values() if v}

    sessions, n_exact, n_fallback, n_dropped, n_neutral_targets = [], 0, 0, 0, 0

    for it in items:
        ref = None
        if it["emotion"] == "neutral":
            if not keep_neutral_targets or id(it) in reserved:
                continue                   # reference only
            cands = [c for c in neutrals.get((it["speaker"], it["sentence_id"]), [])
                     if id(c) != id(it)]
            if not cands:
                cands = [c for c in neutrals.get((it["speaker"], "__any__"), [])
                         if id(c) != id(it)]
            if not cands:
                continue
            n_neutral_targets += 1
            ref = cands[0]
            n_exact += 1

        if ref is None and match_sentence:
            c = neutrals.get((it["speaker"], it["sentence_id"]), [])
            if c:
                ref = c[0]
                n_exact += 1
        if ref is None:
            c = neutrals.get((it["speaker"], "__any__"), [])
            if c:
                ref = c[0]
                n_fallback += 1
        if ref is None:
            n_dropped += 1
            continue

        sid = f"{it['speaker']}__{it['path'].stem}"
        sessions.append({
            "corpus": it["corpus"],
            "user_id": it["speaker"],
            "group": it.get("group", it["speaker"]),
            "session_id": sid,
            "emotion": it["emotion"],
            "freestyle_path": it["path"],
            "passage_path": ref["path"],
            "sentence_id": it["sentence_id"],
            # Honest labelling of what the reference actually controls.
            # IEMOCAP improvised turns all share the placeholder sentence id
            # "impro", so ids matching there does NOT mean the words matched —
            # only the speaker and session did. Calling that a lexically
            # matched reference would overstate the central claim.
            "reference_match": "same_sentence" if (
                match_sentence and it.get("lexically_matched", True)
                and it["sentence_id"] == ref["sentence_id"]
            ) else "speaker_only",
        })

    if verbose:
        n_lex = sum(1 for x in sessions if x["reference_match"] == "same_sentence")
        if n_lex < len(sessions):
            print(f"  lexically matched references          : {n_lex}/{len(sessions)}"
                  f"  (the rest control speaker and session only)")
        if n_neutral_targets:
            print(f"  neutral kept as target (extra takes)  : {n_neutral_targets}")
        print(f"  reference matched on speaker+sentence : {n_exact}")
        print(f"  reference matched on speaker only     : {n_fallback}")
        if n_dropped:
            print(f"  [!] dropped (no neutral for speaker)  : {n_dropped}")
    return sessions


# ─────────────────────────────────────────────────────────────────────────────
# Speaker-disjoint holdout
# ─────────────────────────────────────────────────────────────────────────────
def make_splits(sessions: list[dict], holdout: float, seed: int,
                iemocap_holdout: str = "speaker") -> dict:
    """
    Reserve a fraction of SPEAKERS per corpus for the final benchmark. The
    encoder is trained only on the remainder.

    For IEMOCAP with iemocap_holdout='session', Session5 is reserved instead —
    the split most IEMOCAP papers report, so your numbers line up with theirs.
    """
    by_corpus: dict[str, set] = defaultdict(set)
    spk_group: dict[str, str] = {}
    for s in sessions:
        by_corpus[s["corpus"]].add(s["user_id"])
        spk_group[s["user_id"]] = s.get("group", s["user_id"])

    rng = random.Random(seed)
    splits = {"seed": seed, "holdout_frac": holdout, "by_corpus": {}}

    for corpus, speaker_set in sorted(by_corpus.items()):
        speakers = sorted(speaker_set)
        mode, held = "speaker", []

        if corpus == "iemocap" and iemocap_holdout == "session":
            held = sorted(s for s in speakers if spk_group[s].lower() == "session5")
            if held:
                mode = "session5"

        if not held:
            n_held = max(1, int(round(len(speakers) * holdout)))
            shuffled = speakers[:]
            rng.shuffle(shuffled)
            held = sorted(shuffled[:n_held])

        train = sorted(set(speakers) - set(held))
        splits["by_corpus"][corpus] = {
            "mode": mode,
            "n_speakers": len(speakers),
            "train_speakers": train,
            "holdout_speakers": held,
        }
        print(f"  {corpus:<9} {len(speakers):3d} speakers -> "
              f"train {len(train):3d} | holdout {len(held):3d}  ({mode})")

    return splits


def assign_split(sessions: list[dict], splits: dict) -> None:
    held = set()
    for c in splits["by_corpus"].values():
        held.update(c["holdout_speakers"])
    for s in sessions:
        s["split"] = "holdout" if s["user_id"] in held else "train"


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────
def materialise(copy_plan: list[dict], out_dir: Path, mode: str) -> None:
    """
    Build audios/<stem>.wav directly, without going through the shell script.

    Links, not copies, by design: every emotional utterance points at the same
    neutral reference, so copying duplicates each reference dozens of times. On
    IEMOCAP that is the difference between a few megabytes of links and several
    gigabytes of redundant audio.

    mode='auto' degrades gracefully, which is what Windows needs: symlinks
    require Developer Mode or an elevated shell there, so it falls back to a
    hardlink (same inode, no extra bytes, but source and destination must be on
    the same volume) and only then to a real copy.
    """
    import os
    import shutil

    audio_dir = out_dir / "audios"
    audio_dir.mkdir(parents=True, exist_ok=True)
    counts = {"symlink": 0, "hardlink": 0, "copy": 0}
    failed: list[str] = []

    order = {"auto": ["symlink", "hardlink", "copy"],
             "symlink": ["symlink"], "hardlink": ["hardlink"],
             "copy": ["copy"]}[mode]

    for c in copy_plan:
        src, dst = Path(c["source"]), audio_dir / f"{c['stem']}.wav"
        if not src.exists():
            failed.append(f"{c['stem']} (source missing)")
            continue
        if dst.exists() or dst.is_symlink():
            try:
                dst.unlink()
            except OSError as exc:
                failed.append(f"{c['stem']} ({exc})")
                continue

        last = None
        for how in order:
            try:
                if how == "symlink":
                    dst.symlink_to(src.resolve())
                elif how == "hardlink":
                    os.link(src, dst)
                else:
                    shutil.copy2(src, dst)
                counts[how] += 1
                last = None
                break
            except (OSError, NotImplementedError) as exc:
                last = exc
        if last is not None:
            failed.append(f"{c['stem']} ({last})")

    used = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
    print(f"  materialised {sum(counts.values())} clip(s) into {audio_dir}"
          f"{' — ' + used if used else ''}")
    if counts["copy"] and mode == "auto":
        print("  [note] links were unavailable, so clips were copied. On Windows,"
              " enabling\n         Developer Mode lets symlinks work and saves the"
              " duplicated bytes.")
    if failed:
        print(f"  [!] {len(failed)} failed, e.g. {failed[:3]}")


def write_outputs(sessions: list[dict], out_dir: Path, splits: dict,
                  materialise_mode: str = "none") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "firebase_data").mkdir(exist_ok=True)

    video_index, copy_plan, utterances = [], [], []

    for s in sessions:
        for vtype, key in (("passage", "passage_path"), ("freestyle", "freestyle_path")):
            stem = f"{s['session_id']}__{vtype}"
            video_index.append({
                "video_id": f"{stem}-webm",
                "user_id": s["user_id"],
                "session_id": s["session_id"],
                "video_type": vtype,
                "reported_emotion": s["emotion"].capitalize(),
                "local_path": f"/{stem}.webm",
                "timestamp": 0,
                "corpus": s["corpus"],
                "split": s["split"],
            })
            copy_plan.append({"stem": stem, "source": str(s[key])})

        utterances.append({
            "stem": f"{s['session_id']}__freestyle",
            "ref_stem": f"{s['session_id']}__passage",
            "path": str(s["freestyle_path"]),
            "ref_path": str(s["passage_path"]),
            "corpus": s["corpus"],
            "speaker": s["user_id"],
            "group": s["group"],
            "session_id": s["session_id"],
            "emotion": s["emotion"],
            "sentence_id": s["sentence_id"],
            "reference_match": s["reference_match"],
            "split": s["split"],
        })

    with open(out_dir / "firebase_data" / "video_index.json", "w") as f:
        json.dump(video_index, f, indent=2)
    with open(out_dir / "clip_sources.jsonl", "w") as f:
        for c in copy_plan:
            f.write(json.dumps(c) + "\n")
    with open(out_dir / "utterances.jsonl", "w") as f:
        for u in utterances:
            f.write(json.dumps(u) + "\n")

    # Label set actually present per corpus. 08 uses this to mask the loss:
    # CREMA-D has no 'surprise', so a CREMA-D sample must not be penalised for
    # not predicting it.
    present: dict[str, set] = defaultdict(set)
    for s in sessions:
        present[s["corpus"]].add(s["emotion"])
    splits["labels_per_corpus"] = {c: sorted(v) for c, v in present.items()}
    with open(out_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    sh = out_dir / "materialise_audio.sh"
    with open(sh, "w") as f:
        f.write("#!/bin/bash\n# Symlink corpus audio into the pipeline layout.\n")
        f.write(f"mkdir -p '{out_dir}/audios'\n")
        for c in copy_plan:
            f.write(f"ln -sf '{c['source']}' '{out_dir}/audios/{c['stem']}.wav'\n")
    try:
        sh.chmod(0o755)
    except Exception:
        pass

    # Cross-platform materialiser. Windows refuses symlinks without Developer
    # Mode or an elevated shell, so this tries hardlink first and falls back to
    # copying. Each clip appears under several stems (one neutral reference is
    # shared by many targets), so copying can multiply the corpus on disk —
    # hardlinks avoid that whenever source and destination share a volume.
    py = out_dir / "materialise_audio.py"
    py.write_text(
        "import json, os, shutil, sys\n"
        "from pathlib import Path\n\n"
        "HERE = Path(__file__).resolve().parent\n"
        "DEST = HERE / 'audios'\n"
        "DEST.mkdir(parents=True, exist_ok=True)\n"
        "mode = sys.argv[1] if len(sys.argv) > 1 else 'auto'   # auto|link|copy\n"
        "plan = [json.loads(l) for l in open(HERE / 'clip_sources.jsonl') if l.strip()]\n"
        "made = skipped = copied = 0\n"
        "for c in plan:\n"
        "    src, dst = Path(c['source']), DEST / (c['stem'] + '.wav')\n"
        "    if dst.exists():\n"
        "        skipped += 1\n"
        "        continue\n"
        "    if not src.exists():\n"
        "        print('[!] missing source:', src)\n"
        "        continue\n"
        "    if mode != 'copy':\n"
        "        try:\n"
        "            os.link(src, dst)\n"
        "            made += 1\n"
        "            continue\n"
        "        except OSError:\n"
        "            if mode == 'link':\n"
        "                raise\n"
        "    shutil.copy2(src, dst)\n"
        "    copied += 1\n"
        "print(f'{made} hardlinked, {copied} copied, {skipped} already there '\n"
        "      f'-> {DEST}')\n",
        encoding="utf-8")

    n_hold = sum(1 for s in sessions if s["split"] == "holdout")
    print(f"\n  utterances.jsonl : {len(utterances)} paired sessions "
          f"({len(utterances) - n_hold} train / {n_hold} holdout)")
    print(f"  video_index.json : {len(video_index)} clips")
    print(f"  clip_sources.jsonl, splits.json, materialise_audio.sh written")

    if materialise_mode != "none":
        print(f"\n  Materialising audio ({materialise_mode}) ...")
        materialise(copy_plan, out_dir, materialise_mode)
        print(f"\n  Next:")
    else:
        print(f"\n  Next:")
        print(f"    bash {sh}        # or re-run with --materialise symlink")
    print(f"    python 02_audio_preprocessing.py --audio_root {out_dir}/audios "
          f"--output_dir {out_dir}/PreprocessedAudio")


def summarise(sessions: list[dict]) -> None:
    by_corpus_emo: dict = defaultdict(lambda: defaultdict(int))
    by_spk: dict = defaultdict(int)
    for s in sessions:
        by_corpus_emo[s["corpus"]][s["emotion"]] += 1
        by_spk[s["user_id"]] += 1

    print(f"\n  Speakers: {len(by_spk)}   Paired sessions: {len(sessions)}")
    header = "  " + "corpus".ljust(10) + "".join(e[:5].rjust(8) for e in CANON) + "   total"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for corpus, emos in sorted(by_corpus_emo.items()):
        row = "  " + corpus.ljust(10)
        for e in CANON:
            row += (str(emos.get(e, 0)) if emos.get(e) else "-").rjust(8)
        row += str(sum(emos.values())).rjust(8)
        print(row)
    totals: dict = defaultdict(int)
    for emos in by_corpus_emo.values():
        for e, n in emos.items():
            totals[e] += n
    row = "  " + "ALL".ljust(10) + "".join(str(totals.get(e, 0)).rjust(8) for e in CANON)
    print(row + str(sum(totals.values())).rjust(8))


def main():
    p = argparse.ArgumentParser(description="Public corpora -> pipeline format")
    p.add_argument("--base", type=Path, default=None,
                   help="Folder holding ravdess/ cremad/ iemocap/ subfolders")
    p.add_argument("--datasets", default="ravdess,cremad,iemocap")
    p.add_argument("--ravdess_root", type=Path, default=None)
    p.add_argument("--cremad_root", type=Path, default=None)
    p.add_argument("--iemocap_root", type=Path, default=None)
    p.add_argument("--emodb_root", type=Path, default=None)
    p.add_argument("--iemocap_labels", type=Path, default=None,
                   help="CSV of IEMOCAP labels, for repackaged downloads that "
                        "ship a flat wav folder without EmoEvaluation files")
    p.add_argument("--out", type=Path, default=Path.cwd() / "data_public")
    p.add_argument("--holdout", type=float, default=0.2,
                   help="Fraction of SPEAKERS reserved per corpus (default 0.2)")
    p.add_argument("--iemocap_holdout", default="session",
                   choices=["speaker", "session"],
                   help="'session' reserves Session5, the IEMOCAP convention")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop_neutral", action="store_true",
                   help="Use every neutral take as reference only, removing "
                        "'neutral' from the label set")
    p.add_argument("--no_sentence_match", action="store_true",
                   help="Match the neutral reference on speaker only")
    p.add_argument("--materialise", default="auto",
                   choices=["auto", "symlink", "hardlink", "copy", "none"],
                   help="Build audios/ directly instead of via the shell "
                        "script. 'auto' tries symlink, then hardlink, then "
                        "copy — the one to use on Windows. 'copy' duplicates "
                        "every neutral reference once per emotional utterance")
    args = p.parse_args()

    wanted = [d.strip() for d in args.datasets.split(",") if d.strip()]
    roots: dict[str, Path] = {}
    for d in wanted:
        explicit = getattr(args, f"{d}_root", None)
        if explicit:
            roots[d] = explicit.expanduser().resolve()
        elif args.base:
            roots[d] = (args.base.expanduser().resolve() / d)
        else:
            raise SystemExit(f"[ERROR] Give --base or --{d}_root for '{d}'")
        if not roots[d].exists():
            raise SystemExit(f"[ERROR] Not found: {roots[d]}")

    all_sessions: list[dict] = []
    for d in wanted:
        print(f"\n[INFO] {d}: {roots[d]}")
        items = (parse_iemocap(roots[d], labels_csv=args.iemocap_labels)
                 if d == "iemocap" else PARSERS[d](roots[d]))
        if not items:
            print("  [!] No utterances parsed — check the layout. Skipping.")
            continue
        print(f"  {len(items)} utterances, {len({i['speaker'] for i in items})} speakers")
        sess = build_pairs(items,
                           match_sentence=not args.no_sentence_match,
                           keep_neutral_targets=not args.drop_neutral)
        print(f"  -> {len(sess)} paired sessions")
        all_sessions.extend(sess)

    if not all_sessions:
        raise SystemExit("[ERROR] Nothing parsed from any corpus.")

    summarise(all_sessions)

    print(f"\n[INFO] Speaker-disjoint holdout ({args.holdout:.0%} of speakers):")
    splits = make_splits(all_sessions, args.holdout, args.seed,
                         iemocap_holdout=args.iemocap_holdout)
    assign_split(all_sessions, splits)

    write_outputs(all_sessions, args.out, splits, args.materialise)


if __name__ == "__main__":
    main()