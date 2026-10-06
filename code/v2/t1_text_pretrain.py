"""
t1_text_pretrain.py - Stage 1 of text fine-tuning, on PUBLIC data only.

    python t1_text_pretrain.py --encoder emo_distilroberta
    python t1_text_pretrain.py --encoder goemo_roberta --iemocap_text whisper

Data (from preprocessing/):
  goemotions.json   27 fine emotions collapsed to our 7 with the authors' Ekman
                    grouping; texts whose labels span several groups are dropped.
                    The only source with real disgust / fear / surprise examples.
  iemocap.json      IMPROVISED utterances only: in scripted ones everybody says
                    the same lines, so the text carries no emotion. Whisper
                    transcripts by default, so the model sees real ASR errors.

No in-house data is used, so any CV fold can start from this checkpoint.
Output: models/stage1/text_<encoder>/
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter

import numpy as np
from sklearn.metrics import f1_score

from config import ensure_dirs
from labels import resolve
from text_common import (DEVICE, EMOTION_LABELS, finetune, free, label_to_canon, load_classifier,
                         load_records, load_tokenizer, predict_probs, resolve_encoder)


def goemotions_pairs() -> list[tuple[str, str]]:
    try:
        recs = load_records("goemotions.json").values()
    except FileNotFoundError:
        return []
    out = []
    for r in recs:
        labs = r["label_raw"] if isinstance(r["label_raw"], list) else [r["label_raw"]]
        canon = {label_to_canon(l) for l in labs} - {None}
        if len(canon) == 1:
            out.append((r["text"], canon.pop()))
    return out


def iemocap_pairs(which: str) -> list[tuple[str, str]]:
    try:
        recs = load_records("iemocap.json").values()
    except FileNotFoundError:
        return []
    out = []
    for r in recs:
        if r.get("clip_type") != "improvised":
            continue
        lab = resolve("iemocap", r.get("label_raw"))
        if lab.get("dominant") is None:
            continue
        texts = {"whisper": [r.get("transcript")], "gold": [r.get("transcript_gold")],
                 "both": [r.get("transcript"), r.get("transcript_gold")]}[which]
        out += [(t, lab["dominant"]) for t in texts if t and t.strip()]
    return out


def main():
    ap = argparse.ArgumentParser(description="Stage-1 text fine-tuning")
    ap.add_argument("--encoder", required=True, help="minilm | emo_distilroberta | goemo_roberta | HF id | path")
    ap.add_argument("--no_goemotions", action="store_true")
    ap.add_argument("--no_iemocap", action="store_true")
    ap.add_argument("--iemocap_text", default="whisper", choices=["whisper", "gold", "both"])
    ap.add_argument("--cap_per_class", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--lr_encoder", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    alias, model_id = resolve_encoder(a.encoder)
    random.seed(a.seed)
    data = ([] if a.no_goemotions else goemotions_pairs()) + ([] if a.no_iemocap else iemocap_pairs(a.iemocap_text))
    if not data:
        raise SystemExit("[ERROR] No stage-1 data. Run p4_build_datasets.py (GoEmotions and/or IEMOCAP).")

    by = {c: [] for c in EMOTION_LABELS}
    for t, c in data:
        by[c].append(t)
    train, val = [], []
    for c, texts in by.items():                 # cap each class, keep 10% for validation
        random.shuffle(texts)
        texts = texts[: a.cap_per_class]
        k = max(1, len(texts) // 10) if texts else 0
        val += [(t, c) for t in texts[:k]]
        train += [(t, c) for t in texts[k:]]
    random.shuffle(train)
    cnt = Counter(c for _, c in train)
    print(f"[INFO] stage-1 train={len(train)} val={len(val)} | " +
          " ".join(f"{c}:{cnt.get(c, 0)}" for c in EMOTION_LABELS))

    tok = load_tokenizer(model_id)
    model, head = load_classifier(model_id)
    print(f"[INFO] {model_id}: head {head} | device {DEVICE}")
    vx, vy = [t for t, _ in val], np.array([EMOTION_LABELS.index(c) for _, c in val])

    def val_fn(m):
        return f1_score(vy, predict_probs(m, tok, vx, a.max_len, 64).argmax(1), average="macro", zero_division=0)

    before = val_fn(model.to(DEVICE))
    model, best = finetune(model, tok, [t for t, _ in train], np.array([EMOTION_LABELS.index(c) for _, c in train]),
                           val_fn=val_fn, epochs=a.epochs, batch=a.batch, max_len=a.max_len, seed=a.seed,
                           lr_encoder=a.lr_encoder, lr_head=a.lr_head)
    out = ensure_dirs()["models"] / "stage1" / f"text_{alias}"
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out); tok.save_pretrained(out)
    with open(out / "stage1_info.json", "w", encoding="utf-8") as f:
        json.dump({"source": model_id, "head": head, "train_counts": dict(cnt), "n_val": len(val),
                   "val_macro_f1_before": before, "val_macro_f1_after": best,
                   "iemocap_text": a.iemocap_text, "note": "Public data only."}, f, indent=2)
    free(model)
    print(f"[INFO] val macro F1 {before:.3f} -> {best:.3f} | saved -> {out}")


if __name__ == "__main__":
    main()
