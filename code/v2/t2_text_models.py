"""
t2_text_models.py - Text branch over the shared folds.

    python t2_text_models.py --protocols grouped                           # direct + frozen, 3 encoders
    python t2_text_models.py --modes finetune --encoders emo_distilroberta --protocols grouped
    python t2_text_models.py --lexicon --emolex NRC-Emotion-Lexicon-Wordlevel-v0.92.txt --vad NRC-VAD-Lexicon.txt

Modes
  direct    the encoder's own emotion head, no training on our data (the floor)
  frozen    encoder fixed; [mean embedding, its emotion probs, ASR quality, (lexicon)] -> MLP
  finetune  encoder updated; starts from models/stage1/text_<enc> if it exists
Granularity
  full      one unit per session (the whole freestyle transcript)
  sentence  one unit per sentence, averaged back to the session

Every split is made over SESSIONS (from splits.json), so sentences of one session
never land on both sides. Output: text_result/<run>/<protocol>/seed<k>/ + summary.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_common import OOF, load_evaluation, macro_f1, summarise
from config import TEXT_EVAL_SETS, ensure_dirs, seed_everything, set_eval
from report import write_report
from stats import paired_bootstrap_test
from text_common import (DEVICE, EMOTION_LABELS, Lexicon, aggregate_by_owner, compute_class_weights,
                         embed_texts, finetune, free, load_classifier, load_sessions, load_tokenizer,
                         predict_probs, resolve_encoder, split_sentences)

MAX_LEN = {"full": 512, "sentence": 64}
FT_BATCH = {"full": 8, "sentence": 32}
FT_EPOCHS = {"full": 6, "sentence": 4}


def build_units(sessions, gran):
    texts, owner = [], []
    for i, s in enumerate(sessions):
        parts = [s["text"]] if gran == "full" else split_sentences(s["text"])
        texts += parts
        owner += [i] * len(parts)
    return texts, np.array(owner)


def rows_of(owner, idx):
    return np.where(np.isin(owner, idx))[0]


def frozen_features(model_id, texts, gran, cache: Path):
    """Mean-pooled embeddings (+ emotion-head probs), cached on disk per model and unit set."""
    key = hashlib.md5((model_id + gran + "|".join(texts)).encode("utf-8")).hexdigest()[:16]
    fe, fp = cache / f"{key}_emb.npy", cache / f"{key}_prb.npy"
    if fe.exists():
        return np.load(fe), (np.load(fp) if fp.exists() else None)
    emb, prb = embed_texts(model_id, texts, MAX_LEN[gran])
    cache.mkdir(parents=True, exist_ok=True)
    np.save(fe, emb)
    if prb is not None:
        np.save(fp, prb)
    return emb, prb


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.3),
                                 nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 7))

    def forward(self, x):
        return self.net(x)


def fit_mlp(X, y_unit, owner, y_sess, tr_rows, va_rows, va_sess, n_sess, seed):
    """Train on units; early-stop on SESSION-level macro F1 of the validation sessions."""
    torch.manual_seed(seed)
    mu, sd = X[tr_rows].mean(0), X[tr_rows].std(0)
    Xn = torch.tensor((X - mu) / np.where(sd > 1e-8, sd, 1), dtype=torch.float32, device=DEVICE)
    model = MLP(X.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    w = compute_class_weights(y_unit[tr_rows]).to(DEVICE)
    yt = torch.tensor(y_unit, device=DEVICE)
    rng = np.random.RandomState(seed)
    best, state, bad = -1.0, None, 0
    for _ in range(150):
        model.train()
        for b in np.array_split(rng.permutation(tr_rows), max(1, len(tr_rows) // 32)):
            loss = F.cross_entropy(model(Xn[b]), yt[b], weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            p = F.softmax(model(Xn[va_rows]), -1).cpu().numpy()
        sp = aggregate_by_owner(p, owner[va_rows], n_sess)
        score = macro_f1(y_sess[va_sess], sp[va_sess].argmax(1))
        if score > best:
            best, bad, state = score, 0, {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 20:
                break
    model.load_state_dict(state); model.eval()

    def prob(rows):
        with torch.no_grad():
            return F.softmax(model(Xn[rows]), -1).cpu().numpy()
    return prob


def stage1_checkpoint(dirs, alias, eval_set):
    """
    Stage-1 text checkpoint to start fine-tuning from, or None.
    When evaluating ON IEMOCAP, a checkpoint that saw IEMOCAP in stage 1 would have seen
    the test utterances, so only the --no_iemocap one (GoEmotions only) is allowed.
    """
    import json as _json
    base = dirs["models"] / "stage1"
    if eval_set != "iemocap":
        ck = base / f"text_{alias}"
        return ck if ck.exists() else None
    ck = base / f"text_{alias}_noiemocap"
    if ck.exists():
        info = _json.loads((ck / "stage1_info.json").read_text(encoding="utf-8"))
        if info.get("used_iemocap", True):
            raise SystemExit(f"[ERROR] {ck} was trained with IEMOCAP; cannot evaluate on IEMOCAP.")
        return ck
    if (base / f"text_{alias}").exists():
        print(f"  [INFO] {alias}: stage-1 checkpoint includes IEMOCAP, so it is NOT used for --eval iemocap.")
        print(f"         For stage 1 here run: python t1_text_pretrain.py --encoder {alias} --no_iemocap")
    return None


def main():
    ap = argparse.ArgumentParser(description="Text branch")
    ap.add_argument("--encoders", default="minilm,emo_distilroberta,goemo_roberta")
    ap.add_argument("--modes", default="direct,frozen")
    ap.add_argument("--granularity", default="full,sentence")
    ap.add_argument("--protocols", default="grouped")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--lexicon", action="store_true")
    ap.add_argument("--emolex", default=None)
    ap.add_argument("--vad", default=None)
    ap.add_argument("--no_stage1", action="store_true")
    ap.add_argument("--lr_encoder", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-4)
    ap.add_argument("--eval", default="own", choices=TEXT_EVAL_SETS,
                    help="own or iemocap (RAVDESS / CREMA-D repeat fixed sentences: no text signal)")
    a = ap.parse_args()
    set_eval(a.eval)

    dirs = ensure_dirs()
    print(f"[INFO] evaluation set: {a.eval} | results -> {dirs['text_result']}")
    out_root = dirs["text_result"]
    ev, splits = load_evaluation()
    sess = load_sessions()
    keys = [s["key"] for s in sess]
    pos = {k: i for i, k in enumerate(keys)}
    y = np.array([s["y7"] for s in sess])
    spk = np.array([s["user_id"] for s in sess])
    asr = np.array([[s["asr_logprob"], s["asr_nospeech"]] for s in sess], dtype=np.float32)
    lex = Lexicon(a.emolex, a.vad) if a.lexicon else None
    texts_by_key = [s["text"] for s in sess]
    results = {}

    for enc_spec in a.encoders.split(","):
        alias, model_id = resolve_encoder(enc_spec.strip())
        for gran in a.granularity.split(","):
            texts, owner = build_units(sess, gran)
            unit_y = y[owner]
            for mode in a.modes.split(","):
                if mode in ("direct", "frozen"):
                    emb, prb = frozen_features(model_id, texts, gran, out_root / "cache")
                    if mode == "direct" and prb is None:
                        print(f"  [SKIP] {alias}|{gran}|direct: no emotion head"); continue
                    X = None
                    if mode == "frozen":
                        parts = [emb] + ([prb] if prb is not None else []) + [asr[owner]]
                        if lex:
                            parts.append(np.stack([lex.features(t) for t in texts]))
                        X = np.hstack(parts).astype(np.float32)
                run = f"{alias}_{gran}_{mode}{'_lex' if (lex and mode == 'frozen') else ''}"
                if mode == "finetune":
                    ck = stage1_checkpoint(dirs, alias, a.eval)
                    source = str(ck) if ck is not None and not a.no_stage1 else model_id
                    run += "_stage1" if source == str(ck) else ""

                for proto in a.protocols.split(","):
                    for seed in [int(s) for s in a.seeds.split(",")]:
                        seed_everything(seed)
                        oof = OOF(keys)
                        for fi, fold in enumerate(splits[str(seed)][proto]):
                            tr = np.array([pos[k] for k in fold["train"]]); va = np.array([pos[k] for k in fold["val"]])
                            te = np.array([pos[k] for k in fold["test"]])
                            if mode == "direct":
                                sp = aggregate_by_owner(prb, owner, len(sess))
                                pte, pva = sp[te], sp[va]
                            elif mode == "frozen":
                                prob = fit_mlp(X, unit_y, owner, y, rows_of(owner, tr), rows_of(owner, va),
                                               va, len(sess), seed)
                                r_te, r_va = rows_of(owner, te), rows_of(owner, va)
                                pte = aggregate_by_owner(prob(r_te), owner[r_te], len(sess))[te]
                                pva = aggregate_by_owner(prob(r_va), owner[r_va], len(sess))[va]
                            else:
                                tok = load_tokenizer(source)
                                model, _ = load_classifier(source)
                                r_tr, r_va, r_te = rows_of(owner, tr), rows_of(owner, va), rows_of(owner, te)
                                vt = [texts[j] for j in r_va]

                                def val_fn(m):
                                    p = aggregate_by_owner(predict_probs(m, tok, vt, MAX_LEN[gran]), owner[r_va], len(sess))
                                    return macro_f1(y[va], p[va].argmax(1))
                                model, _ = finetune(model, tok, [texts[j] for j in r_tr], unit_y[r_tr], val_fn=val_fn,
                                                    epochs=FT_EPOCHS[gran], batch=FT_BATCH[gran], max_len=MAX_LEN[gran],
                                                    seed=seed, lr_encoder=a.lr_encoder, lr_head=a.lr_head, verbose=False)
                                pte = aggregate_by_owner(predict_probs(model, tok, [texts[j] for j in r_te], MAX_LEN[gran]),
                                                         owner[r_te], len(sess))[te]
                                pva = aggregate_by_owner(predict_probs(model, tok, vt, MAX_LEN[gran]),
                                                         owner[r_va], len(sess))[va]
                                free(model)
                            oof.add(fi, fold["name"], [keys[i] for i in te], pte, [keys[i] for i in va], pva)

                        exp = out_root / run / proto / f"seed{seed}"
                        exp.mkdir(parents=True, exist_ok=True)
                        done = oof.save(exp / "oof.json", y)
                        pred = oof.test.argmax(1)
                        write_report(exp, y[done], pred[done], oof.test[done], spk[done],
                                     [k for k, d in zip(keys, done) if d], title=f"text {run} | {proto} | seed {seed}",
                                     extra_cols={"transcript": [t for t, d in zip(texts_by_key, done) if d]},
                                     config={"run": run, "encoder": model_id, "mode": mode, "granularity": gran,
                                             "lexicon": bool(lex), "protocol": proto, "seed": seed},
                                     seed=seed, verbose=False)
                        results[(run, proto, seed)] = pred
                        m = summarise(y[done], pred[done])
                        print(f"  {run:<40} {proto:<8} seed {seed}  acc7 {m['acc7']:.3f}  F1_7 {m['f1_7']:.3f}  "
                              f"acc3 {m['acc3']:.3f}  F1_3 {m['f1_3']:.3f}")

    for proto in a.protocols.split(","):
        seed0 = int(a.seeds.split(",")[0])
        runs = sorted({r for r, p, s in results if p == proto and s == seed0},
                      key=lambda r: -summarise(y, results[(r, proto, seed0)])["f1_7"])
        if not runs:
            continue
        rows, best = [], runs[0]
        print(f"\n== {proto}: sorted by macro F1 (7 emotions), seed {seed0}")
        for r in runs:
            m = summarise(y, results[(r, proto, seed0)])
            txt = ""
            if r != best:
                t = paired_bootstrap_test(y, results[(best, proto, seed0)], results[(r, proto, seed0)], spk,
                                          "macro_f1", seed=seed0)
                txt = f"delta {t['delta']:+.3f} p={t['p_value']:.3f}{' *' if t['significant_05'] else ''}"
            print(f"   {r:<40}{m['acc7']:>7.3f}{m['f1_7']:>7.3f}{m['acc3']:>7.3f}{m['f1_3']:>7.3f}   {txt}")
            rows.append({"run": r, **m})
        with open(out_root / f"summary_{proto}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
