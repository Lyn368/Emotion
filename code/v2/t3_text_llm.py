"""
t3_text_llm.py - LLM zero-shot on the freestyle transcript, as probabilities.

Instead of asking the model to WRITE a label and parsing the reply (which can
fail and silently turn into "neutral"), the model SCORES each of the seven
labels: log P(label | prompt). A softmax over the seven scores gives a proper
probability vector, which the fusion stage can use like any other branch.

    python t3_text_llm.py                                   # Qwen2.5-3B, fits 8 GB in fp16
    python t3_text_llm.py --model Qwen/Qwen2.5-7B-Instruct --load_4bit   # needs bitsandbytes
    python t3_text_llm.py --limit 20                        # timing test

The prompt is FROZEN: written before seeing any result and identified by its
SHA-1, which is saved with the results. Editing it while watching the score on
these sessions would be tuning on the test set. No examples from this corpus
are in the prompt (zero-shot).

No training, so there is nothing to cross-validate; the same probabilities are
written into every fold's oof.json so fusion can read them like other branches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time

import numpy as np
import torch

from audio_common import OOF, load_evaluation, summarise
from config import TEXT_EVAL_SETS, ensure_dirs, get_device, set_eval
from labels import EMOTIONS
from report import write_report
from text_common import load_sessions

SYSTEM = "You are an expert annotator of emotion in spoken language."
USER = ("The following is an automatic transcript of a person speaking freely about their day. "
        "It may contain transcription errors. Based on what they say, decide which emotion the "
        "speaker most likely felt at the time of recording.\n\n"
        'Transcript:\n"""{text}"""\n\n'
        "Choose exactly one of: angry, disgust, fear, happy, neutral, sad, surprise.\n"
        "Answer with the single word only.")
# IEMOCAP utterances are single turns of acted two-person dialogues, not free talk about a day.
# Same instructions otherwise. The in-house prompt is unchanged, so its SHA-1 and caches still hold.
USER_IEMOCAP = ("The following is an automatic transcript of one turn spoken by an actor in a two-person "
                "conversation. It may contain transcription errors. Based on what they say, decide which "
                "emotion the speaker most likely expressed in this turn.\n\n"
                'Transcript:\n"""{text}"""\n\n'
                "Choose exactly one of: angry, disgust, fear, happy, neutral, sad, surprise.\n"
                "Answer with the single word only.")


def prompt_for(eval_set: str) -> tuple[str, str]:
    user = USER_IEMOCAP if eval_set == "iemocap" else USER
    return user, hashlib.sha1((SYSTEM + "\n" + user).encode("utf-8")).hexdigest()


PROMPT_SHA1 = prompt_for("own")[1]


def load_llm(name, device, load_4bit):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    kw = {}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        kw["device_map"] = "auto"
    else:
        kw["dtype"] = torch.float16 if device == "cuda" else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    except TypeError:                                            # older transformers
        kw["torch_dtype"] = kw.pop("dtype", None)
        model = AutoModelForCausalLM.from_pretrained(name, **kw)
    if not load_4bit:
        model = model.to(device)
    return tok, model.eval()


def label_probs(tok, model, text, device, user: str = USER) -> np.ndarray:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user.format(text=text)}]
    return score_labels(tok, model, msgs, device)


@torch.no_grad()
def score_labels(tok, model, msgs, device) -> np.ndarray:
    """log P(label | prompt) for the 7 labels in one padded batch -> softmax. Used by fusion too."""
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    seqs = [p_ids + tok(lab, add_special_tokens=False)["input_ids"] for lab in EMOTIONS]
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):                                 # right padding
        ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
    dev = model.device if hasattr(model, "device") else device
    logp = torch.log_softmax(model(input_ids=ids.to(dev), attention_mask=att.to(dev)).logits.float(), -1)
    scores = []
    for i, s in enumerate(seqs):
        pos = range(len(p_ids), len(s))                          # label tokens
        scores.append(sum(logp[i, t - 1, s[t]].item() for t in pos))
    sc = np.array(scores)
    e = np.exp(sc - sc.max())
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser(description="LLM zero-shot text baseline")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--load_4bit", action="store_true", help="for 7B on an 8 GB GPU (pip install bitsandbytes)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval", default="own", choices=TEXT_EVAL_SETS)
    a = ap.parse_args()
    set_eval(a.eval)
    USER_T, PROMPT_SHA1 = prompt_for(a.eval)

    dirs = ensure_dirs()
    device = get_device()
    sess = load_sessions()
    if a.limit:
        sess = sess[: a.limit]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", a.model)
    run = f"llm_{safe}"
    cache = dirs["text_result"] / run / "scores.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache.exists():
        for l in cache.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            if r["prompt_sha1"] == PROMPT_SHA1:
                done[r["key"]] = np.array(r["proba"])
    print(f"[INFO] {len(sess)} sessions | {len(done)} already scored | prompt SHA-1 {PROMPT_SHA1}")

    todo = [s for s in sess if s["key"] not in done]
    if todo:
        tok, model = load_llm(a.model, device, a.load_4bit)
        t0 = time.time()
        with open(cache, "a", encoding="utf-8") as f:
            for i, s in enumerate(todo, 1):
                p = label_probs(tok, model, s["text"], device, USER_T)
                done[s["key"]] = p
                f.write(json.dumps({"key": s["key"], "proba": p.round(6).tolist(), "prompt_sha1": PROMPT_SHA1}) + "\n")
                f.flush()
                if i % 20 == 0 or i == len(todo):
                    rate = i / max(time.time() - t0, 1e-6)
                    print(f"  [{i}/{len(todo)}] {rate:.2f} sessions/s  ETA {(len(todo) - i) / max(rate, 1e-6) / 60:.1f} min")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    if a.limit:
        print("[INFO] --limit run: scores cached; run without --limit to write results.")
        return

    ev, splits = load_evaluation()
    keys = [s["key"] for s in sess]
    pos = {k: i for i, k in enumerate(keys)}
    P = np.stack([done[k] for k in keys])
    y = np.array([s["y7"] for s in sess])
    spk = np.array([s["user_id"] for s in sess])
    for seed, protos in splits.items():
        for proto, folds in protos.items():
            oof = OOF(keys)
            for fi, fold in enumerate(folds):
                te = [pos[k] for k in fold["test"]]; va = [pos[k] for k in fold["val"]]
                oof.add(fi, fold["name"], fold["test"], P[te], fold["val"], P[va])
            exp = dirs["text_result"] / run / proto / f"seed{seed}"
            exp.mkdir(parents=True, exist_ok=True)
            oof.save(exp / "oof.json", y)
    exp = dirs["text_result"] / run / "report"
    write_report(exp, y, P.argmax(1), P, spk, keys, title=f"text LLM zero-shot {a.model}",
                 extra_cols={"transcript": [s["text"] for s in sess]},
                 config={"model": a.model, "prompt_sha1": PROMPT_SHA1, "system": SYSTEM, "user": USER_T, "eval": a.eval,
                         "load_4bit": a.load_4bit})
    m = summarise(y, P.argmax(1))
    print(f"[INFO] acc7 {m['acc7']:.3f} F1_7 {m['f1_7']:.3f} acc3 {m['acc3']:.3f} F1_3 {m['f1_3']:.3f}")


if __name__ == "__main__":
    main()
