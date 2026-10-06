"""
a4_audio_finetune.py - Fine-tune WavLM (the encoder keeps learning).

Three modes:
  stage1   train on IEMOCAP (external data, 7 classes) and save the encoder.
  cv       siamese fine-tuning on the in-house sessions, over the shared folds.
  final    one model on every in-house session, saved for later use.

    python a4_audio_finetune.py stage1
    python a4_audio_finetune.py cv --unfreeze 2 --from_stage1 --protocols grouped
    python a4_audio_finetune.py cv --unfreeze 0      # ablation: nothing unfrozen
    python a4_audio_finetune.py final --unfreeze 2 --from_stage1

Design (fits an 8 GB GPU):
  - The convolutional front-end is always frozen; only the top --unfreeze
    transformer layers learn (0, 2, 4 or all). Lower layers hold generic
    acoustics; the top ones adapt fastest to the task.
  - Mixed precision + gradient checkpointing.
  - Each training step samples a few 4 s windows per clip instead of the whole
    recording; evaluation uses every window.
  - Freestyle and passage go through the SAME encoder and pooling (siamese), so
    the passage acts as the within-session neutral reference inside the network.

Output: audio_result/finetune/<run>/<protocol>/seed<k>/  and  models/stage1|final/...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from audio_common import (N7, OOF, SessionModel, abs_path, class_weights, load_evaluation, load_records,
                          load_wav, macro_f1, summarise, windows)
from config import SAMPLE_RATE, ensure_dirs, get_device, seed_everything
from labels import resolve
from report import write_report

WIN = int(4.0 * SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class WavEncoder(nn.Module):
    """WavLM returning every hidden layer, mean-pooled over valid frames: (N, L, D)."""

    def __init__(self, source: str, unfreeze: str):
        super().__init__()
        from transformers import AutoFeatureExtractor, AutoModel
        self.fe = AutoFeatureExtractor.from_pretrained(source)
        self.m = AutoModel.from_pretrained(source)
        self.m.config.mask_time_prob = 0.05                        # light SpecAugment while training
        self.m.config.layerdrop = 0.0                              # skipping layers would corrupt the learned layer weights
        for p in self.m.parameters():
            p.requires_grad = False
        layers = self.m.encoder.layers
        k = len(layers) if unfreeze == "all" else int(unfreeze)
        for layer in list(layers)[len(layers) - k:] if k else []:
            for p in layer.parameters():
                p.requires_grad = True
        if unfreeze == "all":                                      # everything except the CNN front-end
            for n, p in self.m.named_parameters():
                if not n.startswith("feature_extractor"):
                    p.requires_grad = True
        self.trainable = any(p.requires_grad for p in self.m.parameters())
        if self.trainable:
            self.m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.n_layers = self.m.config.num_hidden_layers + 1
        self.dim = self.m.config.hidden_size

    def prep(self, segs: list[np.ndarray], device):
        inp = self.fe(segs, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True,
                      return_attention_mask=True)
        return inp["input_values"].to(device), inp["attention_mask"].to(device)

    def forward(self, x, am):
        out = self.m(x, attention_mask=am, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 1)                     # (N, L, T, D)
        fm = self.m._get_feature_vector_attention_mask(hs.shape[2], am)[:, None, :, None].to(hs.dtype)
        return (hs * fm).sum(2) / fm.sum(2).clamp(min=1)


class FullModel(nn.Module):
    def __init__(self, enc: WavEncoder, use_delta: bool):
        super().__init__()
        self.enc = enc
        self.head = SessionModel(n_layers=enc.n_layers, dim=enc.dim, use_delta=use_delta)
        self.use_delta = use_delta

    def embed(self, clips: list[list[np.ndarray]], device, chunk=None):
        """clips: list of lists of segments -> padded (B, S, L, D) + mask."""
        flat = [s for c in clips for s in c]
        outs = []
        step = chunk or len(flat)
        for i in range(0, len(flat), step):
            x, am = self.enc.prep(flat[i:i + step], device)
            outs.append(self.enc(x, am))
        e = torch.cat(outs)
        S = max(len(c) for c in clips)
        B = len(clips)
        pad = torch.zeros(B, S, e.shape[1], e.shape[2], device=device, dtype=e.dtype)
        mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        j = 0
        for b, c in enumerate(clips):
            pad[b, :len(c)] = e[j:j + len(c)]; mask[b, :len(c)] = True; j += len(c)
        return pad, mask

    def forward(self, free_clips, pas_clips, device, chunk=None):
        free, fm = self.embed(free_clips, device, chunk)
        if self.use_delta:
            has = torch.tensor([len(p) > 0 for p in pas_clips], device=device)
            pas_in = [p if p else [np.zeros(WIN, np.float32)] for p in pas_clips]
            pas, pm = self.embed(pas_in, device, chunk)
            return self.head(free=free, free_mask=fm, pas=pas, pas_mask=pm, has_pas=has)
        return self.head(free=free, free_mask=fm)


def optimiser(model: FullModel, lr_enc: float, lr_head: float):
    enc = [p for p in model.enc.parameters() if p.requires_grad]
    groups = [{"params": list(model.head.parameters()), "lr": lr_head}]
    if enc:
        groups.append({"params": enc, "lr": lr_enc})
    return torch.optim.AdamW(groups, weight_decay=1e-2)


# ---------------------------------------------------------------------------
# Audio sampling
# ---------------------------------------------------------------------------
class AudioBank:
    """Encoder-version audio of every needed clip, loaded once into RAM."""

    def __init__(self, recs: dict, clip_ids):
        self.y = {}
        for c in clip_ids:
            if c and c in recs and abs_path(recs[c].get("encoder_path")):
                self.y[c] = load_wav(abs_path(recs[c]["encoder_path"]))

    def random_segments(self, cid, k, rng) -> list[np.ndarray]:
        if not cid or cid not in self.y:
            return []
        y = self.y[cid]
        if len(y) <= WIN:
            return [y]
        starts = rng.randint(0, len(y) - WIN, size=k)
        return [y[s:s + WIN] for s in starts]

    def all_segments(self, cid) -> list[np.ndarray]:
        if not cid or cid not in self.y:
            return []
        y = self.y[cid]
        return [y[a:b] for a, b in windows(len(y))]


@torch.no_grad()
def predict(model, bank, sessions, device, chunk) -> np.ndarray:
    model.eval()
    out = []
    for s in sessions:
        with torch.autocast("cuda", enabled=device == "cuda"):
            logits = model([bank.all_segments(s["freestyle_clip"])],
                           [bank.all_segments(s.get("passage_clip"))], device, chunk)
        out.append(F.softmax(logits.float(), -1).cpu().numpy()[0])
    return np.stack(out)


def train(model, bank, sess, y, tr, va, device, a, seed, log=""):
    """Train with sampled windows; early stopping on validation macro F1 (session level)."""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    model.to(device)
    opt = optimiser(model, a.lr_encoder, a.lr_head)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    w = class_weights(y[tr]).to(device)
    best, state, bad, best_ep = -1.0, None, 0, 0
    for ep in range(a.epochs):
        model.train()
        t0, order = time.time(), rng.permutation(tr)
        for i in range(0, len(order), a.sessions_per_step):
            b = order[i:i + a.sessions_per_step]
            free = [bank.random_segments(sess[j]["freestyle_clip"], a.free_segs, rng) for j in b]
            pas = [bank.random_segments(sess[j].get("passage_clip"), a.pas_segs, rng) for j in b]
            with torch.autocast("cuda", enabled=device == "cuda"):
                logits = model(free, pas, device)
                loss = F.cross_entropy(logits.float(), torch.from_numpy(y[b]).to(device), weight=w)
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        score = macro_f1(y[va], predict(model, bank, [sess[j] for j in va], device, a.chunk).argmax(1))
        print(f"    {log} epoch {ep + 1}/{a.epochs}  val macro F1 {score:.3f}  ({time.time() - t0:.0f}s)")
        if score > best:
            best, bad, best_ep = score, 0, ep + 1
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(state)
    return model, best, best_ep


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def run_stage1(a, device, dirs):
    """IEMOCAP utterances as single-clip 'sessions'; Session 5 speakers are the validation set."""
    recs = load_records("iemocap.json")
    items = []
    for r in recs.values():
        lab = resolve("iemocap", r.get("label_raw"))
        if lab.get("y7") is not None:
            items.append({"freestyle_clip": r["clip_id"], "passage_clip": None,
                          "speaker": r["speaker_id"], "y7": lab["y7"]})
    if a.limit:
        items = items[: a.limit]
    y = np.array([s["y7"] for s in items])
    va = np.array([i for i, s in enumerate(items) if s["speaker"].startswith("Session5")])
    if len(va) == 0:                                   # small test sets may lack Session 5
        va = np.arange(len(items))[::5]
    tr = np.setdiff1d(np.arange(len(items)), va)
    print(f"[STAGE1] {len(tr)} train / {len(va)} val utterances | classes {np.bincount(y, minlength=N7).tolist()}")
    bank = AudioBank(recs, [s["freestyle_clip"] for s in items])
    model = FullModel(WavEncoder(a.source, a.unfreeze), use_delta=False)
    model, best, ep = train(model, bank, items, y, tr, va, device, a, 42, log="stage1")
    out = dirs["models"] / "stage1" / f"{a.name}_iemocap"
    out.mkdir(parents=True, exist_ok=True)
    model.enc.m.save_pretrained(out); model.enc.fe.save_pretrained(out)
    with open(out / "stage1_info.json", "w", encoding="utf-8") as f:
        json.dump({"source": a.source, "unfreeze": a.unfreeze, "val_macro_f1": best, "best_epoch": ep,
                   "note": "IEMOCAP only; no in-house data, so any CV fold can start from here."}, f, indent=2)
    print(f"[STAGE1] val macro F1 {best:.3f} -> {out}")


def encoder_source(a, dirs) -> str:
    ck = dirs["models"] / "stage1" / f"{a.name}_iemocap"
    if a.from_stage1:
        if not ck.exists():
            raise SystemExit(f"[ERROR] {ck} not found. Run: python a4_audio_finetune.py stage1")
        return str(ck)
    return a.source


def run_cv(a, device, dirs):
    ev, splits = load_evaluation()
    recs = load_records("own_dataset.json")
    keys = [s["key"] for s in ev]
    pos = {k: i for i, k in enumerate(keys)}
    y = np.array([s["y7"] for s in ev])
    spk = np.array([s["speaker_id"] for s in ev])
    bank = AudioBank(recs, [c for s in ev for c in (s["freestyle_clip"], s.get("passage_clip"))])
    run = f"{a.name}_unfreeze{a.unfreeze}{'_stage1' if a.from_stage1 else ''}{'' if not a.no_delta else '_nodelta'}"
    src = encoder_source(a, dirs)
    for proto in a.protocols.split(","):
        for seed in [int(s) for s in a.seeds.split(",")]:
            seed_everything(seed)
            oof = OOF(keys)
            for fi, fold in enumerate(splits[str(seed)][proto]):
                tr = np.array([pos[k] for k in fold["train"]]); va = np.array([pos[k] for k in fold["val"]])
                te = np.array([pos[k] for k in fold["test"]])
                model = FullModel(WavEncoder(src, a.unfreeze), use_delta=not a.no_delta)
                model, _, _ = train(model, bank, ev, y, tr, va, device, a, seed, log=f"{proto} {fold['name']}")
                oof.add(fi, fold["name"], [keys[i] for i in te], predict(model, bank, [ev[i] for i in te], device, a.chunk),
                        [keys[i] for i in va], predict(model, bank, [ev[i] for i in va], device, a.chunk))
                del model
                if device == "cuda":
                    torch.cuda.empty_cache()
            exp = dirs["audio_result"] / "finetune" / run / proto / f"seed{seed}"
            exp.mkdir(parents=True, exist_ok=True)
            done = oof.save(exp / "oof.json", y)
            pred = oof.test.argmax(1)
            write_report(exp, y[done], pred[done], oof.test[done], spk[done],
                         [k for k, d in zip(keys, done) if d], title=f"audio {run} | {proto} | seed {seed}",
                         config=vars(a) | {"protocol": proto, "seed": seed, "encoder_source": src}, seed=seed)
            m = summarise(y[done], pred[done])
            print(f"[CV] {run} {proto} seed {seed}: acc7 {m['acc7']:.3f} F1_7 {m['f1_7']:.3f} "
                  f"acc3 {m['acc3']:.3f} F1_3 {m['f1_3']:.3f}")


def run_final(a, device, dirs):
    ev, _ = load_evaluation()
    recs = load_records("own_dataset.json")
    y = np.array([s["y7"] for s in ev])
    spk = np.array([s["speaker_id"] for s in ev])
    rng = np.random.RandomState(42)
    hold = set(rng.choice(np.unique(spk), max(1, len(set(spk)) // 5), replace=False))
    va = np.array([i for i in range(len(ev)) if spk[i] in hold]); tr = np.setdiff1d(np.arange(len(ev)), va)
    bank = AudioBank(recs, [c for s in ev for c in (s["freestyle_clip"], s.get("passage_clip"))])
    model = FullModel(WavEncoder(encoder_source(a, dirs), a.unfreeze), use_delta=not a.no_delta)
    model, best, ep = train(model, bank, ev, y, tr, va, device, a, 42, log="final")
    out = dirs["models"] / "final" / f"audio_{a.name}_unfreeze{a.unfreeze}"
    out.mkdir(parents=True, exist_ok=True)
    model.enc.m.save_pretrained(out / "encoder"); model.enc.fe.save_pretrained(out / "encoder")
    torch.save(model.head.state_dict(), out / "head.pt")
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(a) | {"best_epoch": ep, "val_macro_f1": best,
                             "note": "Report the CV result for this configuration; this model saw all sessions."},
                  f, indent=2)
    print(f"[FINAL] saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description="WavLM fine-tuning")
    ap.add_argument("mode", choices=["stage1", "cv", "final"])
    ap.add_argument("--source", default="microsoft/wavlm-base-plus", help="HF id or local folder")
    ap.add_argument("--name", default="wavlm_base_plus", help="short name used for folders")
    ap.add_argument("--unfreeze", default="2", help="0, 2, 4 or all")
    ap.add_argument("--from_stage1", action="store_true")
    ap.add_argument("--no_delta", action="store_true", help="ablation: no passage reference")
    ap.add_argument("--protocols", default="grouped")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--sessions_per_step", type=int, default=4)
    ap.add_argument("--free_segs", type=int, default=4, help="freestyle windows sampled per session per step")
    ap.add_argument("--pas_segs", type=int, default=2, help="passage windows sampled per session per step")
    ap.add_argument("--chunk", type=int, default=16, help="windows per forward pass at evaluation")
    ap.add_argument("--lr_encoder", type=float, default=2e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None, help="stage1 only: first N utterances")
    a = ap.parse_args()

    dirs = ensure_dirs()
    device = get_device()
    {"stage1": run_stage1, "cv": run_cv, "final": run_final}[a.mode](a, device, dirs)


if __name__ == "__main__":
    main()
