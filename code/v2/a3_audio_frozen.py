"""
a3_audio_frozen.py - Audio branch with FROZEN encoders (fast; runs every configuration).

Reads the shared folds from preprocessing/splits.json, trains the siamese
SessionModel on precomputed embeddings (a2) and/or acoustic features (a1), and
writes one result folder per configuration, protocol and seed.

Configurations (combine with --configs):
  acoustic                 eGeMAPS + pauses, [freestyle, freestyle-passage, ratios]
  <enc>                    encoder, freestyle only (no passage reference)
  <enc>_delta              encoder, [f, f - p]  (the within-session reference)
  acoustic+<enc>_delta     both
where <enc> is wavlm_base_plus, emotion2vec_base, ...

    python a3_audio_frozen.py --encoders wavlm_base_plus,emotion2vec_base --protocols grouped,louo
    python a3_audio_frozen.py --final --configs acoustic+wavlm_base_plus_delta

Output: audio_result/frozen/<config>/<protocol>/seed<k>/  (report + oof.json)
        audio_result/frozen/summary_<protocol>.csv       (all configs, mean over seeds)
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audio_common import (N7, OOF, SessionModel, class_weights, load_evaluation, macro_f1,
                          read_json, summarise)
from config import EVAL_SETS, ensure_dirs, eval_paths, get_device, seed_everything, set_eval
from report import write_report
from stats import paired_bootstrap_test

EPOCHS, PATIENCE, BATCH, LR, WD = 80, 12, 16, 1e-3, 1e-2


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
class EmbeddingStore:
    """Segment embeddings of one encoder, memory-mapped from a2's .npy."""

    def __init__(self, model: str, dataset: str = "own"):
        d = ensure_dirs()["embeddings"] / model
        if not (d / f"{dataset}.npy").exists():
            raise SystemExit(f"[ERROR] No {dataset} embeddings for {model}. "
                             f"Run: python a2_encoder_embeddings.py --model {model} --dataset {dataset}")
        self.arr = np.load(d / f"{dataset}.npy", mmap_mode="r")
        self.rows = {}
        for l in (d / f"{dataset}_index.jsonl").read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                self.rows[r["clip_id"]] = (r["start"], r["n"])
        self.L, self.D = self.arr.shape[1], self.arr.shape[2]

    def get(self, clip_id):
        if clip_id not in self.rows:
            return None
        s, n = self.rows[clip_id]
        return torch.from_numpy(np.array(self.arr[s:s + n], dtype=np.float16))   # copy: memmap is read-only


def acoustic_matrix(ev, feats: dict) -> tuple[np.ndarray, list[str]]:
    """Session vector [freestyle, freestyle - passage, ratios]. Never the passage alone."""
    names = sorted(next(iter(feats.values())).keys())
    rows = []
    for s in ev:
        f = np.array([feats.get(s["freestyle_clip"], {}).get(n, 0.0) for n in names], float)
        if s.get("passage_clip") in feats:
            p = np.array([feats[s["passage_clip"]].get(n, 0.0) for n in names], float)
            ratio = [f[names.index(k)] / p[names.index(k)] if abs(p[names.index(k)]) > 1e-6 else 1.0
                     for k in ("F0semitoneFrom27.5Hz_sma3nz_amean", "loudness_sma3_amean",
                               "speaking_rate_wps") if k in names]
            rows.append(np.concatenate([f, f - p, ratio]))
        else:
            n_ratio = sum(k in names for k in ("F0semitoneFrom27.5Hz_sma3nz_amean",
                                                "loudness_sma3_amean", "speaking_rate_wps"))
            rows.append(np.concatenate([f, np.zeros_like(f), np.ones(n_ratio)]))
    X = np.nan_to_num(np.stack(rows)).astype(np.float32)
    return X, names


def collate(items, store, ac, idx):
    """Pad segment tensors of a batch of sessions."""
    B = len(idx)
    out = {}
    if store is not None:
        fr = [items[i][0] for i in idx]
        pa = [items[i][1] for i in idx]
        S = max(x.shape[0] for x in fr)
        P = max((x.shape[0] for x in pa if x is not None), default=1)
        free = torch.zeros(B, S, store.L, store.D, dtype=torch.float16)
        fmask = torch.zeros(B, S, dtype=torch.bool)
        pas = torch.zeros(B, P, store.L, store.D, dtype=torch.float16)
        pmask = torch.zeros(B, P, dtype=torch.bool)
        has = torch.zeros(B, dtype=torch.bool)
        for b in range(B):
            free[b, :fr[b].shape[0]] = fr[b]; fmask[b, :fr[b].shape[0]] = True
            if pa[b] is not None:
                pas[b, :pa[b].shape[0]] = pa[b]; pmask[b, :pa[b].shape[0]] = True; has[b] = True
        out.update(free=free, free_mask=fmask, pas=pas, pas_mask=pmask, has_pas=has)
    if ac is not None:
        out["ac"] = torch.from_numpy(ac[idx])
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def fit(model, items, store, ac, y, tr, va, device, seed):
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    w = class_weights(y[tr]).to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best, best_state, bad, best_ep = -1.0, None, 0, 0
    rng = np.random.RandomState(seed)
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            if len(b) < 2:
                continue
            batch = {k: v.to(device) for k, v in collate(items, store, ac, b).items()}
            with torch.autocast("cuda", enabled=device == "cuda"):
                loss = F.cross_entropy(model(**batch).float(), torch.from_numpy(y[b]).to(device), weight=w)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        score = macro_f1(y[va], predict(model, items, store, ac, va, device).argmax(1))
        if score > best:
            best, bad, best_ep = score, 0, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best, best_ep


@torch.no_grad()
def predict(model, items, store, ac, idx, device) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(idx), 32):
        b = idx[i:i + 32]
        batch = {k: v.to(device) for k, v in collate(items, store, ac, b).items()}
        with torch.autocast("cuda", enabled=device == "cuda"):
            out.append(F.softmax(model(**batch).float(), -1).cpu().numpy())
    return np.concatenate(out)


def standardise(ac, tr):
    if ac is None:
        return None
    mu, sd = ac[tr].mean(0), ac[tr].std(0)
    return ((ac - mu) / np.where(sd > 1e-8, sd, 1.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_config(cfg: str):
    use_ac = cfg.startswith("acoustic")
    enc = cfg.split("+", 1)[1] if "+" in cfg else (None if cfg == "acoustic" else cfg)
    delta = bool(enc and enc.endswith("_delta"))
    enc = enc[:-6] if delta else enc
    return use_ac, enc, delta


def main():
    ap = argparse.ArgumentParser(description="Audio branch, frozen encoders")
    ap.add_argument("--encoders", default="wavlm_base_plus,emotion2vec_base")
    ap.add_argument("--configs", default=None, help="explicit list; default = every config for --encoders")
    ap.add_argument("--protocols", default="grouped,louo")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--final", action="store_true", help="train one model on all sessions and save it")
    ap.add_argument("--eval", default="own", choices=EVAL_SETS,
                    help="dataset to train and test on (public ones need p6_public_sessions.py)")
    args = ap.parse_args()
    set_eval(args.eval)

    dirs = ensure_dirs()
    print(f"[INFO] evaluation set: {args.eval} | results -> {dirs['audio_result']}")
    device = get_device()
    ev, splits = load_evaluation()
    keys = [s["key"] for s in ev]
    y = np.array([s["y7"] for s in ev])
    speakers = [s["speaker_id"] for s in ev]
    pos = {k: i for i, k in enumerate(keys)}

    encs = [e for e in args.encoders.split(",") if e]
    configs = args.configs.split(",") if args.configs else \
        ["acoustic"] + [c for e in encs for c in (e, f"{e}_delta", f"acoustic+{e}_delta")]

    feat_path = eval_paths()["acoustic"]
    ac_all = None
    if any(parse_config(c)[0] for c in configs):
        if not feat_path.exists():
            raise SystemExit(f"[ERROR] {feat_path.name} missing. Run: python a1_acoustic_features.py --dataset {args.eval}")
        ac_all, ac_names = acoustic_matrix(ev, read_json(feat_path))
        print(f"[INFO] acoustic session vector: {ac_all.shape[1]} values")

    stores = {}
    results = {}                                                  # (config, protocol, seed) -> pred
    for cfg in configs:
        use_ac, enc, delta = parse_config(cfg)
        store = None
        if enc:
            store = stores.get(enc) or stores.setdefault(enc, EmbeddingStore(enc, eval_paths()["embeddings"]))
            items = [(store.get(s["freestyle_clip"]), store.get(s.get("passage_clip")) if delta else None)
                     for s in ev]
            missing = [i for i, it in enumerate(items) if it[0] is None]
            if missing:
                raise SystemExit(f"[ERROR] {len(missing)} sessions have no {enc} embedding; re-run a2.")
        else:
            items = [(None, None)] * len(ev)

        def make():
            return SessionModel(n_layers=store.L if store else 1, dim=store.D if store else 1,
                                acoustic_dim=ac_all.shape[1] if use_ac else 0,
                                use_encoder=store is not None, use_delta=delta)

        if args.final:                                            # one model on every session
            rng = np.random.RandomState(42)
            spk = np.array(speakers); hold = set(rng.choice(np.unique(spk), max(1, len(set(spk)) // 5), False))
            va = np.array([i for i in range(len(ev)) if spk[i] in hold]); tr = np.setdiff1d(np.arange(len(ev)), va)
            ac = standardise(ac_all, tr) if use_ac else None
            model, score, ep = fit(make(), items, store, ac, y, tr, va, device, 42)
            out = dirs["final_models"] / "audio_frozen" / cfg.replace("+", "_")
            out.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out / "model.pt")
            with open(out / "config.json", "w", encoding="utf-8") as f:
                json.dump({"config": cfg, "encoder": enc, "delta": delta, "acoustic": use_ac,
                           "best_epoch": ep, "val_macro_f1": score,
                           "note": "Performance is the CV result, not measured on this model."}, f, indent=2)
            print(f"[FINAL] {cfg}: saved -> {out}")
            continue

        for proto in args.protocols.split(","):
            for seed in [int(s) for s in args.seeds.split(",")]:
                seed_everything(seed)
                t0 = time.time()
                oof = OOF(keys)
                for fi, fold in enumerate(splits[str(seed)][proto]):
                    tr = np.array([pos[k] for k in fold["train"]]); va = np.array([pos[k] for k in fold["val"]])
                    te = np.array([pos[k] for k in fold["test"]])
                    ac = standardise(ac_all, tr) if use_ac else None
                    model, _, _ = fit(make(), items, store, ac, y, tr, va, device, seed)
                    oof.add(fi, fold["name"], [keys[i] for i in te], predict(model, items, store, ac, te, device),
                            [keys[i] for i in va], predict(model, items, store, ac, va, device))
                exp = dirs["audio_result"] / "frozen" / cfg.replace("+", "_") / proto / f"seed{seed}"
                exp.mkdir(parents=True, exist_ok=True)
                tested = oof.save(exp / "oof.json", y)
                pred = oof.test.argmax(1)
                write_report(exp, y[tested], pred[tested], oof.test[tested],
                             np.array(speakers)[tested], [k for k, t in zip(keys, tested) if t],
                             title=f"audio {cfg} | {proto} | seed {seed}",
                             config={"config": cfg, "protocol": proto, "seed": seed, "encoder": enc,
                                     "delta": delta, "acoustic": use_ac, "frozen": True},
                             seed=seed, verbose=False)
                results[(cfg, proto, seed)] = pred
                m = summarise(y[tested], pred[tested])
                print(f"  {cfg:<34} {proto:<8} seed {seed}  acc7 {m['acc7']:.3f}  F1_7 {m['f1_7']:.3f}  "
                      f"acc3 {m['acc3']:.3f}  F1_3 {m['f1_3']:.3f}  ({time.time() - t0:.0f}s)")

    if args.final or not results:
        return

    # summary per protocol: mean over seeds + paired test of each config against the best
    for proto in args.protocols.split(","):
        rows = []
        for cfg in configs:
            ms = [summarise(y, results[(cfg, proto, s)]) for s in [int(x) for x in args.seeds.split(",")]
                  if (cfg, proto, s) in results]
            if ms:
                rows.append({"config": cfg, **{k: float(np.mean([m[k] for m in ms])) for k in ms[0]},
                             "f1_7_std": float(np.std([m["f1_7"] for m in ms]))})
        rows.sort(key=lambda r: -r["f1_7"])
        seed0 = int(args.seeds.split(",")[0])
        best = rows[0]["config"]
        print(f"\n== {proto}: mean over seeds, sorted by macro F1 (7 emotions)")
        print(f"   {'config':<34}{'acc7':>7}{'F1_7':>7}{'acc3':>7}{'F1_3':>7}   vs best (paired, seed {seed0})")
        for r in rows:
            txt = ""
            if r["config"] != best:
                t = paired_bootstrap_test(y, results[(best, proto, seed0)], results[(r["config"], proto, seed0)],
                                          np.array(speakers), "macro_f1", seed=seed0)
                txt = f"delta {t['delta']:+.3f} p={t['p_value']:.3f}{' *' if t['significant_05'] else ''}"
            print(f"   {r['config']:<34}{r['acc7']:>7.3f}{r['f1_7']:>7.3f}{r['acc3']:>7.3f}{r['f1_3']:>7.3f}   {txt}")
        with open(dirs["audio_result"] / "frozen" / f"summary_{proto}.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()