"""
09_extract_embeddings.py
═══════════════════════════════════════════════════════════════════════════════
Run the encoder trained in 08 over a folder of clips and merge its output into
features.jsonl, next to the emotion2vec fields, so 04 / 06 / 10 / 11 can switch
between encoders with a flag instead of a rewrite.

    # your own corpus
    python 09_extract_embeddings.py \
        --encoder encoders/wavlm_bp_ft/encoder.pt \
        --audio_dir data/PreprocessedAudio \
        --features data/features.jsonl

    # the public corpus (train + holdout, for 10_benchmark_public.py)
    python 09_extract_embeddings.py \
        --encoder encoders/wavlm_bp_ft/encoder.pt \
        --audio_dir data_public/PreprocessedAudio \
        --features data_public/features.jsonl

FIELDS ADDED PER CLIP
─────────────────────────────────────────────────────────────────────────────
  custom_embedding                 emb_dim floats, the projection layer output
  custom_probs7 / custom_probs3    head outputs (softmax)
  custom_label                     argmax of custom_probs7
  custom_encoder                   checkpoint path, so a file can never be
                                   silently mixed across encoder versions

AND PER SESSION (on the freestyle clip)
  custom_embedding_delta           freestyle − passage, same session
  custom_embedding_cosine_to_passage

LONG CLIPS
─────────────────────────────────────────────────────────────────────────────
Your freestyle recordings run far longer than the 4 s training crop. A single
centre crop would throw most of the recording away, so inference slides a
window with 50% overlap and averages embeddings and probabilities across
windows. Averaging in probability space, not logit space, keeps a single loud
window from dominating the utterance-level decision.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import DEVICE

TARGET_SR = 16_000


def load_encoder(ckpt_path: Path, device: str):
    import importlib
    mod = importlib.import_module("08_train_encoder")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    # Rebuild with the SAME spec_aug the checkpoint was created with, or the
    # backbone's masking parameter is absent and load_state_dict fails.
    # eval() disables masking regardless.
    model = mod.EmotionEncoder(
        cfg["backbone"], emb_dim=cfg["emb_dim"], hc_dim=cfg.get("hc_dim", 0),
        spec_aug=cfg.get("spec_aug", 0.05))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    print(f"[INFO] Encoder: {cfg['backbone']}  emb_dim={cfg['emb_dim']}  "
          f"hc_dim={cfg.get('hc_dim', 0)}  (val macro F1 {ck.get('val_macro_f1', float('nan')):.4f})")
    return model, cfg, ck


def windows(y: np.ndarray, win: int, hop: int, max_windows: int = 24):
    """Return [(chunk, n_valid_samples)], padding only the final short chunk."""
    if len(y) <= win:
        out = np.zeros(win, dtype=np.float32)
        out[:len(y)] = y
        return [(out, len(y))]
    starts = list(range(0, len(y) - win + 1, hop))
    if len(starts) > max_windows:                 # uniform subsample, keep the ends
        idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[i] for i in idx]
    return [(y[s:s + win], win) for s in starts]


@torch.no_grad()
def encode_clip(model, y: np.ndarray, cfg: dict, device: str,
                hc_vec: np.ndarray | None, batch: int = 8):
    win = int(cfg.get("eval_crop", 6.0) * TARGET_SR)
    chunks = windows(y, win, win // 2)
    embs, p7s, p3s = [], [], []

    for i in range(0, len(chunks), batch):
        blk = chunks[i:i + batch]
        wav = torch.from_numpy(np.stack([c for c, _ in blk])).to(device)
        n_valid = torch.tensor([n for _, n in blk], dtype=torch.long)
        hc = None
        if cfg.get("hc_dim", 0) and hc_vec is not None:
            hc = torch.from_numpy(np.tile(hc_vec, (len(blk), 1))).to(device)
        o7, o3, e = model(wav, n_valid, hc)
        embs.append(e.cpu().numpy())
        p7s.append(F.softmax(o7, dim=1).cpu().numpy())
        p3s.append(F.softmax(o3, dim=1).cpu().numpy())

    return (np.concatenate(embs).mean(0),
            np.concatenate(p7s).mean(0),
            np.concatenate(p3s).mean(0))


def main():
    p = argparse.ArgumentParser(description="Apply the trained encoder to clips")
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--audio_dir", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True,
                   help="features.jsonl from 03; merged in place unless --out")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip_existing", action="store_true",
                   help="Leave clips that already have custom_embedding alone")
    args = p.parse_args()

    device = DEVICE
    model, cfg, ck = load_encoder(args.encoder, device)
    labels = ck.get("labels", [])

    records: dict[str, dict] = {}
    order: list[str] = []
    with open(args.features) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            stem = r.get("stem") or Path(r["file"]).stem
            records[stem] = r
            order.append(stem)
    print(f"[INFO] {len(records)} feature records")

    hc_keys = cfg.get("hc_keys") or []
    hc_mu = np.array(cfg["hc_mu"], dtype=np.float32) if cfg.get("hc_mu") else None
    hc_sd = np.array(cfg["hc_sd"], dtype=np.float32) if cfg.get("hc_sd") else None

    import soundfile as sf
    stems = order[:args.limit] if args.limit else order
    t0, done, missing = time.time(), 0, 0

    for i, stem in enumerate(stems, 1):
        rec = records[stem]
        if args.skip_existing and rec.get("custom_embedding"):
            continue
        wav_path = args.audio_dir / f"{stem}.wav"
        if not wav_path.exists():
            missing += 1
            continue
        try:
            y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        except Exception as exc:
            print(f"  [WARN] {stem}: {exc}")
            missing += 1
            continue
        if y.ndim > 1:
            y = y.mean(axis=1)
        if sr != TARGET_SR:
            import librosa
            y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)

        hc_vec = None
        if hc_keys:
            feats = rec.get("handcrafted", {}).get("features", {})
            hc_vec = np.array([float(feats.get(k, 0.0)) for k in hc_keys], dtype=np.float32)
            if hc_mu is not None:
                hc_vec = (hc_vec - hc_mu) / hc_sd
            hc_vec = np.nan_to_num(hc_vec).astype(np.float32)

        emb, p7, p3 = encode_clip(model, np.nan_to_num(y.astype(np.float32)),
                                  cfg, device, hc_vec, batch=args.batch)

        rec["custom_embedding"] = [round(float(v), 6) for v in emb]
        rec["custom_probs7"] = [round(float(v), 6) for v in p7]
        rec["custom_probs3"] = [round(float(v), 6) for v in p3]
        rec["custom_label"] = labels[int(np.argmax(p7))] if labels else int(np.argmax(p7))
        rec["custom_encoder"] = str(args.encoder)
        done += 1

        if i % 50 == 0 or i == len(stems):
            rate = i / max(time.time() - t0, 1e-6)
            print(f"  [{i:5d}/{len(stems)}] {rate:.1f} clips/s  "
                  f"eta {(len(stems) - i) / max(rate, 1e-6) / 60:.1f} min", flush=True)

    if missing:
        print(f"[WARN] {missing} clip(s) had no wav in {args.audio_dir}")

    # ── Session-level delta, mirroring the emotion2vec delta in 03 ───────────
    by_session: dict[tuple, dict] = defaultdict(dict)
    for stem, rec in records.items():
        if "custom_embedding" not in rec:
            continue
        by_session[(rec.get("user_id", ""), rec.get("session_id", ""))][
            rec.get("video_type", "")] = rec

    n_delta = 0
    for pair in by_session.values():
        fs, pa = pair.get("freestyle"), pair.get("passage")
        if fs is None or pa is None:
            continue
        f_emb = np.array(fs["custom_embedding"], dtype=np.float32)
        p_emb = np.array(pa["custom_embedding"], dtype=np.float32)
        if f_emb.shape != p_emb.shape:
            continue
        fs["custom_embedding_delta"] = [round(float(v), 6) for v in (f_emb - p_emb)]
        denom = float(np.linalg.norm(f_emb) * np.linalg.norm(p_emb))
        fs["custom_embedding_cosine_to_passage"] = (
            round(float(np.dot(f_emb, p_emb) / denom), 4) if denom > 0 else 0.0)
        n_delta += 1
    print(f"[INFO] custom delta computed for {n_delta}/{len(by_session)} sessions")

    out_path = args.out or args.features
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for stem in order:
            f.write(json.dumps(records[stem], ensure_ascii=False) + "\n")
    shutil.move(str(tmp), str(out_path))

    print(f"\n[INFO] {done} clips encoded -> {out_path}")
    if labels:
        from collections import Counter
        c = Counter(r.get("custom_label") for r in records.values() if r.get("custom_label"))
        print("[INFO] Zero-shot label distribution:", dict(c))


if __name__ == "__main__":
    main()
