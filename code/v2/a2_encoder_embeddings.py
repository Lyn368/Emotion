"""
a2_encoder_embeddings.py - Frozen encoder embeddings, per 4-second segment.

Each clip (ENCODER audio version) is cut into 4 s windows with 2 s hop and every
window goes through the encoder:

  wavlm_base_plus        microsoft/wavlm-base-plus via transformers, GPU.
                         ALL 13 hidden layers are kept (mean over time), so the
                         classifier learns which layers carry emotion; the last
                         layer alone is a poor choice for WavLM.
  emotion2vec_base       iic/emotion2vec_base via funasr (one 768-d vector per window)
  emotion2vec_plus_large iic/emotion2vec_plus_large via funasr (1024-d)

Base models are the size-matched comparison (~95M parameters each); large is optional.

Output: preprocessing/embeddings/<model>/<dataset>.npy   float16 (segments, layers, dim)
        preprocessing/embeddings/<model>/<dataset>_index.jsonl   clip -> rows

    python a2_encoder_embeddings.py --model wavlm_base_plus
    python a2_encoder_embeddings.py --model emotion2vec_base
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from audio_common import abs_path, load_records, load_wav, windows
from config import ensure_dirs, get_device

MODELS = {
    "wavlm_base_plus": ("hf", "microsoft/wavlm-base-plus"),
    "wavlm_large": ("hf", "microsoft/wavlm-large"),
    "emotion2vec_base": ("funasr", "iic/emotion2vec_base"),
    "emotion2vec_plus_large": ("funasr", "iic/emotion2vec_plus_large"),
}


class HFEncoder:
    """transformers speech encoder returning every hidden layer, mean-pooled over time."""

    def __init__(self, model_id, device):
        from transformers import AutoFeatureExtractor, AutoModel
        self.fe = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, segs: list[np.ndarray]) -> np.ndarray:
        inp = self.fe(segs, sampling_rate=16000, return_tensors="pt", padding=True,
                      return_attention_mask=True)
        x = inp["input_values"].to(self.device)
        am = inp.get("attention_mask")
        with torch.autocast("cuda", enabled=self.device == "cuda"):
            out = self.model(x, attention_mask=am.to(self.device) if am is not None else None,
                             output_hidden_states=True)
        hs = torch.stack(out.hidden_states, 1).float()               # (N, L, T, D)
        if am is not None:                                             # mask padded frames
            fm = self.model._get_feature_vector_attention_mask(hs.shape[2], am.to(self.device))
            fm = fm[:, None, :, None].float()
            return ((hs * fm).sum(2) / fm.sum(2).clamp(min=1)).cpu().numpy()
        return hs.mean(2).cpu().numpy()


class FunasrEncoder:
    """emotion2vec through funasr: one utterance-level vector per window."""

    def __init__(self, model_id, device):
        from funasr import AutoModel
        self.model = AutoModel(model=model_id, device=device, disable_update=True)

        self.use_array = True

    def _one(self, s: np.ndarray):
        if self.use_array:
            try:
                return self.model.generate(s, output_dir=None, granularity="utterance",
                                           extract_embedding=True)[0]
            except Exception:
                self.use_array = False          # some funasr versions only accept file paths
        import tempfile, os
        import soundfile as sf
        fd, tmp = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        try:
            sf.write(tmp, s, 16000)
            return self.model.generate(tmp, output_dir=None, granularity="utterance",
                                       extract_embedding=True)[0]
        finally:
            os.remove(tmp)

    def __call__(self, segs: list[np.ndarray]) -> np.ndarray:
        out = [np.asarray(self._one(s)["feats"], dtype=np.float32).reshape(1, -1) for s in segs]
        return np.stack(out)                                           # (N, 1, D)


def main():
    ap = argparse.ArgumentParser(description="Frozen encoder embeddings")
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--model_path", default=None, help="local folder instead of downloading")
    ap.add_argument("--dataset", default="own")
    ap.add_argument("--batch", type=int, default=16, help="segments per GPU batch")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    kind, model_id = MODELS[args.model]
    model_id = args.model_path or model_id
    device = get_device()
    name = {"own": "own_dataset.json"}.get(args.dataset, f"{args.dataset}.json")
    recs = list(load_records(name).values())
    recs = [r for r in recs if abs_path(r.get("encoder_path")) and abs_path(r["encoder_path"]).exists()]
    if args.limit:
        recs = recs[: args.limit]

    out_dir = ensure_dirs()["embeddings"] / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    enc = HFEncoder(model_id, device) if kind == "hf" else FunasrEncoder(model_id, device)
    print(f"[INFO] {args.model} loaded ({model_id}) | {len(recs)} clips")

    # pass 1: plan the rows so the output array can be written directly (no huge RAM lists)
    plan = []
    for r in recs:
        n = len(load_wav(abs_path(r["encoder_path"])))
        plan.append((r["clip_id"], abs_path(r["encoder_path"]), windows(n)))
    total = sum(len(s) for _, _, s in plan)

    probe = enc([np.zeros(16000, np.float32)])
    L, D = probe.shape[1], probe.shape[2]
    arr = np.lib.format.open_memmap(out_dir / f"{args.dataset}.npy", mode="w+",
                                    dtype=np.float16, shape=(total, L, D))
    print(f"[INFO] {total} segments x {L} layers x {D} dims "
          f"({total * L * D * 2 / 1e9:.2f} GB on disk)")

    index, row, t0 = [], 0, time.time()
    for i, (cid, path, spans) in enumerate(plan, 1):
        y = load_wav(path)
        segs = [y[a:b] for a, b in spans]
        for j in range(0, len(segs), args.batch):
            e = enc(segs[j:j + args.batch])
            arr[row:row + len(e)] = e.astype(np.float16)
            row += len(e)
        index.append({"clip_id": cid, "start": row - len(segs), "n": len(segs)})
        if i % 25 == 0 or i == len(plan):
            rate = i / max(time.time() - t0, 1e-6)
            print(f"  [{i}/{len(plan)}] {rate:.2f} clips/s  ETA {(len(plan) - i) / max(rate, 1e-6) / 60:.1f} min")
    arr.flush()
    with open(out_dir / f"{args.dataset}_index.jsonl", "w", encoding="utf-8") as f:
        for r in index:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "source": model_id, "layers": L, "dim": D,
                   "window_s": 4.0, "hop_s": 2.0}, f, indent=2)
    print(f"[INFO] -> {out_dir}")


if __name__ == "__main__":
    main()
