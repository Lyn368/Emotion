"""
config.py — Paths and device for the v2 pipeline.

Expected layout (Windows):

    C:\\Users\\2022g\\Downloads\\RAVDESS_IEMOCAP_CREMAD\\      <- PROJECT_ROOT
        code\\
            v1\\        (previous pipeline)
            v2\\        (this folder)
        our_audios\\     firebase_data\\     iemocap\\     ravdess\\     cremad\\

Every path can be overridden without editing code:
    set SER_ROOT=D:\\otra\\carpeta          (cmd)
    $env:SER_ROOT="D:\\otra\\carpeta"       (PowerShell)
or with --root on the scripts that read raw data (p1, p4). Later stages
read preprocessing/index.jsonl, which already stores full paths.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

V2_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = V2_DIR.parent.parent                   # code\v2 -> code -> project root

SAMPLE_RATE = 16_000
SEED = 42

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg", ".webm")


def project_root(override: str | Path | None = None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get("SER_ROOT")
    return Path(env).expanduser().resolve() if env else DEFAULT_ROOT


def sources(root: Path) -> dict[str, Path]:
    """Raw inputs. Nothing under these folders is ever modified."""
    return {
        "own_audio": root / "our_audios",
        "own_index_dir": root / "firebase_data",
        "iemocap": root / "iemocap",
        "ravdess": root / "ravdess",
        "cremad": root / "cremad",
        "goemotions": root / "goemotions",        # optional local copy
    }


def outputs() -> dict[str, Path]:
    """Everything v2 writes lives under code\\v2, next to v1, never inside it."""
    d = {
        "prep": V2_DIR / "preprocessing",
        "audio_measure": V2_DIR / "preprocessing" / "audio" / "measurement",
        "audio_encoder": V2_DIR / "preprocessing" / "audio" / "encoder",
        "embeddings": V2_DIR / "preprocessing" / "embeddings",
        "audio_result": V2_DIR / "audio_result",
        "text_result": V2_DIR / "text_result",
        "fusion_result": V2_DIR / "fusion_result",
        "multimodal_result": V2_DIR / "multimodal_result",
        "models": V2_DIR / "models",
    }
    return d


def ensure_dirs() -> dict[str, Path]:
    d = outputs()
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def get_device(verbose: bool = True) -> str:
    """CUDA if available. On Windows this needs the CUDA build of PyTorch:
        pip install torch --index-url https://download.pytorch.org/whl/cu124
    """
    import torch
    if torch.cuda.is_available():
        if verbose:
            print(f"[GPU] {torch.cuda.get_device_name(0)}  "
                  f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        return "cuda"
    if verbose:
        print("[WARN] No CUDA GPU visible to PyTorch; running on CPU (much slower).")
        print("       Check with:  python -c \"import torch; print(torch.cuda.is_available())\"")
        print("       If False, reinstall the CUDA build of PyTorch (see config.py).")
    return "cpu"


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
