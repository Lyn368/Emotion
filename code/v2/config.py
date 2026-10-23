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


# ---------------------------------------------------------------------------
# Evaluation set: which dataset the branches train and test on.
#   own      the in-house sessions (default; unchanged behaviour)
#   ravdess  cremad  iemocap   public benchmarks built by p6_public_sessions.py
# Scripts set it with --eval. Public results go to code\v2\public_result\<name>\
# so they can never be mixed up with the in-house results.
# ---------------------------------------------------------------------------
EVAL_SETS = ("own", "ravdess", "cremad", "iemocap")
TEXT_EVAL_SETS = ("own", "iemocap")      # RAVDESS / CREMA-D repeat fixed sentences: no text signal


def set_eval(name: str) -> None:
    if name not in EVAL_SETS:
        raise SystemExit(f"[ERROR] --eval must be one of {EVAL_SETS}")
    os.environ["SER_EVAL"] = name


def eval_name() -> str:
    return os.environ.get("SER_EVAL", "own")


def eval_paths(name: str | None = None) -> dict:
    """Inputs of the active evaluation set."""
    name = name or eval_name()
    prep = V2_DIR / "preprocessing"
    if name == "own":
        return {"name": "own", "records": "own_dataset.json", "splits": prep / "splits.json",
                "acoustic": prep / "features" / "acoustic_own.json", "embeddings": "own"}
    return {"name": name, "records": f"{name}.json", "splits": prep / "public" / name / "splits.json",
            "acoustic": prep / "features" / f"acoustic_{name}.json", "embeddings": name}


def outputs() -> dict[str, Path]:
    """Everything v2 writes lives under code\\v2, next to v1, never inside it."""
    name = eval_name()
    res = V2_DIR if name == "own" else V2_DIR / "public_result" / name
    d = {
        "prep": V2_DIR / "preprocessing",
        "audio_measure": V2_DIR / "preprocessing" / "audio" / "measurement",
        "audio_encoder": V2_DIR / "preprocessing" / "audio" / "encoder",
        "embeddings": V2_DIR / "preprocessing" / "embeddings",
        "audio_result": res / "audio_result",
        "text_result": res / "text_result",
        "fusion_result": res / "fusion_result",
        "multimodal_result": V2_DIR / "multimodal_result",
        "models": V2_DIR / "models",                     # stage-1 checkpoints are shared
        "final_models": V2_DIR / "models" / "final" if name == "own" else res / "models_final",
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
