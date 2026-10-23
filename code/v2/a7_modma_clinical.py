"""
a7_modma_clinical.py - Depression (and anxiety, if the sheet has GAD-7) from speech on MODMA,
evaluated with the same code as a6, so the numbers can sit next to published MODMA results.

MODMA speech = the "audio_lanzhou_2015" package. The EEG packages (128 channels ERP,
128 channels resting, 3 channels resting) contain no speech; the audio package is a
separate MODMA dataset that has to be selected in the MODMA application. Put it inside
the same modma folder; this script finds it there.

    52 Mandarin speakers (23 MDD, 29 controls), ONE session each, 29 recordings:
    1-18 interview answers | 19 story reading ("The North Wind and the Sun")
    20-25 word reading     | 26-29 picture descriptions

Mapping to our design:
  freestyle  each interview answer (1-18). Add the pictures with --freestyle interview,picture
  passage    the story reading (19): same text for every speaker, so the delta works as on our data
  row        one interview answer. LOSO groups = subject (leave-one-subject-out, 52 folds)
  speaker    a subject's prediction = majority vote of its answers (F1, accuracy) and the mean of
             its scores (AUC). Published MODMA work reports speaker/subject-level numbers.

Targets
  MDD vs HC        clinical diagnosis (what most MODMA papers use)
  PHQ-9 >= 10      the label of the LOSO paper (speakers' PHQ-9 from the sheet)
  GAD-7 >= 10      only if the sheet has GAD-7
  With both scales present, PHQ-9 and GAD-7 also get config B (+ the other scale) and the
  "other scale only" reference, exactly as in a6.

Features (cached in preprocessing/modma/, computed only once)
  egemaps    openSMILE eGeMAPS v02 functionals (88) on the audio resampled to 16 kHz.
             a1's speaking rate counts space-separated words, which fails in Mandarin, so a1 is
             not used here.
  wavlm<k>   WavLM Base+ layer k, mean over time. wavlm7 = the layer of the LOSO baseline.
             wavlmmean = mean over all layers.
  Feature sets: <src>_free | <src>_delta_only | <src>_delta  (same meaning as in a6)

Models: linsvm (linear SVM: the published baseline), svm (RBF, as a6), mlp (as a6).

Published subject-disjoint results to compare with (speaker level):
  PHQ-9 >= 10, LOSO, 52 speakers, WavLM Base+ L7 + linear SVM   F1 (depressed = F1_pos) 0.762, AUC 0.704
      -> run: --targets phq9 --feature_sets wavlm7_free --models linsvm  (approximate replication:
         they classify 3-s chunks; here each answer is averaged)
  MDD vs HC, subject-level 5-fold, interview only, HAREN-CTC       macro-F1 0.644
  MDD vs HC, subject-level 5-fold, 38 speakers, HuBERT-large CN    macro-F1 0.809
  Results from segment-level splits or best-run reporting are not comparable.

    pip install opensmile openpyxl soundfile
    python a7_modma_clinical.py --modma C:\\Users\\2022g\\Downloads\\modma
    python a7_modma_clinical.py --modma ... --n_perm 0                       quick run
    python a7_modma_clinical.py --modma ... --feature_sets egemaps_free      no GPU needed

Output: audio_result/modma/  modma_summary.csv, modma_results.json, predictions_modma.json
MODMA's licence (EULA) forbids redistribution: keep the audio and predictions out of git.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

import a6_audio_clinical as A6
from a6_audio_clinical import NAN, _bal_acc, baselines, cluster_ci, fmt, louo, metrics, paired_delta
from config import ensure_dirs

SR = 16000
INTERVIEW, STORY, PICTURE = range(1, 19), 19, range(26, 30)
TARGETS = {"mdd": ("Depression: MDD vs HC (diagnosis)", None),
           "phq9": ("Depression (PHQ-9>=10)", "gad7"),
           "gad7": ("Anxiety (GAD-7>=10)", "phq9")}
WAVLM_CHUNK_S = 20


# ---------------------------------------------------------------------------
# Locating the data
# ---------------------------------------------------------------------------
def find_audio_root(modma: Path) -> Path | None:
    """The folder whose subfolders hold per-subject .wav files."""
    best, best_n = None, 0
    for d in [modma] + [p for p in modma.rglob("*") if p.is_dir()]:
        n = sum(1 for c in d.iterdir() if c.is_dir() and next(c.rglob("*.wav"), None) is not None)
        if n > best_n:
            best, best_n = d, n
    return best if best_n >= 5 else None


def norm_id(x) -> str:
    d = re.sub(r"\D", "", str(x).split(".")[0])
    return d.lstrip("0") or d


def recording_number(stem: str) -> int | None:
    nums = re.findall(r"\d+", stem)
    if not nums:
        return None
    n = int(nums[-1])
    if n > 29 and len(nums[-1]) >= 3:          # e.g. "0201000201": subject id glued to the number
        n = int(nums[-1][-2:])
    return n if 1 <= n <= 29 else None


def index_audio(root: Path) -> dict:
    """{subject_id: {recording number: wav path}}"""
    subjects, issues = {}, Counter()
    for d in sorted(c for c in root.iterdir() if c.is_dir()):
        clips = {}
        for w in sorted(d.rglob("*.wav")):
            n = recording_number(w.stem)
            if n is None:
                issues["files without a 1-29 number"] += 1
            elif n in clips:
                issues["duplicate numbers (first kept)"] += 1
            else:
                clips[n] = w
        if clips:
            subjects[norm_id(d.name)] = clips
    return subjects, issues


def find_sheet(root: Path, modma: Path) -> Path | None:
    for base in (root, root.parent, modma):
        hits = [p for p in sorted(base.glob("*")) if p.suffix.lower() in (".xlsx", ".xls", ".csv")]
        audio = [p for p in hits if "audio" in p.name.lower()]
        if audio or hits:
            return (audio or hits)[0]
    hits = [p for p in modma.rglob("*") if p.suffix.lower() in (".xlsx", ".xls") and "audio" in p.name.lower()]
    return hits[0] if hits else None


def read_sheet(path: Path) -> dict:
    """{subject_id: {"mdd": 0/1, "phq9": score, "gad7": score, "gender": str}} from the subjects sheet."""
    import pandas as pd
    try:
        raw = pd.read_csv(path, header=None) if path.suffix.lower() == ".csv" else pd.read_excel(path, header=None)
    except ImportError as e:
        raise SystemExit(f"[ERROR] {e}. Run: pip install openpyxl (for .xlsx) or xlrd (for .xls)")
    hdr = next((i for i in range(min(10, len(raw)))
                if any(re.search(r"phq|type|group|类型|组别", str(v), re.I) for v in raw.iloc[i])), 0)
    cols = [str(c).strip() for c in raw.iloc[hdr]]
    df = raw.iloc[hdr + 1:]

    def col(pat):
        return next((i for i, c in enumerate(cols) if re.search(pat, c, re.I)), None)

    c_id = col(r"subject|^id$|\bid\b|编号|被试")
    c_type, c_phq, c_gad = col(r"type|group|label|类型|组别"), col(r"phq"), col(r"gad")
    c_sex = col(r"gender|sex|性别")
    print(f"[INFO] sheet: {path.name} | columns: {cols}")
    print(f"  id -> {cols[c_id] if c_id is not None else 'NOT FOUND'} | type -> "
          f"{cols[c_type] if c_type is not None else 'not found'} | PHQ-9 -> "
          f"{cols[c_phq] if c_phq is not None else 'not found'} | GAD-7 -> "
          f"{cols[c_gad] if c_gad is not None else 'not found'} | gender -> "
          f"{cols[c_sex] if c_sex is not None else 'not found'}")
    if c_id is None:
        raise SystemExit("[ERROR] no subject id column in the sheet.")

    def mdd(v):
        s = str(v).strip().lower()
        if re.search(r"mdd|depress|patient|患者|抑郁", s):
            return 1
        if re.search(r"^(hc|nc)$|control|healthy|norm|健康|正常|对照", s):
            return 0
        return None

    def sex(v):
        s = str(v).strip().lower()
        return ("female" if s in ("f", "female", "女", "2") else "male" if s in ("m", "male", "男", "1")
                else (s if s and s != "nan" else None))

    out = {}
    for _, r in df.iterrows():
        sid = norm_id(r.iloc[c_id])
        if not sid:
            continue
        row = {"mdd": mdd(r.iloc[c_type]) if c_type is not None else None,
               "phq9": A6._num(r.iloc[c_phq]) if c_phq is not None else None,
               "gad7": A6._num(r.iloc[c_gad]) if c_gad is not None else None,
               "gender": sex(r.iloc[c_sex]) if c_sex is not None else None}
        if row["mdd"] is None and c_type is None:
            # MODMA ids: 0201xxxx = MDD, 0203xxxx = control (documented for EEG, used here only as fallback)
            row["mdd"] = 1 if sid.startswith("201") else 0 if sid.startswith("203") else None
        out[sid] = row
    if c_type is None:
        print("  [WARN] no type column: MDD/HC taken from the id prefix (0201 = MDD, 0203 = HC). Check it.")
    bad = [k for k, v in out.items() if c_type is not None and v["mdd"] is None]
    if bad:
        vals = Counter(str(df.iloc[i, c_type]) for i in range(len(df)))
        print(f"  [WARN] {len(bad)} subjects with an unreadable type. Values in the column: {dict(vals)}")
    return out


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def load_16k(path: Path) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = math.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return x


def egemaps_features(clips: dict, cache: Path) -> dict:
    """{clip_key: 88-d vector}. Raw (resampled) audio, no denoising or normalisation."""
    store = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    todo = [k for k in clips if k not in store]
    if todo:
        import opensmile
        smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                feature_level=opensmile.FeatureLevel.Functionals)
        t0 = time.time()
        for i, k in enumerate(todo, 1):
            df = smile.process_signal(load_16k(clips[k]), SR)
            store[k] = {c: float(v) for c, v in df.iloc[0].items()}
            if i % 100 == 0 or i == len(todo):
                cache.write_text(json.dumps(store), encoding="utf-8")
                print(f"  eGeMAPS {i}/{len(todo)} ({time.time() - t0:.0f}s)")
    names = sorted(next(iter(store.values())).keys())
    return {k: np.array([store[k][n] for n in names], dtype=np.float64) for k in clips}


def wavlm_features(clips: dict, cache_dir: Path, model_path: str) -> dict:
    """{clip_key: (layers+1, D) float16}: mean over time of every hidden layer."""
    idx_p, arr_p = cache_dir / "wavlm_index.json", cache_dir / "wavlm.npy"
    meta_p = cache_dir / "wavlm_model.txt"
    keys, arr = [], None
    if idx_p.exists() and arr_p.exists():
        if meta_p.exists() and meta_p.read_text().strip() != model_path:
            print(f"  [INFO] WavLM cache was built with another model; rebuilding for {model_path}")
        else:
            keys, arr = json.loads(idx_p.read_text()), np.load(arr_p)
    have = set(keys)
    todo = [k for k in clips if k not in have]
    if todo:
        import torch
        from transformers import AutoFeatureExtractor, WavLMModel
        device = "cuda" if torch.cuda.is_available() else "cpu"
        fe = AutoFeatureExtractor.from_pretrained(model_path)
        model = WavLMModel.from_pretrained(model_path, layerdrop=0.0).to(device).eval()
        print(f"  WavLM on {device}: {len(todo)} clips")
        new, t0 = [], time.time()
        for i, k in enumerate(todo, 1):
            x = load_16k(clips[k])
            total, frames = None, 0
            for s in range(0, max(len(x), 1), WAVLM_CHUNK_S * SR):
                seg = x[s:s + WAVLM_CHUNK_S * SR]
                if len(seg) < SR // 2:
                    if frames:
                        continue                      # tiny tail of a long clip
                    seg = np.pad(seg, (0, SR // 2 - len(seg)))
                inp = fe(seg, sampling_rate=SR, return_tensors="pt").input_values.to(device)
                with torch.no_grad(), torch.autocast(device_type="cuda", enabled=device == "cuda"):
                    hs = model(inp, output_hidden_states=True).hidden_states
                h = torch.stack(hs, 0)[:, 0].float()          # (layers+1, T, D)
                total = h.sum(1) if total is None else total + h.sum(1)
                frames += h.shape[1]
            new.append((total / frames).cpu().numpy().astype(np.float16))
            if i % 100 == 0 or i == len(todo):
                print(f"  WavLM {i}/{len(todo)} ({time.time() - t0:.0f}s)")
        arr = np.concatenate([arr, np.stack(new)]) if arr is not None else np.stack(new)
        keys = keys + todo
        np.save(arr_p, arr)
        idx_p.write_text(json.dumps(keys))
        meta_p.write_text(model_path)
    pos = {k: i for i, k in enumerate(keys)}
    return {k: arr[pos[k]] for k in clips}


def parse_set(name: str):
    for suf in ("_delta_only", "_delta", "_free"):
        if name.endswith(suf):
            return name[: -len(suf)], suf[1:]
    return name, "free"


def build_sets(names, rows, clips, cache_dir, wavlm_path):
    """Feature matrices for every requested set, one row per freestyle answer."""
    srcs = {parse_set(n)[0] for n in names}
    bad = [s for s in srcs if s != "egemaps" and not re.fullmatch(r"wavlm(\d+|mean)", s)]
    if bad:
        raise SystemExit(f"[ERROR] unknown feature source(s) {bad}. Use egemaps, wavlm<k> or wavlmmean.")
    eg = egemaps_features(clips, cache_dir / "egemaps.json") if "egemaps" in srcs else None
    wl = wavlm_features(clips, cache_dir, wavlm_path) if any(s.startswith("wavlm") for s in srcs) else None

    def vec(src, key):
        if key is None:
            return None
        if src == "egemaps":
            return eg[key]
        a = wl[key].astype(np.float64)
        if src == "wavlmmean":
            return a.mean(0)
        k = int(src[5:])
        if k >= a.shape[0]:
            raise SystemExit(f"[ERROR] {src}: the model has layers 0..{a.shape[0] - 1}")
        return a[k]

    sets = {}
    for name in names:
        src, mode = parse_set(name)
        D = len(vec(src, rows[0]["answer"]))
        F = np.vstack([vec(src, r["answer"]) for r in rows])
        P = np.vstack([vec(src, r["passage"]) if r["passage"] else np.full(D, np.nan) for r in rows])
        X = F if mode == "free" else (F - P if mode == "delta_only" else np.hstack([F, F - P]))
        sets[name] = {"X": X, "pca": src != "egemaps"}
    return sets


# ---------------------------------------------------------------------------
# Speaker-level evaluation
# ---------------------------------------------------------------------------
def to_speaker(pred, score, groups, order):
    """Majority vote of a speaker's answers (ties -> positive) and mean score."""
    p = np.array([int(pred[groups == g].mean() >= 0.5) for g in order])
    s = np.array([score[groups == g].mean() for g in order])
    return p, s


def _perm_spk(kind, X, y, groups, order, use_pca, n_audio, seed, k):
    rng = np.random.RandomState(seed * 100003 + k)
    yp = A6.block_permute(y, groups, rng)                 # one label per speaker -> plain shuffle of speakers
    ys = np.array([yp[groups == g][0] for g in order])
    if len(np.unique(ys)) < 2:
        return NAN
    pred, sc = louo(kind, X, yp, groups, use_pca, n_audio, seed)
    return _bal_acc(ys, to_speaker(pred, sc, groups, order)[0])


def perm_test_spk(kind, X, y, groups, order, use_pca, n_audio, seed, observed, n_perm, jobs):
    null = Parallel(n_jobs=jobs)(delayed(_perm_spk)(kind, X, y, groups, order, use_pca, n_audio, seed, k)
                                 for k in range(n_perm))
    null = np.array([v for v in null if not math.isnan(v)])
    if not len(null):
        return {"p_perm": NAN, "null_mean": NAN, "null_95": NAN}
    return {"p_perm": float((1 + (null >= observed - 1e-12).sum()) / (1 + len(null))),
            "null_mean": float(null.mean()), "null_95": float(np.percentile(null, 95))}


def print_table(title, rows):
    print(f"\n{'=' * 140}\n{title}\n{'=' * 140}")
    print(f"{'':<44}{'':<7}{'answer':>8}{'| speaker':>10}")
    print(f"{'config':<44}{'model':<7}{'bal_acc':>8}{'acc':>8}{'bal_acc':>9}{'95% CI':>15}{'mF1':>7}"
          f"{'F1_pos':>8}{'AUC':>7}{'p_perm':>8}   delta speaker bal_acc vs ref [95% CI]")
    for r in rows:
        ci = f"[{fmt(r.get('spk_bal_acc_lo'), 2)},{fmt(r.get('spk_bal_acc_hi'), 2)}]" if r.get("spk_bal_acc_lo") is not None else ""
        dl = (f"{r['delta_ref']:+.3f} [{r['delta_lo']:+.2f},{r['delta_hi']:+.2f}] vs {r['vs_ref'].split(' (')[0][5:]}"
              if r.get("vs_ref") else "")
        print(f"{r['config'][:43]:<44}{r['model']:<7}{fmt(r.get('ans_bal_acc')):>8}{fmt(r['spk_acc']):>8}"
              f"{fmt(r['spk_bal_acc']):>9}{ci:>15}{fmt(r['spk_macro_f1']):>7}{fmt(r['spk_f1_pos']):>8}"
              f"{fmt(r.get('spk_auc')):>7}{fmt(r.get('p_perm')):>8}   {dl}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Depression / anxiety from speech on MODMA (LOSO)")
    ap.add_argument("--modma", required=True, help="folder that contains the MODMA packages")
    ap.add_argument("--targets", default="mdd,phq9,gad7", help="mdd, phq9, gad7 (missing ones are skipped)")
    ap.add_argument("--freestyle", default="interview", help="interview and/or picture")
    ap.add_argument("--feature_sets", default="egemaps_free,egemaps_delta_only,egemaps_delta,"
                                                "wavlm7_free,wavlm7_delta_only")
    ap.add_argument("--models", default="linsvm,svm,mlp")
    ap.add_argument("--wavlm", default="microsoft/wavlm-base-plus", help="HF name or local folder")
    ap.add_argument("--n_perm", type=int, default=100, help="speaker-level permutations (0 = skip)")
    ap.add_argument("--perm_models", default="linsvm,svm")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=-1)
    a = ap.parse_args()

    t0 = time.time()
    dirs = ensure_dirs()
    cache_dir = dirs["prep"] / "modma"
    out_dir = dirs["audio_result"] / "modma"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(a.seed)
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    perm_models = {m.strip() for m in a.perm_models.split(",") if m.strip()}

    # ---- data -----------------------------------------------------------------
    modma = Path(a.modma)
    if not modma.is_dir():
        raise SystemExit(f"[ERROR] {modma} is not a folder")
    root = find_audio_root(modma)
    if root is None:
        found = ", ".join(sorted(p.name for p in modma.iterdir() if p.is_dir())) or "nothing"
        raise SystemExit(
            f"[ERROR] no speech recordings (.wav) under {modma}.\n"
            f"  Found: {found}\n"
            f"  Those are EEG packages. The speech data is a separate MODMA dataset\n"
            f"  (audio_lanzhou_2015): select it in your MODMA application, download it\n"
            f"  into {modma} and run again.")
    subjects, issues = index_audio(root)
    print(f"[INFO] audio: {root} | {len(subjects)} subjects")
    for k, v in issues.items():
        print(f"  [WARN] {k}: {v}")
    sheet = find_sheet(root, modma)
    if sheet is None:
        raise SystemExit("[ERROR] no subjects sheet (.xlsx / .xls / .csv) found next to the audio.")
    info = read_sheet(sheet)
    only_audio, only_sheet = set(subjects) - set(info), set(info) - set(subjects)
    print(f"  subjects in audio and sheet: {len(set(subjects) & set(info))}"
          + (f" | audio only: {sorted(only_audio)[:5]}" if only_audio else "")
          + (f" | sheet only: {len(only_sheet)}" if only_sheet else ""))

    free_nums = []
    for f in a.freestyle.split(","):
        free_nums += list(INTERVIEW) if f.strip() == "interview" else list(PICTURE) if f.strip() == "picture" else []
    rows, clips, no_story = [], {}, 0
    for sid in sorted(set(subjects) & set(info)):
        c = subjects[sid]
        story = f"{sid}:{STORY}" if STORY in c else None
        if story:
            clips[story] = c[STORY]
        else:
            no_story += 1
        for n in free_nums:
            if n in c:
                key = f"{sid}:{n}"
                clips[key] = c[n]
                rows.append({"subject": sid, "answer": key, "passage": story})
    per = Counter(r["subject"] for r in rows)
    print(f"  freestyle rows: {len(rows)} | answers per subject: min {min(per.values())} "
          f"max {max(per.values())} | subjects without the story reading: {no_story}")
    mdd_phq = Counter((info[s]["mdd"], None if info[s]["phq9"] is None else int(info[s]["phq9"] >= 10)) for s in per)
    if any(k[1] is not None for k in mdd_phq):
        print(f"  MDD x PHQ-9>=10 (subjects): MDD&>=10 {mdd_phq[(1, 1)]} | MDD&<10 {mdd_phq[(1, 0)]} "
              f"| HC&>=10 {mdd_phq[(0, 1)]} | HC&<10 {mdd_phq[(0, 0)]}")

    sets = build_sets([s.strip() for s in a.feature_sets.split(",") if s.strip()], rows, clips,
                      cache_dir, a.wavlm)
    groups_all = np.array([r["subject"] for r in rows])

    # ---- tasks ----------------------------------------------------------------
    all_rows, predictions, diagnostics = [], {}, {}
    for tkey in [t.strip() for t in a.targets.split(",") if t.strip()]:
        if tkey not in TARGETS:
            raise SystemExit(f"[ERROR] unknown target {tkey}")
        title, other = TARGETS[tkey]
        lab = {s: (info[s][tkey] if tkey == "mdd" else
                   (None if info[s][tkey] is None else int(info[s][tkey] >= A6.CUTOFF))) for s in per}
        keep = np.array([lab[g] is not None for g in groups_all])
        if not keep.any():
            print(f"\n[{title}] no labels in the sheet; skipped.")
            continue
        idx = np.where(keep)[0]
        groups = groups_all[idx]
        y = np.array([lab[g] for g in groups])
        order = np.array(sorted(set(groups)))
        y_spk = np.array([lab[g] for g in order])
        if len(np.unique(y_spk)) < 2:
            print(f"\n[{title}] one class only; skipped.")
            continue
        n_pos = int(y_spk.sum())
        print(f"\n[{title}] speakers: {len(order)} ({n_pos} positive, {len(order) - n_pos} negative) "
              f"| answers: {len(y)}")
        diagnostics[title] = {"speakers": int(len(order)), "positive": n_pos, "answers": int(len(y))}

        if other and any(info[s][other] is not None for s in order):
            oth = np.array([np.nan if info[g][other] is None else info[g][other] for g in groups], float)
            other_cols = np.column_stack([oth, np.isnan(oth).astype(float)])
        else:
            other_cols = None

        tasks = []
        gvals = np.array([info[g]["gender"] or "unknown" for g in groups])
        if len(set(gvals) - {"unknown"}) >= 2:
            cats = sorted(set(gvals))
            tasks.append(("ref: gender only", "-", np.column_stack([gvals == c for c in cats]).astype(float), False, 0))
        ref_other = f"ref: {other.upper()} only (no audio)" if other_cols is not None else None
        if ref_other:
            tasks.append((ref_other, "-", other_cols, False, 0))
        for name, s in sets.items():
            Xa = s["X"][idx]
            tasks.append((f"A: audio only | {name}", name, Xa, s["pca"], Xa.shape[1]))
            if other_cols is not None:
                tasks.append((f"B: audio + {other.upper()} | {name}", name, np.hstack([Xa, other_cols]),
                              s["pca"], Xa.shape[1]))

        rows_out, preds_spk = [], {}
        for bname, m in baselines(y_spk, order, rng).items():
            rows_out.append({"config": bname, "model": "-", "feature_set": "-",
                             "spk_acc": m["acc"], "spk_bal_acc": m["bal_acc"], "spk_macro_f1": m["macro_f1"],
                             "spk_f1_pos": m["f1_elevated"], "spk_auc": NAN})
        for label, fset, X, use_pca, n_audio in tasks:
            for kind in models:
                t1 = time.time()
                pred, sc = louo(kind, X, y, groups, use_pca, n_audio or X.shape[1], a.seed)
                ps, ss = to_speaker(pred, sc, groups, order)
                ma, msp = metrics(y, pred, sc), metrics(y_spk, ps, ss)
                r = {"config": label, "model": kind.upper(), "feature_set": fset, "n_features": int(X.shape[1]),
                     "ans_bal_acc": ma["bal_acc"], "ans_macro_f1": ma["macro_f1"], "ans_auc": ma["auroc"],
                     "spk_acc": msp["acc"], "spk_bal_acc": msp["bal_acc"], "spk_macro_f1": msp["macro_f1"],
                     "spk_f1_pos": msp["f1_elevated"], "spk_auc": msp["auroc"]}
                r["spk_bal_acc_lo"], r["spk_bal_acc_hi"] = cluster_ci(y_spk, ps, order, rng)
                if a.n_perm > 0 and kind in perm_models and label[:2] in ("A:", "B:"):
                    r |= perm_test_spk(kind, X, y, groups, order, use_pca, n_audio or X.shape[1], a.seed,
                                       r["spk_bal_acc"], a.n_perm, a.jobs)
                print(f"  {label} {kind}: speaker F1_pos {r['spk_f1_pos']:.3f} "
                      f"mF1 {r['spk_macro_f1']:.3f} ({time.time() - t1:.0f}s)")
                rows_out.append(r)
                preds_spk[(label, r["model"])] = ps
                predictions.setdefault(title, {})[f"{label} | {kind}"] = {
                    "speakers": order.tolist(), "y": y_spk.tolist(), "pred": ps.tolist(),
                    "score": np.round(ss, 5).tolist()}

        for r in rows_out:
            ref = (ref_other if r["config"].startswith("B:")
                   else "ref: gender only" if r["config"].startswith("A:") else None)
            if ref and (ref, r["model"]) in preds_spk:
                dlt, lo, hi = paired_delta(y_spk, preds_spk[(r["config"], r["model"])],
                                           preds_spk[(ref, r["model"])], order, rng)
                r |= {"vs_ref": ref, "delta_ref": dlt, "delta_lo": lo, "delta_hi": hi}
        print_table(title, rows_out)
        for r in rows_out:
            all_rows.append({"target": title, "n_speakers": int(len(order)), "n_answers": int(len(y))} | r)

    # ---- save -----------------------------------------------------------------
    cols = ["target", "config", "feature_set", "model", "n_speakers", "n_answers", "n_features",
            "ans_bal_acc", "ans_macro_f1", "ans_auc", "spk_acc", "spk_bal_acc", "spk_bal_acc_lo",
            "spk_bal_acc_hi", "spk_macro_f1", "spk_f1_pos", "spk_auc", "p_perm", "null_mean", "null_95",
            "vs_ref", "delta_ref", "delta_lo", "delta_hi"]
    with open(out_dir / "modma_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    with open(out_dir / "modma_results.json", "w", encoding="utf-8") as f:
        json.dump({"settings": vars(a) | {"svm": A6.SVM_PARAMS, "linsvm": A6.LINSVM_PARAMS, "mlp": A6.MLP_PARAMS,
                                          "pca_var_wavlm": A6.PCA_VAR, "protocol": "leave-one-subject-out",
                                          "speaker_rule": "majority vote of answers (ties -> positive); AUC on mean score"},
                   "diagnostics": diagnostics, "rows": all_rows}, f, indent=2, default=str)
    with open(out_dir / "predictions_modma.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f)
    print(f"\n[DONE] {time.time() - t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()