# Pipeline v2

Rebuilt pipeline, kept side by side with v1 so every v2 number can be compared
against the v1 number it replaces. Code and outputs both live in `code\v2`.

## Status

| Phase | Scripts | State |
|---|---|---|
| 1. Preprocessing | `p1`-`p5`, `labels.py`, `cv.py`, `report.py`, `stats.py` | **done, tested** |
| 2. Audio branch | `a1`-`a4`, `audio_common.py` | **done, tested** |
| 3. Text branch | `t1`-`t3`, `text_common.py` | **done, tested** |
| 4. Fusion | `f1_fusion.py` | **done, tested** |
| 5. Multimodal | `m1_multimodal.py` | **done, tested on synthetic files; check the config on the real ones** |

## Setup on Windows (GPU)

Install the **CUDA build of PyTorch first**. If another package pulls in PyTorch
before you do, pip installs the CPU build and everything silently runs on CPU.

```bat
python -m venv .venv
.venv\Scripts\activate

:: check your CUDA version with:  nvidia-smi
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install numpy scipy scikit-learn matplotlib soundfile pyloudnorm silero-vad opensmile
pip install openai-whisper transformers datasets

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The last line must print `True` and your GPU name. Every script also prints the
GPU it found, or a warning if it fell back to CPU.

ffmpeg is only needed if your audio is not WAV/FLAC (e.g. `.webm`, `.m4a`).
Whisper does not need it: audio is passed to it as an array.

## Run order

```bat
cd C:\Users\2022g\Downloads\RAVDESS_IEMOCAP_CREMAD\code\v2

python p0_reconcile_own.py        :: link every video record (firebase + teammates) to an audio
python p1_index_datasets.py        :: find every clip in all datasets
python p2_preprocess_audio.py      :: two audio versions + VAD timestamps
python p3_transcribe.py            :: Whisper on GPU (resumable)
python p4_build_datasets.py        :: one JSON per dataset
python p5_make_splits.py --seeds 42,43,44,45,46
```

Useful options: `--limit 20` on p2/p3 for a timing test; `--datasets own,iemocap`
to restrict; `--model large-v3` on p3 for better transcripts. p2 and p3 skip
clips already done, so an interrupted run resumes.

## What each stage writes (`code\v2\preprocessing\`)

| File | Content |
|---|---|
| `index.jsonl` | every clip found, with raw label and speaker |
| `audio\measurement\` | audio with no denoising or normalisation (for pitch, jitter, shimmer, energy) |
| `audio\encoder\` | cleaned audio for Whisper and the encoders |
| `audio_manifest.jsonl` | speech timestamps, measured loudness, SNR per clip |
| `transcripts.jsonl` | Whisper output per clip |
| `own_dataset.json`, `iemocap.json`, `ravdess.json`, `cremad.json`, `goemotions.json` | one record per clip/text |
| `splits.json` | shared folds for every protocol and seed |

## Design rules that matter

- **Raw labels are stored, not resolved.** `"Neutral,Happiness"` and IEMOCAP's
  `"fru"` are kept as-is; each branch decides how to map them.
- **`label_source`** marks self-report vs annotator labels so they are never mixed.
- **IEMOCAP is transcribed with Whisper too** (plus the official transcript), so
  text models see real ASR errors before they see yours.
- **Every branch reads `splits.json`.** No script creates its own folds. This is
  what makes audio and text predictions line up for fusion.
- Splits are verified: no overlap between train/val/test, speakers disjoint for
  grouped and LOUO (outer and inner), every session tested once per protocol.
- Paths inside the JSONs are relative to `code\v2`, so the folder can be moved.
- Every file is read and written as UTF-8 (Windows defaults to cp1252).

## Reports (`report.py`)

Every result folder gets `metrics.json` (7- and 3-class, 95% speaker-bootstrap
CIs, permutation test), `predictions_oof.csv`, `errors.csv` (opens correctly in
Excel), and plots: confusion matrices (7 and 3), TP/FP/FN/TN per class,
per-class precision/recall/F1 with CIs, accuracy per speaker, true vs predicted
distribution, confidence of right vs wrong predictions, and calibration.
Fusion also gets the audio-vs-text agreement matrix.

## Phase 2 - audio branch

```bat
pip install funasr                                   :: only for emotion2vec

python a1_acoustic_features.py                       :: eGeMAPS + pauses + F0/loudness shape
python a2_encoder_embeddings.py --model wavlm_base_plus
python a2_encoder_embeddings.py --model emotion2vec_base
python a3_audio_frozen.py --protocols grouped        :: every frozen configuration

python a4_audio_finetune.py stage1                   :: WavLM on IEMOCAP
python a4_audio_finetune.py cv --unfreeze 2 --from_stage1 --protocols grouped
python a4_audio_finetune.py cv --unfreeze 0 --from_stage1 --protocols grouped   :: layer ablation
python a4_audio_finetune.py cv --unfreeze 4 --from_stage1 --protocols grouped
python a4_audio_finetune.py cv --unfreeze 2 --from_stage1 --no_delta             :: passage ablation
```

Iterate with `--protocols grouped`; run `louo` and `--seeds 42,43,44,45,46` only
for the finalists (fine-tuning retrains WavLM once per fold).

| Script | What it does |
|---|---|
| `a1` | ~100 continuous features per clip from the measurement audio. Nothing is discretised. |
| `a2` | Frozen 4 s window embeddings. WavLM keeps all 13 layers; emotion2vec one vector. |
| `a3` | Frozen runs: `acoustic`, `<enc>`, `<enc>_delta`, `acoustic+<enc>_delta`; ranked table with paired tests. |
| `a4` | WavLM fine-tuning: `stage1` (IEMOCAP), `cv`, `final`. |

Model: freestyle and passage go through the SAME encoder and pooling (siamese);
the session vector is `[f, f - p]`, never the passage alone. Layer weights are
learned. The acoustic block is `[freestyle, freestyle - passage, ratios]`.

8 GB GPU settings in `a4`: CNN front-end frozen, top `--unfreeze` layers train,
mixed precision, gradient checkpointing, 4 sessions x (4 + 2) windows per step.
If you get out-of-memory, lower `--sessions_per_step` or `--free_segs`.

emotion2vec is used frozen only: it loads through funasr, which is built for
inference. WavLM is the fine-tuned encoder.

Outputs: `audio_result/frozen|finetune/<run>/<protocol>/seed<k>/` with the full
report, plots and `oof.json` (the input of the fusion stage);
`models/stage1/` and `models/final/`.

## Phase 3 - text branch

```bat
python t2_text_models.py --protocols grouped                 :: direct + frozen, 3 encoders, full + sentence
python t1_text_pretrain.py --encoder emo_distilroberta       :: GoEmotions + IEMOCAP improvised
python t2_text_models.py --modes finetune --encoders emo_distilroberta --protocols grouped
python t3_text_llm.py --limit 20                             :: timing test
python t3_text_llm.py                                        :: Qwen2.5-3B zero-shot (fits 8 GB)
```

Optional NRC lexicons (free for research, saifmohammad.com):
`--lexicon --emolex NRC-Emotion-Lexicon-Wordlevel-v0.92.txt --vad NRC-VAD-Lexicon.txt`

`t3` SCORES the seven labels instead of parsing a written answer, so it cannot
fail to parse and it returns real probabilities. Its prompt is frozen (SHA-1 saved).
For Qwen 7B on 8 GB: `pip install bitsandbytes` and add `--load_4bit`.

## Phase 4 - fusion

```bat
python f1_fusion.py --audio best --text best --protocol grouped
python f1_fusion.py --audio best --text best --strategies late,stack,llm
```

`best` = the run with the highest macro F1 on its VALIDATION sessions, searched
over every saved run (choosing on test would be selection on the test set).
Strategies: `late` (audio weight chosen per fold on validation), `stack`
(logistic regression on validation), `llm` (Qwen scores the labels from the
transcript, speaker-relative voice changes and both branches' probabilities).
Also writes the audio-vs-text agreement matrix: if audio is rarely right where
text is wrong, no fusion can help.

## Phase 5 - multimodal

```bat
python m1_multimodal.py --inspect          :: reads teammates' files, writes multimodal_sources.json
:: open multimodal_sources.json and check it
python m1_multimodal.py --protocol grouped
```

Reports audio+text, audio+text+<each source>, and all, for 7 emotions and for 3
valence classes, with paired tests against audio+text. Guards: label-like and
identifier fields are dropped from metadata; any source that is >= 95% accurate
alone is flagged. **Teammates' predictions must be out-of-fold on the same
speaker-disjoint splits; send them `preprocessing\splits.json`.**
