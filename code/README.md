# Audio branch — CE903-SP team01

Rebuilt pipeline. Six scripts, run in order. `common.py` holds everything the
classifiers share, which is why `04`, `05` and `06` are short — the six
near-identical CV scripts (`sc.py`, `sc3.py`, `sc_kfold.py`, `sc3_kfold.py`,
`sc3_groupedkfold.py`) collapse into one `--cv` flag.

## Order

```bash
python 01_audio_extraction.py --recursive          # unchanged; already done
python 02_audio_preprocessing.py                   # writes wavs + manifest
python 03_feature_extraction.py                    # writes features.jsonl
python 04_audio_emotion.py  --cv all               # audio-only
python 05_text_emotion.py   --cv all               # text-only
python 06_session_fusion.py --cv all --strategies late,joint,llm
```

`04`, `05` and `06` each produce 7-class **and** 3-class results in a single
run. `--cv all` runs K-Fold, Grouped K-Fold and LOUO and prints a comparison
table at the end.

---

## Three bugs this fixes

**Pitch was a speaker-sex detector.** `ER.py:257` had
`pitches.append(150.0)` — a hardcoded constant instead of `extract_pitch(p)`.
That made μ=150, σ=0, so `_discretise(v, 150, 0)` degenerated into a hard
threshold at 150 Hz, which separates typical male from typical female voices.
`pitch_level` was encoding *who was speaking*, which is exactly the leakage
LOUO exists to prevent. Now every clip is measured, and `discretise()` returns
`"Medium"` rather than a garbage bucket when σ is degenerate.

**Energy was constant.** `Audio_Preprocessing_v2.py` normalised every clip to
−23 LUFS, then `ER.py` read `energy_lufs` back off those normalised files.
Every clip reported ≈ −23, σ ≈ 0. `02` now measures loudness *before*
normalising and records it in `preprocess_manifest.jsonl`; `03` reads it from
there. The normalised audio still exists — Whisper and emotion2vec want
consistent levels — but the real loudness survives.

**Speaking rate was contaminated.** Only freestyle was transcribed, so passage
clips had `transcript=""` → rate 0, and those zeros dragged the corpus μ and σ.
`03` now transcribes both. Rate is also computed over post-VAD *voiced*
seconds rather than total duration, so it measures articulation rate instead
of confounding tempo with how much the person paused.

---

## Two design changes

**Per-user statistics.** C2SER discretises pitch/energy/rate at μ ± σ. With 9
speakers of differing baselines, *global* μ and σ make "High pitch" mean
"this is speaker 4". Per-user μ and σ make it mean "high **for this person**",
which is what carries emotional information in a corpus of neutral-toned,
non-acted speech. Users with fewer than 8 clips fall back to global stats and
are flagged in the output.

**Passage as a per-session neutral reference.** The passage script is
identical every session, so its *lexical* content carries nothing and it stays
out of the text model. But that is precisely what makes its *acoustics*
valuable: a fixed script, same speaker, same session, same microphone. The
difference between freestyle and passage cancels both the speaker's baseline
voice and that session's recording conditions. This follows Aldeneh & Mower
Provost — emotion measured as departure from the speaker's own neutral
reference rather than in absolute terms.

This is **not leakage**: the passage is available at inference time for the
same session, it never crosses users, and it never touches labels. It is a
different thing from the speaker-level normalisation that paper criticises,
which does use speaker identity at evaluation time.

---

## Ablations to run for the report

```bash
python 04_audio_emotion.py  --cv all --no_delta     # is the passage reference earning its place?
python 05_text_emotion.py   --cv all --no_augment   # does ASR-noise augmentation help?
python 06_session_fusion.py --cv louo --strategies late,joint,llm
```

Three comparisons worth a table each:

**Leakage gap.** `--cv all` prints `kfold − louo` macro F1. That number *is*
the amount standard K-Fold overstates performance by letting the model
recognise users it has already seen. Report grouped or LOUO as the headline;
kfold belongs in the table only as an upper bound.

**Dedicated 3-class head vs post-hoc mapping.** Every run prints both. A
positive gap means the valence head learns something the 7-class head does
not; near zero means post-hoc mapping was already sufficient. This is the
question your earlier result (Macro F1 0.270 → 0.482 under 3-class) raises but
cannot answer — post-hoc mapping alone can't distinguish "valence is easier"
from "a dedicated head helps".

**LLM vs late fusion.** `06` runs both. If the LLM only matches a weighted
average of the two branches, it is not earning its cost. The
`llm_agreement_report` shows how often it copied the text branch, matched
audio, or went independent — MASKSER's whole KL-alignment contribution exists
because one modality tends to dominate, and with neutral-toned speech plus
content-rich transcripts the expected failure here is the LLM ignoring prosody.

---

## Known limitations to state in the write-up

**Early stopping uses the test fold.** With 9 users there is no room for a
separate validation split inside each fold, so `train_fold()` early-stops on
test macro F1. This is mildly optimistic. It must be stated, but the relative
comparison across CV methods stays valid because all three share it.

**Self-report vs perception-of-other labels.** Your labels are self-reported.
emotion2vec, and essentially every pretrained SER model, is trained on
perception-of-other annotations. Prior work finds perception-of-other labels
transfer poorly to self-report contexts (see the PRIORI paper in the project
folder). This is a legitimate, citable reason your absolute numbers are not
comparable to IEMOCAP-style results, and it explains part of the low Macro F1.

**"Surprise" is grouped as positive.** Valence-ambiguous in affective science
generally; grouped positive here because it co-occurs with happiness far more
often than with negative emotions in this corpus. Defensible but not
universal — say so in the report. It lives in `EMOTION_TO_3CLASS` in
`common.py` if you want to test the alternative.

---

## Dependencies

```bash
pip install torch torchaudio numpy scikit-learn
pip install silero-vad pyloudnorm soundfile        # 02
pip install funasr openai-whisper penn             # 03
pip install sentence-transformers                  # 05
pip install transformers accelerate                # 06 (Qwen)
```

`penn` and `pyloudnorm` are optional — `03` falls back to autocorrelation
pitch and RMS energy — but the fallbacks are noticeably worse, and without
`pyloudnorm` in `02` the manifest has no `original_lufs` and `03` will warn
that energy is unusable.

---

# ADDENDUM — Cambios para publicación

## 1. CV anidado (arreglo obligatorio)

`train_fold_explicit()` ahora hace early stopping sobre un split de validación
interno separado del entrenamiento. Para `grouped` y `louo` ese split interno
es **disjunto por hablante**, igual que el fold externo — si no, el early
stopping seleccionaría checkpoints que generalizan dentro del hablante pero no
entre hablantes, que es justo lo que esos esquemas existen para medir.

Los datos aumentados se añaden **después** del split interno, así que nunca
entran a validación. Validar sobre transcripts corrompidos sintéticamente
seleccionaría checkpoints que ajustan al proceso de corrupción.

**Espera números más bajos que antes.** Esa es la señal de que el arreglo
funcionó.

## 2. Rigor estadístico (`stats.py`)

- `cluster_bootstrap_ci` — IC al 95% remuestreando **hablantes**, no sesiones.
  Las sesiones de un mismo hablante no son independientes; remuestrear filas
  trata 300 observaciones correlacionadas como 300 independientes y produce
  intervalos falsamente estrechos. El intervalo honesto refleja que tu N real
  es 15 personas.
- `paired_bootstrap_test` — el test para cada ablation del paper: con vs sin
  referencia neutral, con vs sin augmentación, LLM vs late fusion.
- `permutation_test_predictions` — screening contra azar.
- `aggregate_seeds` — reporta media ± std sobre ≥5 semillas. Un paper con una
  sola semilla en un corpus pequeño recibe petición de rehacer experimentos.

Formato para el paper: `macro F1 = 0.312, 95% CI [0.241, 0.388]`.

## 3. Corpus públicos (`00_prepare_public_dataset.py`)

Convierte RAVDESS, CREMA-D, EmoDB e IEMOCAP al formato del pipeline,
emparejando cada utterance emocional con la referencia neutral **del mismo
hablante y la misma frase**. Ninguna utterance es su propia referencia, y la
selección de referencia nunca mira la etiqueta del target.

RAVDESS con `--drop_neutral` da 576 sesiones / 6 clases balanceadas / 24
hablantes. Sin el flag, 720 sesiones / 7 clases (los takes neutrales extra se
usan como target).

## Estructura del paper que esto habilita

El aporte no es el pipeline: es **la referencia neutral intra-sesión con
contenido léxico controlado**. Con 15 participantes ese aporte no se puede
sostener estadísticamente. Con RAVDESS (24 hablantes, referencia emparejada por
frase) y CREMA-D (91 hablantes) sí.

- **Experimento 1 — validación del mecanismo.** RAVDESS + CREMA-D, ablation
  con y sin el bloque delta, `paired_bootstrap_test` sobre la diferencia.
  Aquí es donde se demuestra que la normalización funciona.
- **Experimento 2 — demostración in-the-wild.** Los 15 participantes, habla
  espontánea, etiquetas de autorreporte. Aquí es donde está la novedad de los
  datos.

**No los presentes como un benchmark unificado.** RAVDESS y CREMA-D son
actuados: la referencia es neutral-actuado, no neutral-leído, y el habla
emocional es interpretada, no espontánea. Validan el mecanismo; no demuestran
que transfiera a habla naturalista. Eso es lo que aporta tu corpus propio.

## Lo que sigue pendiente

- Fine-tuning del encoder en vez de features congeladas
- Procesamiento a nivel segmento (ventanas de 2s con solapamiento)
- SpecAugment para balancear clases minoritarias
