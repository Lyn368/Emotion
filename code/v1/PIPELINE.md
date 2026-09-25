# Pipeline v3 — encoder propio + corpus públicos + selección de features

Archivos nuevos: `features_audio.py`, `08_train_encoder.py`,
`09_extract_embeddings.py`, `10_benchmark_public.py`, `11_feature_selection.py`.
Modificados: `00`, `03`, `04`, `06`, `common.py`.
`handcrafted.py` queda obsoleto — `features_audio.py` lo reemplaza y mantiene la
misma API (`extract_handcrafted`, `feature_vector`), así que nada más se rompe.

---

## Orden completo

### Fase A — corpus públicos

```bash
# A1. Parsear los 3 corpus y separar el 20% de HABLANTES (no de archivos)
python 00_prepare_public_dataset.py \
    --base ~/Downloads/RAVDESS_IEMOCAP_CREMAD \
    --datasets ravdess,cremad,iemocap \
    --out data_public --holdout 0.2 --iemocap_holdout session

python 00_prepare_public_dataset.py --base "C:\Users\2022g\Downloads\RAVDESS_IEMOCAP_CREMAD" --datasets ravdess,cremad,iemocap --out data_public --holdout 0.2

bash data_public/materialise_audio.sh
& "C:\Program Files\Git\bin\bash.exe" data_public/materialise_audio.sh

# A2. Limpiar / normalizar (mismo script de siempre)
python 02_audio_preprocessing.py \
    --audio_root data_public/audios \
    --output_dir data_public/PreprocessedAudio

# A3. Banco de features handcrafted. Sin Whisper: RAVDESS y CREMA-D leen
#     frases fijas, el transcript no aporta señal de sesión. Sin emotion2vec
#     tampoco, salvo que quieras la comparación e2v sobre el holdout (la
#     quieres, así que déjalo).
python 03_feature_extraction.py \
    --audio_dir data_public/PreprocessedAudio \
    --out data_public/features.jsonl \
    --video_index data_public/firebase_data/video_index.json \
    --no_asr --workers 8 --fast
```

### Fase B — entrenar tu encoder

```bash
# B1. Corrida barata primero: backbone congelado, solo entrena la cabeza.
#     Sirve para verificar que todo corre antes de gastar horas.
python 08_train_encoder.py \
    --utterances data_public/utterances.jsonl \
    --backbone microsoft/wavlm-base-plus --freeze_backbone \
    --epochs 6 --batch 16 --out encoders/wavlm_bp_frozen

# B2. La corrida real: fine-tune de las 6 capas superiores
python 08_train_encoder.py \
    --utterances data_public/utterances.jsonl \
    --backbone microsoft/wavlm-base-plus --unfreeze_top 6 \
    --epochs 12 --batch 8 --accum 2 --crop 4.0 \
    --out encoders/wavlm_bp_ft

python 08_train_encoder.py --utterances data_public/utterances.jsonl --backbone microsoft/wavlm-base-plus --unfreeze_top 6 --epochs 12 --batch 8 --accum 2 --crop 4.0 --out encoders/wavlm_bp_ft

# B3. Baseline del profesor (log-mel + CNN), entrena en minutos
python 08_train_encoder.py \
    --utterances data_public/utterances.jsonl \
    --backbone melcnn --epochs 30 --batch 32 \
    --out encoders/melcnn

# B4. Opcional: fusionar el banco handcrafted dentro del encoder
python 08_train_encoder.py \
    --utterances data_public/utterances.jsonl \
    --backbone microsoft/wavlm-base-plus --unfreeze_top 6 \
    --hc_features data_public/features.jsonl \
    --hc_blocks mfcc,dmfcc,ddmfcc,spectral,lsf,prosody \
    --epochs 12 --out encoders/wavlm_bp_hc
```

### Fase C — benchmark en el holdout público

```bash
python 09_extract_embeddings.py \
    --encoder encoders/wavlm_bp_ft/encoder.pt \
    --audio_dir data_public/PreprocessedAudio \
    --features data_public/features.jsonl

python 10_benchmark_public.py --data data_public \
    --systems custom_zeroshot,e2v_zeroshot,custom_mlp,e2v_mlp,hc_mlp,custom_mlp_delta,e2v_mlp_delta \
    --compare custom_mlp:e2v_mlp,custom_mlp_delta:custom_mlp,e2v_mlp_delta:e2v_mlp
```

Sale una tabla markdown lista para pegar, con accuracy (WA), UA, macro/micro/
weighted F1, por corpus y global, con intervalos bootstrap por hablante.

### Fase D — nuestros audios

```bash
# D1. Preprocesar (la extracción de audio ya está hecha, no corras 01)
python 02_audio_preprocessing.py \
    --audio_root data/audios --output_dir data/PreprocessedAudio

# D2. Features: aquí SÍ va Whisper (habla espontánea) y emotion2vec
python 03_feature_extraction.py --workers 8

# D3. Aplicar tu encoder
python 09_extract_embeddings.py \
    --encoder encoders/wavlm_bp_ft/encoder.pt \
    --audio_dir data/PreprocessedAudio \
    --features data/features.jsonl

# D4. Emoción de audio: mismo protocolo, cambia solo la representación
python 04_audio_emotion.py --cv all --encoder emotion2vec
python 04_audio_emotion.py --cv all --encoder custom
python 04_audio_emotion.py --cv all --encoder custom --hc_blocks mfcc,dmfcc,spectral,lsf,prosody
python 04_audio_emotion.py --cv all --encoder both

# D5. Texto y fusión
python 05_text_emotion.py --cv all
python 06_session_fusion.py --cv louo --encoder custom --strategies late,joint,llm
```

### Fase E — qué features sirven

```bash
# Cada familia sola, y el conjunto completo menos una familia
python 11_feature_selection.py --cv grouped --mode single,loo

# Selección greedy + ranking feature por feature
python 11_feature_selection.py --cv grouped --mode greedy,perfeature --top_k 40

# Lo mismo sobre el corpus público (más hablantes = conclusiones más firmes)
python 11_feature_selection.py --data data_public --cv grouped --mode single,loo
```

El modo que más te va a servir es `loo`. Una familia puede rendir bien sola y no
aportar nada al conjunto: eso significa redundante, que es el resultado normal
entre familias acústicas correlacionadas. Corre `single` y `loo` en el corpus
público antes de sacar conclusiones del tuyo — con 15 hablantes cualquier
diferencia de 3 puntos está dentro del ruido.

---

## Decisiones que hay que defender en el paper

**El holdout es por hablante, no por archivo.** RAVDESS da 60 grabaciones por
actor. Un 20% aleatorio de archivos deja las otras 48 del mismo actor en
entrenamiento, el encoder reporta sobre voces que ya memorizó, y el número no es
comparable con nada publicado. Para IEMOCAP se reserva Session5, que es el split
que usa la literatura. `splits.json` se escribe una vez: si lo regeneras con otra
semilla, todo lo calculado antes deja de ser comparable.

**Máscara de etiquetas por corpus.** CREMA-D no tiene `surprise`. Un softmax de 7
vías entrenado sobre la unión aprendería que audio de CREMA-D es evidencia *en
contra* de surprise — artefacto de diseño del corpus, no de habla. `08` pone a
−inf los logits de las clases que el corpus del sample no contiene.

**Comparación justa entre encoders.** `custom_mlp` vs `e2v_mlp`: ambos
congelados, mismo MLP, mismas sesiones, mismo split. `custom_zeroshot` es
end-to-end y NO va en la misma columna sin decirlo.

**micro F1 = accuracy** exactamente, en tarea single-label multiclase. Se
reportan los dos porque los pediste, pero ponlo en nota al pie. UA/UAR (recall
promedio por clase) es la métrica que sí complementa a WA en los papers de SER, y
ya se imprime en todos lados.

**LLM solo en `06`.** Whisper (ASR) y MiniLM (sentence embeddings) no son LLMs
generativos. Vale decirlo explícito: *"no generative model is used in the
unimodal branches; the only LLM in the pipeline operates at the fusion stage"*.

**Dominio.** Los corpus públicos son actuados, tu corpus es espontáneo con
etiquetas de autorreporte. Si tu encoder transfiere peor que emotion2vec a tus
audios, eso es un hallazgo publicable, no un fracaso: es la brecha
acted → in-the-wild, que es exactamente lo que tu corpus aporta.

---

## El banco de features (`features_audio.py`)

445 features con los defaults, ~0.17 s por clip una vez que numba compila.
Bloques con nombre, para que `11` pueda prenderlos y apagarlos:

| bloque | contenido | dims aprox |
|---|---|---|
| `mfcc` | MFCC 1-20, mean/std/skew/kurt | 80 |
| `dmfcc` | Δ MFCC | 80 |
| `ddmfcc` | ΔΔ MFCC | 80 |
| `logmel` | 40 bandas log-mel, mean/std | 80 |
| `spectral` | momentos espectrales reales (centroid, spread, skewness, kurtosis, slope, decrease, entropy), flux, rolloff 85/95, bandwidth, flatness, contrast, zcr, rms | 74 |
| `lsf` | Line Spectral Frequencies, LPC orden 16 | 32 |
| `prosody` | F0 (mean/std/mediana/p5/p95/rango/IQR/log/Δ), voiced fraction, jitter, shimmer, HNR | 19 |
| `egemaps` | openSMILE eGeMAPSv02, opcional | 88 |

Dos tipos de skewness, porque la literatura usa la misma palabra para ambos:
`sm_skewness_*` es el 3er momento de la energía **a través de la frecuencia**
dentro de un frame; los sufijos `_skew` en cualquier contorno son el 3er momento
**a lo largo del tiempo**. Los dos se calculan.

Jitter, shimmer y HNR son aproximaciones a nivel de frame, no las de Praat por
ciclo glotal. Está documentado en el docstring — dilo en el paper. Si resultan
importantes, instala `opensmile` y usa el bloque `egemaps`, que sí son las
implementaciones citables.

Tus Δpitch / Δenergy / Δrate y las razones siguen ahí. Además `03` ahora calcula
el delta contra el passage para **todas** las features handcrafted, no solo esas
tres, así que `11` puede evaluar cada familia con y sin su gemelo delta.

## Encoder: por qué WavLM

El pre-entrenamiento de WavLM agrega denoising de habla superpuesta sobre la
predicción enmascarada, lo que lo obliga a modelar *quién habla y cómo*, no solo
*qué se dice*. Ese es el contenido paralingüístico que necesita SER, y es por lo
que lidera la tarea de emoción en SUPERB. Conecta directo con el paper que ya
tienes en la carpeta sobre features de emoción emergiendo del reconocimiento de
hablante — es la misma afirmación desde el otro lado.

Alternativas si quieres una fila más en la tabla: `microsoft/hubert-large-ls960`,
`facebook/wav2vec2-large-robust`, o el encoder de `openai/whisper-large-v3`
(el más robusto a ruido, buena opción para habla in-the-wild).

Orden práctico en Mac: `wavlm-base-plus` congelado (minutos) →
`wavlm-base-plus --unfreeze_top 6` (horas en MPS) → `wavlm-large` solo con GPU
CUDA. No empieces con large en laptop.

## Dependencias nuevas

```bash
pip install librosa transformers soundfile
pip install opensmile          # opcional, para el bloque egemaps
```

## Qué falta

- Fine-tuning completo del encoder en vez de solo las capas superiores
- Procesamiento a nivel segmento (`08` ya recorta a 4 s; `09` promedia ventanas)
- SpecAugment está activo vía `--spec_aug` en el backbone SSL y en `melcnn`
- Cuando lleguen video / wearable / glucosa / multiactividad, entran en `06`
  como ramas adicionales: la estructura de `build_aligned` ya es por sesión
