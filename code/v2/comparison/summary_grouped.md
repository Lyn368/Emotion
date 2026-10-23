# Results across datasets - protocol: grouped

Mean over seeds. WA = accuracy, UA = mean per-class recall. `perm p` tests the first seed's macro F1 against chance.

| Dataset | Sessions | Speakers | Majority acc | Chance UA |
|---|---|---|---|---|
| own | 384 | 14 | 0.419 | 0.143 |
| ravdess | 1248 | 24 | 0.154 | 0.143 |
| cremad | 7441 | 91 | 0.171 | 0.167 |
| iemocap | 5531 | 10 | 0.309 | 0.250 |

| Dataset | Branch | Best run | Seeds | WA | UA | Macro F1 | 95% CI (seed 1) | perm p | Valence acc | Valence F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| own | audio | frozen/acoustic | 1 | 0.276 | 0.155 | 0.153 | [0.100, 0.196] | 0.1770 | 0.341 | 0.320 |
| own | fusion | frozen-emotion2vec_base__goemo_roberta_full_frozen/llm | 1 | 0.464 | 0.400 | 0.370 | [0.290, 0.444] | 0.0002 | 0.560 | 0.495 |
| own | text | emo_distilroberta_full_finetune_stage1 | 1 | 0.510 | 0.440 | 0.440 | [0.316, 0.524] | 0.0002 | 0.555 | 0.543 |
| ravdess | audio | frozen/wavlm_base_plus_delta | 1 | 0.790 | 0.800 | 0.793 | [0.744, 0.836] | 0.0002 | 0.898 | 0.877 |
| cremad | audio | frozen/wavlm_base_plus_delta | 1 | 0.780 | 0.784 | 0.784 | [0.767, 0.801] | 0.0002 | 0.912 | 0.886 |
| iemocap | audio | frozen/wavlm_base_plus | 1 | 0.660 | 0.674 | 0.667 | [0.645, 0.689] | 0.0002 | 0.671 | 0.664 |
| iemocap | fusion | frozen-wavlm_base_plus__emo_distilroberta_sentence_finetune_stage1/stack | 1 | 0.721 | 0.734 | 0.729 | [0.708, 0.749] | 0.0002 | 0.732 | 0.726 |
| iemocap | text | emo_distilroberta_full_finetune_stage1 | 1 | 0.647 | 0.650 | 0.522 | [0.506, 0.671] | 0.0002 | 0.671 | 0.667 |

Best run is picked on these test results, so it is optimistic; use it to read the table, not as a claim. Claims need the paired tests below or in each branch's summary.

## Neutral reference (delta): with vs without

| Dataset | Encoder | F1 without | F1 with | Delta | 95% CI | p |
|---|---|---|---|---|---|---|
| own | emotion2vec_base | 0.109 | 0.117 | +0.008 | [-0.030, +0.046] | 0.3370 |
| own | wavlm_base_plus | 0.117 | 0.116 | -0.001 | [-0.022, +0.018] | 0.5285 |
| own | finetune:wavlm_base_plus_unfreeze2_stage1 | 0.112 | 0.114 | +0.002 | [-0.015, +0.019] | 0.4220 |
| ravdess | wavlm_base_plus | 0.739 | 0.793 | +0.054 * | [+0.018, +0.091] | 0.0005 |
| cremad | wavlm_base_plus | 0.727 | 0.784 | +0.057 * | [+0.045, +0.070] | 0.0000 |
| iemocap | wavlm_base_plus | 0.667 | 0.631 | -0.037 | [-0.065, -0.014] | 1.0000 |

* p < 0.05 (one-sided paired speaker bootstrap). A CI crossing zero = no detected effect.