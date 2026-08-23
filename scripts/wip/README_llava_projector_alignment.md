# LLaVA Projector-Space Alignment Experiment

This script tests the question: after LLaVA's multimodal projector maps visual tokens into the LLM hidden dimension, are those projected image representations geometrically aligned with color-word embeddings in the LLM input space?

It complements two existing experiments:

- OpenCLIP alignment: dual-encoder image-text cosine alignment.
- Generative color naming: final text output behavior.

This script sits in between: it analyzes LLaVA's internal projector-space representations.

## Install

```bash
pip install -r requirements-generative.txt
```

LLaVA 7B is large. Python 3.10-3.12 and enough RAM/VRAM are recommended.

## Quick Run: Projector Output Only

```bash
python3 -B scripts/run_llava_projector_alignment_experiment.py \
  --model-id llava-hf/llava-1.5-7b-hf \
  --ground-truth data/color_patches_aggregated_100/ground_truth.csv \
  --output-dir experiments/llava_projector_alignment \
  --batch-size 1
```

## Layer-wise Run

This extracts image-token hidden states and color-token hidden states at every LLM layer, then computes alignment and RSA per layer.

```bash
python3 -B scripts/run_llava_projector_alignment_experiment.py \
  --model-id llava-hf/llava-1.5-7b-hf \
  --ground-truth data/color_patches_aggregated_100/ground_truth.csv \
  --output-dir experiments/llava_layerwise_alignment \
  --batch-size 1 \
  --run-layerwise
```

To analyze only selected layers:

```bash
python3 -B scripts/run_llava_projector_alignment_experiment.py \
  --model-id llava-hf/llava-1.5-7b-hf \
  --ground-truth data/color_patches_aggregated_100/ground_truth.csv \
  --output-dir experiments/llava_layerwise_alignment_selected \
  --batch-size 1 \
  --run-layerwise \
  --layer-indices 0,1,8,16,24,32
```

To also analyze image-conditioned query tokens:

```bash
python3 -B scripts/run_llava_projector_alignment_experiment.py \
  --model-id llava-hf/llava-1.5-7b-hf \
  --ground-truth data/color_patches_aggregated_100/ground_truth.csv \
  --output-dir experiments/llava_query_alignment_selected \
  --batch-size 1 \
  --run-layerwise \
  --layer-indices 0,1,8,16,24,32 \
  --run-query-analysis
```

Recommended structured run for the thesis analysis, including no-color-list query and a previous generative-output run:

```bash
python3 -B scripts/run_llava_projector_alignment_experiment.py \
  --model-id llava-hf/llava-1.5-7b-hf \
  --ground-truth data/color_patches_aggregated_100/ground_truth.csv \
  --output-dir experiments/llava_structured_alignment_selected \
  --batch-size 1 \
  --run-layerwise \
  --layer-indices 0,1,8,16,24,32 \
  --run-query-analysis \
  --run-query-no-color-list-analysis \
  --generative-results-dir experiments/generative_color_naming_llava
```

Projector-space diagnostic plots are disabled by default because projector output is treated as the layer-0/baseline representation. Add `--plot-projector-diagnostics` only if you want the old projector-specific heatmaps.

## Reanalysis From Saved Embeddings

```bash
python3 -B scripts/run_llava_projector_alignment_experiment.py \
  --embeddings-npz experiments/llava_projector_alignment/llava_projector_embeddings.npz \
  --output-dir experiments/llava_projector_alignment_reanalysis \
  --permutation-rounds 10000
```

## Outputs

- `llava_projector_embeddings.npz`: projected image vectors and LLM color text vectors.
- `summary.json`: projector diagnostics, layer-wise alignment, linear probes, generative-output summaries, and p-values.
- `sample_predictions.csv`: diagnostic similarity-based predictions in projector space; this is not the final VLM output.
- `class_metrics.csv`: diagnostic per-color cosine-prototype accuracy and class-wise alignment.
- `correlations.csv`: correlations and two-sided permutation p-values.
- `confusion_matrix.csv`: projector-space similarity confusion matrix.
- `rdm_projected_image_centroids.csv`: RDM of projected image centroids.
- `rdm_llm_color_text_embeddings.csv`: RDM of LLM color text embeddings.
- `plots/*.png`: visual summaries when matplotlib is installed.
- `layerwise_metrics.csv`: per-layer cosine-prototype diagnostic accuracy, pair alignment, correct-color similarity, RSA, CKA, Procrustes error, and p-values when `--run-layerwise` is used.
- `image_layerwise_metrics.csv`: same content with a clearer image-token name; use this as the main image-token layerwise table.
- `layerwise_correlations.csv`: diagnostic layer-level correlations against image-only linear probe accuracy.
- `image_layerwise_correlations.csv`: same diagnostic correlations under the image-token naming convention.
- `image_layerwise_vs_generative.csv`: main image-token test. For each selected layer, reports whether pair margin, correct-color similarity, and class-wise RSA predict the final generative VLM output correctness/accuracy from `--generative-results-dir`.
- `layerwise_linear_probe_class_metrics.csv`: image-only linear probe accuracy for each selected layer, by color.
- `image_layerwise_linear_probe_class_metrics.csv`: same per-color probe results under the image-token naming convention.
- `query_layerwise_metrics.csv`: query-token linear probe, query-token/text cosine-prototype diagnostic accuracy, correct-color similarity, RSA, CKA, and Procrustes by selected layer.
- `query_layerwise_correlations.csv`: diagnostic layer-level correlations for query-token alignment metrics against query linear probe accuracy.
- `query_layerwise_vs_generative.csv`: main query-token test against final generative VLM output correctness/accuracy.
- `query_layerwise_linear_probe_class_metrics.csv`: query-token linear probe accuracy for each selected layer, by color.
- `query_no_color_list_layerwise_metrics.csv`: same query-token analysis using a prompt that does not list the color options.
- `query_no_color_list_layerwise_correlations.csv`: layer-level correlations for no-color-list query analysis.
- `query_no_color_list_layerwise_vs_generative.csv`: main no-color-list query-token test against final generative VLM output correctness/accuracy.
- `query_no_color_list_layerwise_linear_probe_class_metrics.csv`: per-color probe results for no-color-list query analysis.
- `vlm_output_summary.json`: copied/summarized behavior result from `--generative-results-dir`.
- `vlm_output_class_accuracy.csv`: per-color accuracy from actual generated text outputs.
- `vlm_output_confusion_matrix.csv`: confusion matrix from actual generated text outputs.
- `final_layer_linear_probe_class_metrics.csv`: image-only linear probe accuracy for the last selected layer, by color.
- `final_layer_linear_probe_predictions.csv`: held-out predictions from the final selected layer linear probe.
- `plots/layerwise_*.png`: layer-wise curves when `--run-layerwise` and matplotlib are available.

## Interpretation

Projected image tokens and LLM token embeddings share hidden dimensionality, but direct cosine nearest-neighbor classification is only a diagnostic. The main question is whether image/query-to-color-word similarity and relational alignment predict the actual colors generated by the VLM.

## Layer-wise Interpretation

Layer index `0` corresponds to the LLM input embedding state before transformer blocks. Later indices correspond to successive LLM hidden states. If projector output is poorly aligned but later layers improve in `pair_alignment_mean`, `mean_correct_similarity`, `rsa_spearman`, or `linear_cka`, this supports the idea that color-word alignment emerges through LLM computation rather than existing immediately after the projector.

The layer-wise image representation is the mean-pooled hidden state over inserted image token positions. The text representation is the mean-pooled hidden state over the color word tokens in each color prompt.

For the layer-wise run, the main rows to inspect are:

- `pair_alignment_mean`: instance-level image-text alignment margin for that layer.
- `mean_correct_similarity`: direct cosine similarity between each image/query representation and its true color-word embedding.
- `rsa_spearman`: structural alignment between image color centroid RDM and text color RDM for that layer.
- `linear_cka`: geometry similarity between image color centroids and text color embeddings.
- `procrustes_error`: residual error after orthogonal text-to-image alignment; lower is better.
- `cosine_prototype_accuracy`: diagnostic cosine prototype classification accuracy using that layer's image and text embeddings.
- `linear_probe_accuracy`: image-only linear probe accuracy using that layer's image hidden states.

When `--generative-results-dir` points to a completed `run_generative_color_naming_experiment.py` directory, inspect these main fields in `image_layerwise_vs_generative.csv`, `query_layerwise_vs_generative.csv`, and `query_no_color_list_layerwise_vs_generative.csv`:

- `sample_pair_margin_vs_generative_correct_r/p`: whether the sample-level true-color margin predicts whether the final generated color is correct.
- `sample_correct_similarity_vs_generative_correct_r/p`: whether direct similarity to the true color word predicts final generated correctness.
- `class_pair_margin_vs_generative_accuracy_r/p`: whether color classes with higher margins have higher final generated accuracy.
- `class_correct_similarity_vs_generative_accuracy_r/p`: whether color classes with higher true-word similarity have higher final generated accuracy.
- `classwise_rsa_vs_generative_accuracy_r/p`: whether local color-space structural alignment predicts per-color final generated accuracy.

The final-layer linear probe uses only image hidden states from the last selected layer. This helps separate whether color is linearly decodable from the image representation itself versus whether that image representation is aligned to color words.

The per-layer linear probe makes the same comparison at every selected layer. A layer can have low `cosine_prototype_accuracy` but high `linear_probe_accuracy`; that pattern means color is decodable from image hidden states but is not directly aligned with color-word embeddings. Whether that misalignment matters behaviorally should be checked in the `*_layerwise_vs_generative.csv` files.

## Query Interpretation

`--run-query-analysis` extracts the image-conditioned query/prompt token states from the same selected layers. These states test whether color information has moved from image tokens into the text-side context. In `query_layerwise_metrics.csv`, `query_linear_probe_accuracy` asks whether color is linearly decodable from query states, while `rsa_spearman` and `linear_cka` ask whether query color geometry aligns with color-word geometry.

`--run-query-no-color-list-analysis` repeats the same analysis with a prompt that does not list the 11 color options. Use this as the cleaner query-token control when interpreting whether query states carry image-derived color information rather than prompt-list leakage.
