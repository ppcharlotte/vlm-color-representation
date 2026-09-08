# LLaVA Structured Color Pipeline

This is the modular pipeline for the color representation thesis experiments.
The current story line compares visual, textual, perceptual, query, encoding-side,
and generation-side color representations.

## Research Logic

1. Compare image color embeddings, text color embeddings, and CIEDE2000 perceptual color geometry.
2. Analyze query embeddings under two prompt conditions: `force_choice` and `no_color_list`.
3. Analyze CEe: encoding-side contextual color-word embeddings using ground-truth color words.
4. Analyze CEg: generation-side embeddings from actual model generation.
5. Relate representation alignment metrics to color naming accuracy.

## Main Representations

- `image`: image-token embeddings from selected LLaVA layers.
- `text`: context-averaged color-word embeddings from 30 text contexts.
- `query_force_choice`: query embeddings when the prompt lists the 11 color choices.
- `query_no_color_list`: query embeddings when the prompt asks for a basic color without listing choices.
- `cee_ground_truth`: encoding-side color-word hidden states from prompts like `<image> The color of this image is blue`.
- `ceg_decision`: hidden states used by the LM head to predict the generated color word.
- `ceg_token_from_generate`: generated color-token hidden states from the original `generate()` pass, taken at the following generation step.

## Full Pipeline

```bash
python3 -B scripts/run_llava_structured_pipeline.py \
  --input-csv data/colorsurvey_rgb_aggregated_new.csv \
  --output-dir experiments/llava_structured_alignment \
  --model-id llava-hf/llava-1.5-7b-hf \
  --layer-indices 0,1,8,16,24,32 \
  --samples-per-color 100 \
  --patch-size 224 \
  --permutation-rounds 1000
```

The color naming step evaluates two prompt conditions:

- `force_choice`: lists the 11 basic color names.
- `no_color_list`: asks for one basic color name without listing candidates.

Patch sampling uses `majority_count > 2` by default. If a color has fewer than 10 samples after that filter, that color falls back to `majority_count >= 2`. Black is a special case: if strict black samples are fewer than `samples_per_color`, it is filled with `majority_count == 2` black samples.

The CEg step uses this prompt by default:

```text
What color is this image? Answer with a short sentence in this exact format: The color is <color>.
```

## OneVision Minimal Run

The shared `LlavaHelper` now auto-detects `lmms-lab/LLaVA-OneVision-1.5-*` models and uses the remote `AutoModelForCausalLM` loader with Qwen2VL-style processor inputs. For OneVision, start with a small run before launching the full pipeline:

```bash
python3 -B scripts/color_naming.py \
  --input-csv data/colorsurvey_rgb_aggregated_new.csv \
  --output-dir experiments/onevision_minimal \
  --model-id lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --samples-per-color 2

python3 -B scripts/image_embedding_extraction.py \
  --ground-truth experiments/onevision_minimal/data/ground_truth.csv \
  --dataset-root experiments/onevision_minimal/data \
  --output-dir experiments/onevision_minimal \
  --model-id lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --layer-indices 0,1,9,18,27,36 \
  --permutation-rounds 100

python3 -B scripts/text_embedding_extraction.py \
  --output-dir experiments/onevision_minimal \
  --model-id lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --layer-indices 0,1,9,18,27,36

python3 -B scripts/context_embedding_encoding_analysis.py \
  --ground-truth experiments/onevision_minimal/data/ground_truth.csv \
  --dataset-root experiments/onevision_minimal/data \
  --output-dir experiments/onevision_minimal \
  --model-id lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --layer-indices 0,1,9,18,27,36 \
  --permutation-rounds 100

python3 -B scripts/generation_context_embedding_analysis.py \
  --ground-truth experiments/onevision_minimal/data/ground_truth.csv \
  --dataset-root experiments/onevision_minimal/data \
  --output-dir experiments/onevision_minimal \
  --model-id lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
  --device cuda \
  --dtype bfloat16 \
  --layer-indices 0,1,9,18,27,36 \
  --permutation-rounds 100
```

Recommended OneVision layer sets:

- smoke/debug: `0,1,9,18,27,36`
- final comparison: `0,1,8,16,24,32,36`

For OneVision, `pre_projector` and `post_projector` in `image_embeddings_*.npz` are compatibility placeholders copied from the layer-0 image-token representation. The real OneVision analysis target is `image_layer_embeddings`.

## Independent Steps

Create color patches and run color naming:

```bash
python3 -B scripts/color_naming.py \
  --input-csv data/colorsurvey_rgb_aggregated_new.csv \
  --output-dir experiments/llava_structured_alignment
```

Extract image embeddings and image-CIEDE2000 alignment:

```bash
python3 -B scripts/image_embedding_extraction.py \
  --ground-truth experiments/llava_structured_alignment/data/ground_truth.csv \
  --dataset-root experiments/llava_structured_alignment/data \
  --output-dir experiments/llava_structured_alignment
```

Extract context-averaged text color-word embeddings:

```bash
python3 -B scripts/text_embedding_extraction.py \
  --output-dir experiments/llava_structured_alignment
```

Compare text color-word RDMs with perceptual CIEDE2000 color geometry:

```bash
python3 -B scripts/text_perceptual_rsa.py \
  --ground-truth experiments/llava_structured_alignment/data/ground_truth.csv \
  --dataset-root experiments/llava_structured_alignment/data \
  --output-dir experiments/llava_structured_alignment \
  --permutation-rounds 1000
```

Extract query embeddings:

```bash
python3 -B scripts/query_embedding_extraction.py \
  --ground-truth experiments/llava_structured_alignment/data/ground_truth.csv \
  --dataset-root experiments/llava_structured_alignment/data \
  --output-dir experiments/llava_structured_alignment
```

Extract and analyze CEe embeddings:

```bash
python3 -B scripts/context_embedding_encoding_analysis.py \
  --ground-truth experiments/llava_structured_alignment/data/ground_truth.csv \
  --dataset-root experiments/llava_structured_alignment/data \
  --output-dir experiments/llava_structured_alignment \
  --model-id llava-hf/llava-1.5-7b-hf \
  --layer-indices 0,1,8,16,24,32 \
  --permutation-rounds 1000
```

Extract and analyze CEg embeddings:

```bash
python3 -B scripts/generation_context_embedding_analysis.py \
  --ground-truth experiments/llava_structured_alignment/data/ground_truth.csv \
  --dataset-root experiments/llava_structured_alignment/data \
  --output-dir experiments/llava_structured_alignment \
  --model-id llava-hf/llava-1.5-7b-hf \
  --layer-indices 0,1,8,16,24,32 \
  --permutation-rounds 1000
```

Run image/query vs text similarity, RSA, CKA, and probe analyses:

```bash
python3 -B scripts/similarity_analysis.py \
  --ground-truth experiments/llava_structured_alignment/data/ground_truth.csv \
  --dataset-root experiments/llava_structured_alignment/data \
  --output-dir experiments/llava_structured_alignment
```

Relate representation metrics to final behavior:

```bash
python3 -B scripts/relation_analysis.py \
  --output-dir experiments/llava_structured_alignment
```

Run negative controls:

```bash
python3 -B scripts/negative_control.py \
  --ground-truth experiments/llava_structured_alignment/data/ground_truth.csv \
  --dataset-root experiments/llava_structured_alignment/data \
  --output-dir experiments/llava_structured_alignment
```


## Randomness / Sampling Robustness Plots

After running the full pipeline with multiple random seeds, use this script to plot each seed as a thin line and the across-seed mean +/- SD as a thick line with a shaded band.

Example with three runs:

```bash
python3 -B scripts/plot_randomness_summary.py \
  --run-dirs \
    experiments/llava_seed1 \
    experiments/llava_seed2 \
    experiments/llava_seed3 \
  --run-labels seed1,seed2,seed3 \
  --output-dir experiments/randomness_plots
```

By default, the script tries the main robustness plots, including:

- image vs CIEDE2000 centroid/sample RSA
- query no-color-list vs text color RSA
- CEe vs text color RSA
- CEg-decision vs CIEDE2000/image centroid RSA
- CEg-token-from-generate vs CIEDE2000/image centroid RSA

To plot one specific preset:

```bash
python3 -B scripts/plot_randomness_summary.py \
  --run-dirs experiments/llava_seed1 experiments/llava_seed2 experiments/llava_seed3 \
  --run-labels seed1,seed2,seed3 \
  --spec image_no_color_list_cielab_centroid_rsa \
  --output-dir experiments/randomness_plots
```

To plot a custom layer-wise CSV metric:

```bash
python3 -B scripts/plot_randomness_summary.py \
  --run-dirs experiments/llava_seed1 experiments/llava_seed2 experiments/llava_seed3 \
  --relative-csv similarity/query_no_color_list_layerwise.csv \
  --y-column linear_cka \
  --name query_no_color_list_linear_cka_randomness
```

## Output Structure

Main output directory:

```text
experiments/llava_structured_alignment/
```

Important subdirectories:

- `data/`: selected RGB samples, ground truth CSV, generated PNG patches.
- `color_naming/`: VLM responses, class metrics, and confusion matrices.
- `embeddings/`: image, query, text, and CEe embeddings.
- `rdm/`: model RDMs and CIEDE2000 RDMs.
- `similarity/`: layer-wise cosine, pair margin, RSA, CKA, Procrustes, and probe summaries for image/query/CEe.
- `linear_probe/`: linear color probe results by layer.
- `encoding_side/`: CEe prompts and summary.
- `ceg_generation_side/`: CEg predictions, embeddings, CIELAB/image/text comparisons, and plots.
- `relation/`: correlations and regressions linking representation metrics to color naming behavior.
- `negative_control/`: shuffled/permuted label controls.
- `plots/`: confusion matrices, error cases, and layer-wise plots.
- `summary.json`: final pipeline summary.
