# LLaVA Structured Color Pipeline

This is the new modular pipeline for the color representation thesis experiments. Old experiment entry scripts have been moved to `scripts/wip/`.

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

The full run evaluates two prompt conditions:

- `force_choice`: lists the 11 basic color names.
- `no_color_list`: asks for one basic color name without listing candidates.

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

Extract text color-word embeddings:

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

Key outputs:

- `rdm/text_cielab_alignment.csv`
- `rdm/text_cielab_alignment_summary.json`
- `plots/text_cielab_rsa_spearman.png`
- `plots/text_cielab_rsa_kendall.png`
- `plots/text_cielab_rsa_pearson.png`

Materialize query embeddings:

```bash
python3 -B scripts/query_embedding_extraction.py \
  --ground-truth experiments/llava_structured_alignment/data/ground_truth.csv \
  --dataset-root experiments/llava_structured_alignment/data \
  --output-dir experiments/llava_structured_alignment
```

Extract and analyze answer-position hidden states before color-word generation:

```bash
python3 -B scripts/answer_position_embedding_analysis.py \
  --ground-truth experiments/llava_structured_alignment/data/ground_truth.csv \
  --dataset-root experiments/llava_structured_alignment/data \
  --output-dir experiments/llava_structured_alignment \
  --model-id llava-hf/llava-1.5-7b-hf \
  --layer-indices 0,1,8,16,24,32 \
  --permutation-rounds 1000
```

This step saves generation-side color representations and compares them with input-side text color embeddings, LM-head output color-token vectors, image embeddings, and CIEDE2000 perceptual color distances.

Key outputs:

- `embeddings/answer_position_embeddings_force_choice.npz`
- `embeddings/answer_position_embeddings_no_color_list.npz`
- `embeddings/lm_head_color_vectors.npy`
- `similarity/answer_force_choice_layerwise.csv`
- `similarity/answer_lm_head_force_choice_layerwise.csv`
- `generation_side/answer_image_alignment.csv`
- `generation_side/answer_cielab_alignment.csv`

Run similarity/RSA/CKA/probe analyses:

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

## Output Structure

Main output directory:

```text
experiments/llava_structured_alignment/
```

Important subdirectories:

- `data/`: selected RGB samples, ground truth CSV, generated PNG patches.
- `color_naming/`: VLM responses, class metrics, confusion matrices.
- `embeddings/`: image, query, and text embeddings as `.npz` files.
- `rdm/`: model RDMs and CIEDE2000 RDMs.
- `similarity/`: layer-wise cosine, pair margin, RSA, CKA, and Procrustes results.
- `linear_probe/`: linear color probe results by layer.
- `relation/`: correlations and regressions linking representation metrics to final color naming behavior.
- `generation_side/`: answer-position hidden-state analyses before color-word generation.
- `negative_control/`: shuffled/permuted label controls.
- `plots/`: confusion matrices, error cases, and layer-wise plots.
- `summary.json`: final pipeline summary.

