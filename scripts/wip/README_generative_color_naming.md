# Generative VLM Color Naming Experiment

This script evaluates a generative VLM on pure color patches by asking it to answer exactly one of the 11 basic color names.

## Install

For Qwen-VL style HuggingFace models:

```bash
pip install torch torchvision transformers accelerate pillow
```

Large VLMs may need a Python 3.10-3.12 environment and enough RAM/VRAM.

## Quick Test

```bash
python3 -B scripts/run_generative_color_naming_experiment.py \
  --backend transformers-qwen-vl \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --ground-truth data/color_patches_aggregated_100/ground_truth.csv \
  --output-dir experiments/generative_color_naming_qwen_test \
  --limit 20
```

## Full Run

```bash
python3 -B scripts/run_generative_color_naming_experiment.py \
  --backend transformers-qwen-vl \
  --model-id Qwen/Qwen2.5-VL-3B-Instruct \
  --ground-truth data/color_patches_aggregated_100/ground_truth.csv \
  --output-dir experiments/generative_color_naming_qwen
```


## LLaVA Run

```bash
python3 -B scripts/run_generative_color_naming_experiment.py \
  --backend transformers-llava \
  --model-id llava-hf/llava-1.5-7b-hf \
  --ground-truth data/color_patches_aggregated_100/ground_truth.csv \
  --output-dir experiments/generative_color_naming_llava \
  --limit 20
```

For a full run, remove `--limit 20`.

## Outputs

- `sample_predictions.csv`: raw model answer, normalized color, correctness, latency.
- `summary.json`: normalized accuracy, parse rate, macro-F1, metadata-correctness correlations, and p-values.
- `correlations.csv`: sample-level correlations with two-sided permutation p-values.
- `class_metrics.csv`: per-color recall/precision/F1.
- `confusion_matrix.csv`: true color by predicted color.
- `plots/*.png`: confusion matrix, per-color recall/precision/F1, parse status, and metadata-vs-correct scatter plots.

## Why This Is Separate From OpenCLIP

OpenCLIP classifies by image-text cosine similarity and does not generate text. This script tests generative color naming: the model receives an image and must produce the color word directly.

## Significance Values and Plots

The script reports two-sided permutation p-values for sample-level correlations such as `agreement_rate -> correct`, `label_entropy_bits -> correct`, and `total_votes -> correct` when those metadata columns exist in the ground-truth CSV. Increase precision with `--permutation-rounds 10000`.

Plots require matplotlib. Disable plots with:

```bash
python3 -B scripts/run_generative_color_naming_experiment.py --no-plots ...
```
