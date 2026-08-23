# Color Patch Alignment Experiment

This experiment script implements the pure-color-patch pilot from `research_notes/color_alignment_method_plan_zh.md`.

## Input

Default dataset:

```text
data/color_patches_aggregated_100/ground_truth.csv
```

Each row points to one generated pure color patch and has a `basic_color` label.

## Install Model Dependencies

The analysis metrics only need Python, NumPy, and Pillow, but embedding extraction needs CLIP/OpenCLIP:

```bash
python3 -m pip install torch open_clip_torch pillow numpy
```

If you already have embeddings, you can skip model loading with `--embeddings-npz`.

## Run OpenCLIP Experiment

```bash
python3 -B scripts/run_color_patch_alignment_experiment.py \
  --ground-truth data/color_patches_aggregated_100/ground_truth.csv \
  --output-dir experiments/color_patch_alignment_vitb32 \
  --model ViT-B-32 \
  --pretrained openai \
  --device auto \
  --batch-size 64
```

## Outputs

The output directory contains:

- `embeddings.npz`: image/text embeddings and labels.
- `summary.json`: main accuracy, alignment, RSA, CKA, Procrustes, separability, linear-probe results, and p-values.
- `correlations.csv`: correlation coefficients with two-sided permutation p-values.
- `sample_predictions.csv`: per-image zero-shot predictions and pair-alignment margins.
- `class_metrics.csv`: per-color accuracy and class-wise RSA.
- `confusion_matrix.csv`: zero-shot confusion matrix.
- `rdm_visual_centroids.csv`: visual color-space RDM.
- `rdm_text_prototypes.csv`: text color-space RDM.
- `linear_probe_class_metrics.csv`: image-only linear probe per-color results.
- `procrustes_interventions.csv`: original/correct/random/permuted alignment intervention results.
- `plots/*.png`: confusion matrix, visual/text RDM heatmaps, class accuracy, margin histogram, class-level scatter plots, and intervention bar chart.

## Analyze Existing Embeddings

```bash
python3 -B scripts/run_color_patch_alignment_experiment.py \
  --embeddings-npz experiments/color_patch_alignment_vitb32/embeddings.npz \
  --output-dir experiments/color_patch_alignment_vitb32_reanalysis
```

## Significance Values

The script reports two-sided permutation p-values for global RDM correlations and class/sample-level alignment-accuracy correlations. Use `--permutation-rounds` to increase precision, for example `--permutation-rounds 10000`.

## Disable Plots

```bash
python3 -B scripts/run_color_patch_alignment_experiment.py --no-plots ...
```
