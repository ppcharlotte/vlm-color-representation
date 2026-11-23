# Color Encoding in Multimodal Language Models

Analyze how vision-language models represent color and how those representations align with perceptual color geometry.

The pipeline generates solid-color patches from human color-survey data, extracts representations from pretrained LLaVA models, and compares them using representational similarity analysis (RSA). It runs inference and statistical analysis without training or fine-tuning.

## What the Code Does

1. Filter RGB samples by human naming agreement (`agreement_rate > 0.75` and `majority_count >= 2`) across 11 basic color categories.
2. Generate image patches and extract visual, text-only, and multimodal color representations across model layers.
3. Compare encoding-side color-word embeddings (CEe) and generation-side embeddings (CEg), including text-only and mismatched-image controls.
4. Compute sample- and category-level RSA against CIEDE2000 color distances, then export tables and plots.

Human labels define the categories; the perceptual reference is computed from RGB values in CIELAB space, rather than directly measured human similarity judgments.

## Installation

Use Python 3.10–3.12. A CUDA GPU with enough memory for the selected model is recommended.

```bash
git clone https://github.com/ppcharlotte/vlm-color-representation.git
cd vlm-color-representation

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-generative.txt scipy
```

`scipy` is required by the analysis code but is currently missing from the requirements files. The first inference run downloads model weights from Hugging Face.

## Quick Start

Run all commands from the repository root. The input dataset is included at `data/colorsurvey_rgb_aggregated_new.csv`.

### Check dataset generation

This step does not load a model:

```bash
python scripts/build_perceptual_patch_dataset.py \
  --samples-per-color 2 \
  --output-dir experiments/data_check
```

### Run a small experiment

```bash
python scripts/run_llava_structured_pipeline.py \
  --model-id llava-hf/llava-1.5-7b-hf \
  --device cuda \
  --dtype float16 \
  --samples-per-color 2 \
  --layer-indices 0,16,32 \
  --sample-permutation-rounds 10 \
  --category-permutation-rounds 10 \
  --skip-cee \
  --skip-ceg \
  --no-plots \
  --output-dir experiments/smoke_llava15
```

This checks the visual/text extraction and analysis workflow while still loading the full model. The layer indices apply to LLaVA-1.5-7B; index `0` is the language model's input hidden state. The small permutation budgets are for testing only.

### Run the full experiment

```bash
python scripts/run_llava_structured_pipeline.py \
  --model-id llava-hf/llava-1.5-7b-hf \
  --device cuda \
  --dtype float16 \
  --samples-per-color 100 \
  --layer-indices all \
  --seed 42 \
  --mismatch-repeats 5 \
  --sample-permutation-rounds 1000 \
  --category-permutation-rounds 10000 \
  --output-dir experiments/llava15_seed42
```

This includes CEe, CEg, mismatched-image controls, and plots. Each category contributes at most 100 samples; strict filtering may leave fewer. Use a separate output directory when changing the model, sampling seed, or experimental settings.

For all available options:

```bash
python scripts/run_llava_structured_pipeline.py --help
```

## Rerun Analysis

Reuse an existing run's dataset and embedding caches without loading the model:

```bash
python scripts/perceptual_alignment_analysis.py \
  --ground-truth experiments/llava15_seed42/data/ground_truth.csv \
  --dataset-root experiments/llava15_seed42/data \
  --output-dir experiments/llava15_seed42 \
  --sample-permutation-rounds 1000 \
  --category-permutation-rounds 10000 \
  --seed 42
```

This overwrites the corresponding analysis results. The main pipeline's `--reuse-dataset` flag only skips dataset construction, not embedding extraction.

## Outputs

All results are saved under `--output-dir`:

| Path | Contents |
| --- | --- |
| `data/` | Image patches, sample labels, and sampling metadata |
| `embeddings/` | Cached representations in NPZ format |
| `encoding_side/` | CEe prompts and mismatched-image assignments |
| `ceg_generation_side/predictions.csv` | Generated answers and sample inclusion status |
| `analysis/alignment_layerwise_v2.csv` | Main layer-wise RSA results |
| `analysis/sample_support_v2.csv` | Samples included in each comparison |
| `analysis/summary.json` | Analysis settings and completion status |
| `plots/v2/` | Result figures |
| `summary.json` | Overall pipeline summary |

CEg analysis includes only samples with successfully parsed color tokens and available hidden states. Check sample inclusion before comparing results across conditions.

## License

The original code and documentation in this repository are licensed under the [MIT License](LICENSE). Third-party survey data, model weights, and dependencies remain subject to their respective licenses.
