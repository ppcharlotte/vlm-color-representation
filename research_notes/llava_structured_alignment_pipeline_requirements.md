# LLaVA Color Representation Pipeline Requirements

This document records the agreed requirements for restructuring the color representation and color naming experiments. It is intended as the reference document before rebuilding the code.

## 1. Overall Goal

The project will be reorganized into a clear experimental pipeline for studying how a vision-language model represents colors and how these representations relate to color naming performance.

The pipeline should support both independent execution of each analysis step and one-command execution of the full workflow.

Default model:

```text
llava-hf/llava-1.5-7b-hf
```

Default selected layers:

```text
0,1,8,16,24,32
```

Default output directory:

```text
experiments/llava_structured_alignment/
```

## 2. Input Data

Primary input CSV:

```text
data/colorsurvey_rgb_aggregated_new.csv
```

Note: the originally mentioned name was `colorsurvey_rgb_aggregate_new.csv`, but the file currently present in the project is `colorsurvey_rgb_aggregated_new.csv`.

Detected columns:

```text
r, g, b, hex,
total_votes,
majority_basic_color,
majority_count,
agreement_rate,
label_entropy_bits,
num_basic_labels,
basic_label_set,
example_colornames,
count_black, p_black,
count_blue, p_blue,
count_brown, p_brown,
count_gray, p_gray,
count_green, p_green,
count_orange, p_orange,
count_pink, p_pink,
count_purple, p_purple,
count_red, p_red,
count_white, p_white,
count_yellow, p_yellow
```

The current dataset has already been filtered to keep high-agreement colors. No additional `agreement_rate > 0.75` filter is required unless explicitly requested later.

## 3. Basic Color Set

Use the following 11 basic colors:

```text
black, blue, brown, gray, green, orange, pink, purple, red, white, yellow
```

Normalize `grey` to `gray` if it appears in any source data or model output.

For each basic color, sample up to 100 RGB values from `majority_basic_color`. If a color has fewer than 100 samples, keep all available samples.

Color patch size:

```text
224 x 224
```

## 4. Script Structure

The codebase should be rebuilt as separate scripts with clear inputs and outputs. Each script should be runnable independently. A full pipeline runner should also be provided.

Planned scripts:

```text
scripts/color_naming.py
scripts/image_embedding_extraction.py
scripts/text_embedding_extraction.py
scripts/query_embedding_extraction.py
scripts/similarity_analysis.py
scripts/relation_analysis.py
scripts/negative_control.py
scripts/run_llava_structured_pipeline.py
```

## 5. `color_naming.py`

Purpose: build the sampled color patch dataset and evaluate VLM color naming behavior.

Inputs:

- `data/colorsurvey_rgb_aggregated_new.csv`
- basic color list
- model id

Main steps:

1. Read RGB/color metadata.
2. Select up to 100 samples per basic color.
3. Generate 224 x 224 pure color patches.
4. Ask the VLM to name each color patch.
5. Normalize model output to the 11 basic color labels.
6. Compare prediction against ground truth.

Prompt:

```text
What color is this image? Answer only with exactly one color name from this list: black, blue, brown, gray, green, orange, pink, purple, red, white, yellow.
```

Outputs:

- sampled ground-truth CSV
- generated patch images
- sample-level prediction CSV
- overall accuracy
- per-color accuracy / precision / recall / F1 CSV
- confusion matrix CSV
- confusion matrix plot
- plot containing all error cases
- JSON summary for color naming behavior

## 6. `image_embedding_extraction.py`

Purpose: extract visual representations from LLaVA and evaluate whether they preserve perceptual color relationships.

Inputs:

- sampled color patch dataset from `color_naming.py`
- selected layers
- model id
- query prompt condition

Required image representations:

1. Pre-projector image embedding.
2. Post-projector embedding, corresponding to LLM input / layer 0.
3. Selected LLM layer image-token hidden states.

Prompt conditions:

- force-choice query with the color-word list
- no-color-list query

Decision: for rigor, extract LLM image-token hidden states separately for both query conditions. Pre-projector and immediate post-projector representations may still be stored as shared baseline representations if they are identical across query conditions.

RDM outputs:

1. Sample-level image RDM: pairwise distances between all color patches.
2. Class-centroid image RDM: pairwise distances between color centroids.

Perceptual color reference:

1. Convert RGB to CIELAB coordinates.
2. Compute perceptual color distances using CIEDE2000.
3. Build CIELAB/CIEDE2000 sample-level RDM.
4. Build CIELAB/CIEDE2000 class-centroid RDM.

Analyses:

- RSA between image embedding RDM and CIEDE2000 RDM.
- RSA between image centroid RDM and CIEDE2000 centroid RDM.
- Linear regression predicting CIEDE2000 perceptual distances from image embedding distances.

Outputs:

- image embeddings `.npz`
- image RDM CSV files
- CIEDE2000 RDM CSV files
- RSA results CSV
- regression results CSV
- layer-wise plots for image-CIELAB alignment

## 7. `text_embedding_extraction.py`

Purpose: extract color-word embeddings from the LLM component of the VLM.

Contexts:

```text
The item is {color}.
a {color} color patch.
the image is {color}.
```

Main steps:

1. Tokenize each context prompt.
2. Identify the color-word token span.
3. Extract color-word embedding from selected layers when available.
4. Mean-pool multi-token color words if needed.
5. Aggregate across contexts, while also preserving context-specific embeddings.

Outputs:

- text/color-word embeddings `.npz`
- context-specific color-word embeddings
- aggregated color-word embeddings
- color-word RDM CSV
- layer-wise color-word RDM files when applicable

## 8. `query_embedding_extraction.py`

Purpose: extract image-conditioned query embeddings from the color naming prompt.

Two query conditions:

1. Force-choice query with color-word list.
2. No-color-list query.

The no-color-list query should also be evaluated behaviorally, because changing the prompt may change final color naming accuracy. Therefore, color naming results should be available for both prompt conditions.

Main steps:

1. Run the model with image and query prompt.
2. Extract query-token hidden states from selected layers.
3. Pool query-token states into one query representation per sample per layer.
4. Save separate embeddings for force-choice and no-color-list prompts.
5. Build query RDMs for each query condition and layer.

Outputs:

- query embeddings `.npz`
- no-color-list query embeddings `.npz`
- query RDM CSV files
- no-color-list query RDM CSV files
- optional query behavior outputs if not already produced by `color_naming.py`

## 9. `similarity_analysis.py`

Purpose: analyze image/text/query representation similarity and color information in embeddings.

Analyses for selected layers:

Image vs text:

- cosine similarity to correct color word
- pair alignment margin
- image RDM vs text RDM RSA
- image vs text CKA
- Procrustes error where useful

Query vs text:

- cosine similarity to correct color word
- pair alignment margin
- query RDM vs text RDM RSA
- query vs text CKA
- Procrustes error where useful

No-color-list query vs text:

- same metrics as query vs text

Linear probing:

- image embedding linear probe by layer
- query embedding linear probe by layer
- no-color-list query embedding linear probe by layer

Outputs:

- layer-wise similarity/alignment CSV files
- linear probe CSV files
- per-color linear probe metrics
- layer-wise plots for cosine similarity, pair margin, RSA, CKA, Procrustes error, and linear probe accuracy

## 10. `relation_analysis.py`

Purpose: test whether representation metrics are associated with final color naming behavior.

Behavior targets:

- sample-level final correctness
- class-level final accuracy

Predictors:

- correct-color cosine similarity
- pair alignment margin
- RDM alignment / RSA
- CKA
- Procrustes error
- linear probe confidence / accuracy when available
- image-CIELAB RSA or regression scores when relevant

Statistical analyses:

- sample-level Spearman/Pearson correlations
- class-level Spearman/Pearson correlations
- permutation p-values
- sample-level logistic regression predicting correct vs incorrect color naming
- class-level linear regression predicting per-color accuracy

Outputs:

- sample-level relation CSV
- class-level relation CSV
- regression result CSV/JSON
- layer-wise relation plots
- summary entries for best layer and strongest predictors

## 11. `negative_control.py`

Purpose: test whether observed similarity/alignment effects are specific to correct color correspondences.

Negative controls:

1. Randomly shuffle color-word labels and recompute similarity/alignment.
2. Use label-permuted text embeddings and test whether predictive relationships disappear.
3. Use wrong-color similarity and test whether it fails to positively predict correctness or predicts errors instead.
4. Compare correct-label alignment against shuffled/permuted baselines.

Outputs:

- negative-control CSV files
- permutation baseline distributions
- plots comparing correct vs shuffled/permuted conditions
- summary JSON entries

## 12. Full Pipeline Runner

`run_llava_structured_pipeline.py` should orchestrate the full workflow:

1. `color_naming.py`
2. `image_embedding_extraction.py`
3. `text_embedding_extraction.py`
4. `query_embedding_extraction.py`
5. `similarity_analysis.py`
6. `relation_analysis.py`
7. `negative_control.py`
8. final summary aggregation

The pipeline should expose core arguments:

```text
--input-csv
--output-dir
--model-id
--layer-indices
--samples-per-color
--patch-size
--device
--dtype
--batch-size
--run-negative-control
--reuse-existing-embeddings
```

## 13. Output Directory Structure

All outputs should be stored under:

```text
experiments/llava_structured_alignment/
```

Proposed structure:

```text
experiments/llava_structured_alignment/
  data/
    selected_color_samples.csv
    ground_truth.csv
    patches/
  color_naming/
    force_choice_predictions.csv
    no_color_list_predictions.csv
    class_metrics.csv
    confusion_matrix.csv
    summary.json
  embeddings/
    image_embeddings.npz
    image_embeddings_force_choice.npz
    image_embeddings_no_color_list.npz
    text_embeddings.npz
    query_embeddings_force_choice.npz
    query_embeddings_no_color_list.npz
  rdm/
    image_sample_rdms/
    image_centroid_rdms/
    text_rdms/
    query_rdms/
    cielab_sample_rdm.csv
    cielab_centroid_rdm.csv
  similarity/
    image_text_layerwise.csv
    query_text_layerwise.csv
    query_no_color_list_text_layerwise.csv
  linear_probe/
    image_layerwise_probe.csv
    query_layerwise_probe.csv
    query_no_color_list_layerwise_probe.csv
    per_color_probe_metrics.csv
  relation/
    sample_level_relations.csv
    class_level_relations.csv
    regression_results.json
  negative_control/
    shuffled_label_results.csv
    wrong_color_similarity_results.csv
    permutation_baselines.csv
  plots/
    confusion_matrix.png
    error_cases.png
    layerwise_*.png
  summary.json
```

## 14. Final Summary JSON

The final `summary.json` should include:

- model id
- input CSV path
- sampled dataset size
- samples per color
- prompt conditions
- selected layers
- color naming accuracy for each prompt condition
- per-color accuracy highlights
- strongest image-CIELAB RSA/regression results
- strongest image-text similarity/alignment results
- strongest query-text similarity/alignment results
- linear probe best layers
- relation-analysis best predictors
- negative-control results
- paths to all important CSV and plot files

## 15. Confirmed Decisions

- Include `orange` in the basic color list.
- Remove duplicated `green` from the originally written color list.
- Use `gray` as the canonical label; normalize `grey` to `gray`.
- Use 224 x 224 pure color patches.
- Save generated color patches as `.png` files.
- Use a fixed random seed for sampling. Default seed: `42`.
- Use standard sRGB to CIELAB conversion with D65 illuminant.
- Use `llava-hf/llava-1.5-7b-hf` as the default VLM.
- Use layers `0,1,8,16,24,32` by default.
- Use mean pooling by default for image tokens, query tokens, and multi-token color-word spans.
- Extract image-token LLM hidden states separately for force-choice and no-color-list query conditions for rigor.
- Scripts should be independently runnable and also callable through one full pipeline runner.
- Regression analysis should include correlation analyses, sample-level logistic regression, and class-level linear regression. Mixed-effects models are not required for the first implementation.
- Existing scripts should be preserved rather than deleted. To avoid confusion, old experiment scripts should be moved into a `wip` folder before creating the new modular pipeline.

## 16. Remaining Implementation Notes

- Use the following prompt conditions:

Force-choice prompt:

```text
What color is this image? Answer only with exactly one color name from this list: black, blue, brown, gray, green, orange, pink, purple, red, white, yellow.
```

No-color-list prompt:

```text
What color is this image? Answer only with exactly one basic color name.
```

- The exact pooling strategy should be explicit for every representation:
  - image tokens: mean pooling by default
  - query tokens: mean pooling by default
  - color-word tokens: mean pooling over color-token span
- The pipeline should cache embeddings to avoid repeated expensive LLaVA inference.
- Existing scripts can be mined for reusable code, but output naming should be made cleaner and more modular.
