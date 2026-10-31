"""CPU-only numerical and synthetic end-to-end regression tests (no checkpoints)."""
from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau

from alignment_statistics import compare_rdms, paired_category_bootstrap, kendall_tau_b
from extended_alignment_analysis import load_sources, run_extended_analysis
from structured_color_utils import BASIC_COLORS, TEXT_CONTEXTS, Dataset, LlavaHelper, rdm, rgb_to_lab, write_csv
from mismatched_color_control import mismatched_assignments
from analysis_plotting import plot_all_results, _fixed_axis


def fixture(output, onevision=False):
    rng = np.random.default_rng(182)
    n, d, layers = 33, 8, [0, 2]
    labels = np.repeat(BASIC_COLORS, 3).tolist()
    rgb = rng.integers(20, 230, size=(n, 3))
    rows = [dict(sample_index=i, basic_color=labels[i], r=rgb[i, 0], g=rgb[i, 1], b=rgb[i, 2],
                 image_path=f"{i}.png", absolute_image_path=str(output / f"{i}.png")) for i in range(n)]
    write_csv(output / "data/ground_truth.csv", list(rows[0]), rows)
    paths = np.asarray([r["absolute_image_path"] for r in rows])
    embeddings = output / "embeddings"
    embeddings.mkdir()
    text = rng.normal(size=(2, 11, d))
    context = text[:, :, None, :] + rng.normal(scale=.2, size=(2, 11, 30, d))
    context /= np.linalg.norm(context, axis=-1, keepdims=True)
    text = context.mean(axis=2)
    text /= np.linalg.norm(text, axis=-1, keepdims=True)
    visual = rng.normal(size=(2, n, d))
    matched = np.repeat(text, 3, axis=1)
    np.savez(embeddings / "text_embeddings.npz", text_layer_embeddings=text, layer_indices=layers,
             colors=BASIC_COLORS, model_id="synthetic", context_text_layer_embeddings=context, contexts=TEXT_CONTEXTS)
    np.savez(embeddings / "image_query_embeddings.npz", image_layer_embeddings=visual,
             query_layer_embeddings=visual + 0.1, pre_projector=visual[0, :, :5], post_projector=visual[0],
             layer_indices=layers, image_paths=paths, is_onevision=int(onevision),
             projector_capture_version=2, model_id="synthetic", pooling="mean")
    np.savez(embeddings / "cee_ground_truth_embeddings.npz", cee_layer_embeddings=visual + matched,
             matched_text_layer_embeddings=matched, cee_image_layer_embeddings=visual,
             cee_raw_image_layer_embeddings=visual, image_pooling="mean",
             layer_indices=layers, image_paths=paths, labels=labels, matched_design_version=1,
             template="The color of this image is {color}", model_id="synthetic")
    mappings = mismatched_assignments(labels, 2, 183)
    np.savez(embeddings / "cee_mismatched_embeddings.npz",
             mismatched_layer_embeddings=np.stack([visual[:, m] + matched for m in mappings]),
             donor_indices=mappings, layer_indices=layers, image_paths=paths, labels=labels,
             model_id="synthetic", template="The color of this image is {color}")
    gen = output / "ceg_generation_side/embeddings"
    gen.mkdir(parents=True)
    np.savez(gen / "generation_image_embeddings.npz", image_layer_embeddings=visual,
             layer_indices=layers, image_paths=paths, model_id="synthetic", pooling="mean")
    ids = np.arange(n)[np.arange(n) % 3 != 2]
    for filename, key in (("ceg_decision_embeddings.npz", "ceg_decision_layer_embeddings"),
                          ("ceg_token_from_generate_embeddings.npz", "ceg_token_from_generate_layer_embeddings")):
        np.savez(gen / filename, **{key: visual[:, ids]}, sample_indices=ids, layer_indices=layers,
                 human_categories=np.asarray(labels)[ids], image_paths=paths[ids], model_id="synthetic")
    return Dataset(rows)


class StatisticsTests(unittest.TestCase):
    def test_fixed_plot_axis_does_not_claim_lm_layer_zero(self):
        self.assertTrue(_fixed_axis([{"layer_index": 0, "source_a_layer": "fixed", "source_b_layer": "fixed"}]))
        self.assertTrue(_fixed_axis([{"layer_index": "fixed"}]))
        self.assertFalse(_fixed_axis([{"layer_index": 0, "source_a_layer": "fixed", "source_b_layer": 0}]))

    def test_requested_plots_fail_explicitly_without_matplotlib(self):
        import analysis_plotting
        with TemporaryDirectory() as temp, patch.object(analysis_plotting, "try_import_matplotlib", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Matplotlib"):
                plot_all_results(Path(temp), [], [], [], [], [])

    def test_all_numeric_output_families_are_plotted(self):
        import analysis_plotting
        with TemporaryDirectory() as temp:
            output = Path(temp)
            write_csv(output / "rdm/category_v2/example.csv", ["color", "red", "blue"],
                      [{"color": "red", "red": 0, "blue": 1}, {"color": "blue", "red": 1, "blue": 0}])
            common = dict(model_id="synthetic", layer_index=0, source_a="image_tokens", source_b="text_color_words",
                          alignment=.4, p_value=.01, q_fdr=.04)
            rsa = [dict(common, level="category", embedding_distance="cosine", rsa_coefficient="spearman")]
            cka = [dict(common)]
            delta = [dict(common, source_a="cee", source_b="matched_text", mismatch_repeat=None,
                          embedding_distance="cosine", rsa_coefficient="spearman", delta_alignment=.2,
                          alignment_a=.6, alignment_b=.4, ci_low=.1, ci_high=.3)]
            mismatch = [dict(model_id="synthetic", layer_index=0, contrast="correct_minus_mismatched",
                             embedding_distance="cosine", rsa_coefficient="spearman", mean_delta=.1, sd_delta=.02)]
            probes = [dict(model_id="synthetic", source="pre_projector", layer_index="fixed", accuracy=.5)]
            def fake_line(path, *args, **kwargs):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                return True
            def fake_rdm(csv_path, output_root):
                path = output_root / "plots/v2/rdm" / csv_path.relative_to(output_root / "rdm").with_suffix(".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
                return path, 2
            if analysis_plotting.try_import_matplotlib() is not None:
                manifest = plot_all_results(output, rsa, cka, delta, mismatch, probes)
            else:
                with patch.object(analysis_plotting, "try_import_matplotlib", return_value=object()), \
                     patch.object(analysis_plotting, "_line_plot", side_effect=fake_line), \
                     patch.object(analysis_plotting, "_mismatch_plot", side_effect=fake_line), \
                     patch.object(analysis_plotting, "_rdm_plot", side_effect=fake_rdm):
                    manifest = plot_all_results(output, rsa, cka, delta, mismatch, probes)
            self.assertEqual({row["plot_type"] for row in manifest},
                             {"rsa_layerwise", "cka_layerwise", "rq3_alignment_and_delta",
                              "mismatch_repeat_summary", "linear_probe_layerwise", "rdm_heatmap"})
            self.assertTrue(all(Path(row["plot_path"]).exists() for row in manifest))
            self.assertEqual(next(row for row in manifest if row["plot_type"] == "rdm_heatmap")["n_rows"], 2)
            self.assertTrue((output / "plots/v2/plot_manifest.csv").exists())

    def test_mismatch_assignments(self):
        labels = ["red"] * 4 + ["blue"] * 3 + ["green"] * 3
        mappings = mismatched_assignments(labels, 5, 42)
        np.testing.assert_array_equal(mappings, mismatched_assignments(labels, 5, 42))
        for mapping in mappings:
            np.testing.assert_array_equal(np.sort(mapping), np.arange(len(labels)))
            self.assertTrue(np.all(np.asarray(labels) != np.asarray(labels)[mapping]))
        self.assertGreater(len({tuple(m) for m in mappings}), 1)
        with self.assertRaises(ValueError):
            mismatched_assignments(["red"] * 5 + ["blue"], 1)
        self.assertEqual(mismatched_assignments(labels, 0).shape, (0, 10))

    def test_mismatch_extraction_uses_target_words(self):
        import context_embedding_encoding_analysis as extraction
        with TemporaryDirectory() as temp:
            output = Path(temp)
            dataset = fixture(output)
            args = SimpleNamespace(output_dir=output, ground_truth=output / "data/ground_truth.csv",
                dataset_root=output, layer_indices="0,2", model_id="synthetic", device="cpu", dtype="float32",
                template="The color of this image is {color}", seed=42, skip_extraction=True,
                mismatch_repeats=2, mismatch_seed=42)
            calls = []
            def fake_extract(helper, paths, labels, layers, template):
                calls.append((paths, labels))
                self.assertEqual(labels, dataset.labels)
                donor_ids = [int(Path(p).stem) for p in paths]
                self.assertTrue(all(dataset.labels[d] != labels[i] for i, d in enumerate(donor_ids)))
                prompts = [dict(statement=template.format(color=c), expanded_prompt=template.format(color=c), n_color_tokens=1) for c in labels]
                values = np.ones((2, len(labels), 8), np.float32)
                return values, values, values, prompts
            with patch.object(extraction, "parse_args", return_value=args), patch.object(extraction, "LlavaHelper", return_value=object()), patch.object(extraction, "extract_cee_layers", side_effect=fake_extract):
                extraction.main()
            self.assertEqual(len(calls), 2)
            with np.load(output / "embeddings/cee_mismatched_embeddings.npz") as data:
                self.assertEqual(data["mismatched_layer_embeddings"].shape, (2, 2, 33, 8))
            with (output / "encoding_side/cee_mismatched_pairings.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 66)
            self.assertTrue(all(r["target_color"] != r["donor_color"] for r in rows))
            # Reusing image paths must not allow stale target-word labels.
            baseline = output / "embeddings/cee_ground_truth_embeddings.npz"
            with np.load(baseline) as data:
                saved = {key: data[key] for key in data.files}
            saved["labels"] = np.roll(saved["labels"], 1)
            np.savez(baseline, **saved)
            with patch.object(extraction, "parse_args", return_value=args):
                with self.assertRaisesRegex(ValueError, "Cached CEe"):
                    extraction.main()

    def test_pearson_distance(self):
        x = np.random.default_rng(1).normal(size=(9, 7))
        np.testing.assert_allclose(rdm(x, "pearson"), squareform(pdist(x, "correlation")), atol=1e-12)
        np.testing.assert_allclose(rdm(x, "pearson"), rdm(x + np.arange(9)[:, None], "pearson"), atol=1e-12)
        self.assertTrue(np.isnan(rdm(np.ones((3, 4)), "pearson")[0, 1]))

    def test_tau_ties_and_permutations(self):
        a, b = [1, 1, 3, 2], [4, 2, 2, 1]
        self.assertAlmostEqual(kendall_tau_b(a, b), kendalltau(a, b).statistic)
        matrix = rdm(np.random.default_rng(2).normal(size=(7, 5)))
        for coefficient in ("spearman", "kendall_tau_b"):
            first = compare_rdms(matrix, matrix, coefficient, 99, 42)
            self.assertAlmostEqual(first[0], 1)
            self.assertGreaterEqual(first[1], 0.01)
            self.assertEqual(first, compare_rdms(matrix, matrix, coefficient, 99, 42))
            self.assertEqual(compare_rdms(matrix * 0, matrix, coefficient, 9, 0)[3], "constant_or_insufficient_rdm")

    def test_paired_bootstrap(self):
        rng = np.random.default_rng(6)
        labels = np.repeat(BASIC_COLORS[:4], 5).tolist()
        x, y = rng.normal(size=(20, 6)), rng.normal(size=(20, 6))
        lab = rgb_to_lab(rng.integers(0, 255, (20, 3)))
        args = (lab, labels, BASIC_COLORS[:4], ["cosine", "pearson"], ["spearman", "kendall_tau_b"], 40, 4)
        same = paired_category_bootstrap(x, x, *args)
        for row in same:
            self.assertEqual(row["delta_alignment"], 0)
            self.assertEqual(row["p_value"], 1)
        forward, reverse = paired_category_bootstrap(x, y, *args), paired_category_bootstrap(y, x, *args)
        for a, b in zip(forward, reverse):
            self.assertAlmostEqual(a["delta_alignment"], -b["delta_alignment"])
            self.assertAlmostEqual(a["ci_low"], -b["ci_high"])
            self.assertAlmostEqual(a["p_value"], b["p_value"])

    def test_prompt_pair(self):
        helper = object.__new__(LlavaHelper)
        helper.processor = object()
        statement = "The color of this image is red"
        with_image = helper.chat_prompt(statement)
        without_image = helper.chat_prompt(statement, with_image=False)
        self.assertEqual(with_image.replace("<image>\n", ""), without_image)
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return repr(messages)
        helper.processor = Processor()
        helper.chat_prompt(statement, with_image=False)
        self.assertEqual(helper.processor.messages[0]["content"], [{"type": "text", "text": statement}])

    def test_pipeline_csv_coverage_and_support(self):
        for onevision in (False, True):
            with self.subTest(onevision=onevision), TemporaryDirectory() as temp:
                output = Path(temp)
                dataset = fixture(output, onevision)
                sources, layers = load_sources(output, dataset)
                self.assertEqual(layers, [0, 2])
                self.assertEqual({s.name for s in sources if s.fixed}, {"pre_projector", "post_projector"})
                args = SimpleNamespace(output_dir=output, ground_truth=output / "data/ground_truth.csv",
                    dataset_root=output, exclude_cee=False, exclude_ceg=False, embedding_distances="cosine,pearson",
                    rsa_coefficients="spearman,kendall_tau_b", sample_permutation_rounds=9,
                    category_permutation_rounds=9, bootstrap_rounds=20, seed=42,
                    linear_probe_test_size=0.3, ridge_alpha=1.0, no_plots=True)
                args.appendix_metrics = True
                run_extended_analysis(args)
                def rows(filename):
                    with (output / "analysis" / filename).open() as stream:
                        return list(csv.DictReader(stream))
                result = rows("alignment_layerwise_v2.csv")
                supports = {r["support_id"]: r for r in rows("sample_support_v2.csv")}
                for row in result:
                    self.assertEqual(int(row["n_samples"]), len(supports[row["support_id"]]["sample_indices"].split("|")))
                    self.assertIn("q_fdr", row)
                    if row["status"] == "ok":
                        self.assertGreaterEqual(float(row["q_fdr"]), float(row["p_value"]) - 1e-12)
                stage_text = [r for r in result if r["source_a"] == "text_color_words" and r["source_b"] == "pre_projector"]
                self.assertEqual({r["layer_index"] for r in stage_text}, {"0", "2"})
                self.assertEqual(len(stage_text), 8)
                self.assertTrue(all(r["source_b_layer"] == "fixed" for r in stage_text))
                pairs = [r for r in result if "ceg_token" in (r["source_a"], r["source_b"]) and r["source_a"] == "image_tokens"]
                self.assertTrue(pairs)
                self.assertTrue(all(int(r["n_samples"]) == 22 for r in pairs))
                delta = rows("rq3_permutation_v2.csv")
                self.assertEqual(len(delta), 40)
                self.assertTrue(all("permutation_rounds" in r and "q_fdr" in r for r in delta))
                wrong_delta = [r for r in delta if r["source_a"] == "cee" and r["source_b"].startswith("cee_mismatched")]
                self.assertEqual(len(wrong_delta), 16)
                self.assertTrue(all(r["pairing_scope"] == "fixed_assignment" for r in wrong_delta))
                self.assertEqual(len(rows("rq3_mismatch_repeat_summary.csv")), 16)
                fixed = [r for r in result if r["source_a"] == "pre_projector" and r["source_b"] == "post_projector"]
                self.assertEqual({r["layer_index"] for r in fixed}, {"0"})
                cka = rows("appendix_cka_layerwise_v2.csv")
                self.assertTrue(all(r["human_reference"] == "CIELAB_coordinates" for r in cka if r["source_b"] == "perceptual"))
                # Exclusion flags must not revive old caches on a partial run.
                subset, _ = load_sources(output, dataset, include_cee=False, include_ceg=False)
                self.assertNotIn("cee", {s.name for s in subset})
                self.assertNotIn("ceg_token", {s.name for s in subset})
                self.assertFalse(any(s.mismatch_repeat is not None for s in subset))
                subset, _ = load_sources(output, dataset, include_mismatched=False)
                self.assertFalse(any(s.mismatch_repeat is not None for s in subset))


if __name__ == "__main__":
    unittest.main()
