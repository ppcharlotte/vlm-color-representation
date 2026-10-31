"""Numerical and integration checks for template/seed/contrast analyses."""
import csv
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import numpy as np

from alignment_statistics import (distance_regression, paired_category_swap, mean_rsa_permutation,
                                  rdm_alignment, stable_seed)
from analysis_extensions import template_subsets, template_robustness, visual_equivalence
from extended_alignment_analysis import run_extended_analysis, write_rows
from plot_randomness_summary import aggregate_runs, category_pair, plot_aggregates
from structured_color_utils import BASIC_COLORS, TEXT_CONTEXTS, rdm, rgb_to_lab, save_matrix_csv
from test_alignment_v2 import fixture
from analysis_plotting import plot_all_results
from generation_context_embedding_analysis import generation_image_states, prefill_image_positions
from test_onevision_projector_extraction import FakeTensor


class AnalysisUpdateTests(unittest.TestCase):
    def test_actual_generation_prefill_raw_pooling_and_mask_expansion(self):
        values = np.array([[[90, 90], [2, 4], [4, 8], [80, 80], [70, 70]]], float)
        steps = [(FakeTensor(values), FakeTensor(values * 2))]
        for mask in ([False, True, True, False, False], [False, True, False, False]):
            helper = SimpleNamespace(image_token_mask=lambda ids: FakeTensor(mask))
            result = generation_image_states(helper, {"input_ids": None}, steps, [0, 1])
            np.testing.assert_allclose(result, [[3, 6], [6, 12]])
        with self.assertRaises(ValueError):
            prefill_image_positions([False, True, True, False], 6)

    def test_new_output_families_render(self):
        with TemporaryDirectory() as temp:
            output = Path(temp)
            common = dict(model_id="synthetic", layer_index=2, source_a="image_tokens", source_b="perceptual",
                          level="category", embedding_distance="cosine", rsa_coefficient="spearman", q_fdr=.02)
            regression = [dict(common, R2=.3)]
            templates = [dict(common, subset_a="balanced_A", subset_b="perceptual", comparison_type="subset_vs_perceptual", alignment=.4)]
            paired = [dict(common, contrast="cee_image_vs_no_image", reference="image_tokens", delta_rsa=.1, q_delta_fdr=.04)]
            sanity = [dict(layer_index=2, condition_a="visual_only", condition_b="generation",
                           mean_cosine_similarity=1, min_cosine_similarity=1, max_abs_difference=0, mean_abs_difference=0)]
            manifest = plot_all_results(output, [], [], [], [], [], regression_rows=regression,
                template_rows=templates, paired_rows=paired, sanity_rows=sanity, rdm_paths=[])
            self.assertEqual(len(manifest), 7)
            self.assertTrue(all(Path(row["plot_path"]).stat().st_size > 1000 for row in manifest))

    def test_regression_affine_and_undefined(self):
        x = rdm(np.random.default_rng(1).normal(size=(7, 9)))
        y = 4 * x + 3
        np.fill_diagonal(y, 0)
        result = distance_regression(x, y)
        self.assertAlmostEqual(result["R2"], 1)
        self.assertAlmostEqual(result["slope"], 4)
        self.assertAlmostEqual(result["intercept"], 3)
        self.assertEqual(result["n_pairs"], 21)
        self.assertNotIn("p_value", result)
        self.assertTrue(np.isnan(distance_regression(x * 0, y)["R2"]))

    def test_exact_swap_matches_enumeration_and_sign_reversal(self):
        rng = np.random.default_rng(4)
        a, b = rng.normal(size=(4, 7)), rng.normal(size=(4, 7))
        reference = rdm(rng.normal(size=(4, 6)))
        for coefficient in ("spearman", "kendall_tau_b"):
            result = paired_category_swap(a, b, reference, "cosine", coefficient, 100, 2)
            observed = rdm_alignment(rdm(a), reference, coefficient) - rdm_alignment(rdm(b), reference, coefficient)
            values = []
            for bits in product((False, True), repeat=4):
                mask = np.asarray(bits)[:, None]
                first, second = np.where(mask, b, a), np.where(mask, a, b)
                values.append(rdm_alignment(rdm(first), reference, coefficient) - rdm_alignment(rdm(second), reference, coefficient))
            expected = np.mean(np.abs(values) >= abs(observed) - 1e-14)
            self.assertEqual(result["permutation_rounds"], 16)
            self.assertEqual(result["p_value"], expected)
            reverse = paired_category_swap(b, a, reference, "cosine", coefficient, 100, 2)
            self.assertAlmostEqual(reverse["delta_rsa"], -result["delta_rsa"])
            self.assertAlmostEqual(reverse["p_value"], result["p_value"])
            same = paired_category_swap(a, a, reference, "cosine", coefficient, 100, 2)
            self.assertEqual(same["delta_rsa"], 0)
            self.assertEqual(same["p_value"], 1)

    def test_shared_seed_permutation_uses_one_order(self):
        rng = np.random.default_rng(9)
        pairs = [(rdm(rng.normal(size=(5, 7))), rdm(rng.normal(size=(5, 7)))) for _ in range(3)]
        for coefficient in ("spearman", "kendall_tau_b"):
            mean, p, status = mean_rsa_permutation(pairs, coefficient, 39, 71)
            expected_mean = np.mean([rdm_alignment(a, b, coefficient) for a, b in pairs])
            orders = np.random.default_rng(71)
            null = []
            for _ in range(39):
                order = orders.permutation(5)
                null.append(np.mean([rdm_alignment(a, b[np.ix_(order, order)], coefficient) for a, b in pairs]))
            self.assertEqual(status, "ok")
            self.assertAlmostEqual(mean, expected_mean)
            self.assertEqual(p, (1 + np.count_nonzero(np.abs(null) >= abs(mean) - 1e-14)) / 40)

    def test_template_balance_and_cached_pooling(self):
        subsets = template_subsets(TEXT_CONTEXTS)
        self.assertEqual(sorted(sum(subsets.values(), [])), list(range(30)))
        for indices in subsets.values():
            self.assertEqual(len(indices), 15)
            self.assertEqual(sum(i < 10 for i in indices), 5)
        with TemporaryDirectory() as temp:
            output = Path(temp)
            data = fixture(output)
            rows, membership, status = template_robustness(output, rgb_to_lab(data.rgb), data.labels,
                [0, 2], "synthetic", 3, 9, ["spearman", "kendall_tau_b"])
            self.assertEqual(status, "computed")
            self.assertEqual(len(rows), 20)
            self.assertEqual(len(membership), 30)
            with np.load(output / "embeddings/text_embeddings.npz") as saved:
                prototypes = saved["context_text_layer_embeddings"][0][:, subsets["balanced_A"]].mean(1)
                expected = rdm_alignment(rdm(prototypes), rdm(saved["text_layer_embeddings"][0]), "spearman")
            actual = next(r for r in rows if r["layer_index"] == 0 and r["subset_a"] == "balanced_A"
                          and r["subset_b"] == "full_30" and r["rsa_coefficient"] == "spearman")
            self.assertAlmostEqual(expected, actual["alignment"])

    def test_visual_raw_difference_is_not_hidden_by_normalization(self):
        with TemporaryDirectory() as temp:
            output = Path(temp)
            data = fixture(output)
            path = output / "embeddings/cee_ground_truth_embeddings.npz"
            with np.load(path) as cache:
                values = {key: cache[key] for key in cache.files}
            values["cee_raw_image_layer_embeddings"] *= 2
            np.savez(path, **values)
            rows, missing = visual_equivalence(output, data, [0, 2], "synthetic", 42)
            self.assertEqual(missing, [])
            comparison = next(r for r in rows if r["condition_b"] == "cee")
            self.assertAlmostEqual(comparison["mean_cosine_similarity"], 1)
            self.assertGreater(comparison["max_abs_difference"], .1)

    def test_integration_metric_scope_and_contrast_metadata(self):
        with TemporaryDirectory() as temp:
            output = Path(temp)
            fixture(output)
            args = SimpleNamespace(output_dir=output, ground_truth=output / "data/ground_truth.csv",
                dataset_root=output, exclude_cee=False, exclude_ceg=False, embedding_distances="cosine",
                rsa_coefficients="spearman,kendall_tau_b", sample_permutation_rounds=9,
                category_permutation_rounds=9, seed=42, no_plots=True, cee_reference_difference=True)
            run_extended_analysis(args)
            def read(name):
                with (output / "analysis" / name).open() as handle:
                    return list(csv.DictReader(handle))
            self.assertFalse((output / "analysis/cka_layerwise_v2.csv").exists())
            self.assertFalse((output / "analysis/appendix_linear_probe_v2.csv").exists())
            regression = read("perceptual_distance_regression_v2.csv")
            self.assertTrue(regression)
            self.assertTrue(all("perceptual" in (r["source_a"], r["source_b"]) for r in regression))
            self.assertTrue(all("p_value" not in r for r in regression))
            rsa = read("alignment_layerwise_v2.csv")
            for row in rsa:
                if row["source_a"].startswith("ceg_") or row["source_b"].startswith("ceg_"):
                    self.assertTrue({row["source_a"], row["source_b"]}.intersection({"image_tokens", "text_color_words", "perceptual"}))
            contrasts = read("paired_rsa_differences_v2.csv")
            self.assertEqual(len(contrasts), 20)
            for row in contrasts:
                self.assertEqual(int(row["n_samples"]), 22 if row["contrast"] == "generation_post_vs_pre" else 33)
                self.assertIn("q_delta_fdr", row)
                self.assertIn("permutation_seed", row)
            generation = read("generation_pre_post_differences_v2.csv")
            self.assertEqual(len(generation), 4)
            self.assertTrue(all(float(r["delta_visual"]) == 0 and float(r["p_delta_visual"]) == 1 for r in generation))
            self.assertEqual(len(read("visual_state_equivalence_v2.csv")), 6)
            self.assertEqual(len(read("text_context_robustness_v2.csv")), 20)

    def test_three_run_csv_aggregation_ddof_and_descriptive_samples(self):
        rng = np.random.default_rng(28)
        with TemporaryDirectory() as temp:
            dirs, values = [], []
            for index in range(3):
                run = Path(temp) / str(index)
                dirs.append(run)
                a, b = rdm(rng.normal(size=(11, 7))), rdm(rng.normal(size=(11, 7)))
                value = rdm_alignment(a, b, "spearman")
                values.append(value)
                order = rng.permutation(11)
                names = [BASIC_COLORS[i] for i in order]
                save_matrix_csv(run / "rdm/a.csv", names, a[np.ix_(order, order)])
                save_matrix_csv(run / "rdm/b.csv", names, b[np.ix_(order, order)])
                row = dict(model_id="synthetic", source_a="image_tokens", source_b="text_color_words",
                    level="category", embedding_distance="cosine", rsa_coefficient="spearman", layer_index=2,
                    source_a_layer=2, source_b_layer=2, alignment=value, seed=index,
                    rdm_a_csv="rdm/a.csv", rdm_b_csv="rdm/b.csv", p_value=.001, q_fdr=.001)
                write_rows(run / "analysis/alignment_layerwise_v2.csv", [row, dict(row, level="sample")])
            rows = aggregate_runs(dirs, rounds=19, seed=8)
            category = next(r for r in rows if r["level"] == "category")
            sample = next(r for r in rows if r["level"] == "sample")
            self.assertEqual(category["n_runs"], 3)
            self.assertAlmostEqual(category["mean_alignment"], np.mean(values))
            self.assertAlmostEqual(category["sd_alignment"], np.std(values, ddof=1))
            self.assertGreaterEqual(category["mean_perm_p"], .05)
            self.assertTrue(np.isnan(sample["mean_perm_p"]))
            self.assertEqual(sample["status"], "sample_level_descriptive_only")
            plot_aggregates(Path(temp) / "summary_plots", rows)
            self.assertEqual(len(list((Path(temp) / "summary_plots").glob("*.png"))), 2)
            with self.assertRaisesRegex(ValueError, "same run"):
                aggregate_runs([dirs[0], dirs[0]])


if __name__ == "__main__":
    unittest.main()
