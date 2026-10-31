"""CPU checks for all-LM-layer defaults and propagation (no checkpoints)."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from layer_selection import resolve_layers, language_layer_count
import run_llava_structured_pipeline as runner


class LayerSelectionTests(unittest.TestCase):
    def test_language_depth_not_vision_depth(self):
        for depth in (32, 36):
            config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=depth),
                                     vision_config=SimpleNamespace(num_hidden_layers=24))
            self.assertEqual(resolve_layers("all", config=config), list(range(depth + 1)))
        self.assertEqual(language_layer_count({"llm_config": {"num_hidden_layers": 36}}), 36)
        self.assertEqual(language_layer_count({"num_hidden_layers": 32}), 32)
        with self.assertRaises(ValueError):
            language_layer_count({"vision_config": {"num_hidden_layers": 24}})

    def test_explicit_subset_and_invalid(self):
        self.assertEqual(resolve_layers("0,1,8,32"), [0, 1, 8, 32])
        for invalid in ("", "0,0", "-1,0"):
            with self.assertRaises(ValueError):
                resolve_layers(invalid)

    def test_all_extraction_entry_defaults(self):
        import image_embedding_extraction, text_embedding_extraction
        import context_embedding_encoding_analysis, generation_context_embedding_analysis
        for module in (runner, image_embedding_extraction, text_embedding_extraction,
                       context_embedding_encoding_analysis, generation_context_embedding_analysis):
            with patch.object(sys, "argv", [module.__name__]):
                self.assertEqual(module.parse_args().layer_indices, "all")

    def test_runner_passes_identical_full_range(self):
        for depth in (32, 36):
            with patch.object(sys, "argv", ["runner"]), \
                 patch.object(runner, "resolve_layers", return_value=list(range(depth + 1))) as resolver, \
                 patch.object(runner, "run_step") as steps, \
                 patch.object(runner, "load_if_present", return_value={}), \
                 patch.object(runner, "write_json"):
                runner.main()
            resolver.assert_called_once_with("all", "llava-hf/llava-1.5-7b-hf")
            commands = [call.args[1] for call in steps.call_args_list if "--layer-indices" in call.args[1]]
            self.assertEqual(len(commands), 4)
            for command in commands:
                self.assertEqual(command[command.index("--layer-indices") + 1], ",".join(map(str, range(depth + 1))))

    def test_ceg_rejects_selected_layer_cache(self):
        import generation_context_embedding_analysis as ceg
        for stale_index in (0, 1):
            with TemporaryDirectory() as temp:
                output = Path(temp)
                directory = output / "ceg_generation_side/embeddings"
                directory.mkdir(parents=True)
                filenames = ("ceg_decision_embeddings.npz", "ceg_token_from_generate_embeddings.npz")
                for index, filename in enumerate(filenames):
                    np.savez(directory / filename, layer_indices=[0, 2] if index == stale_index else [0, 1, 2])
                (directory.parent / "predictions.csv").touch()
                args = SimpleNamespace(output_dir=output, ground_truth=None, dataset_root=None,
                    layer_indices="0,1,2", model_id="synthetic", skip_extraction=True)
                with patch.object(ceg, "parse_args", return_value=args), \
                     patch.object(ceg, "load_dataset", return_value=SimpleNamespace(rows=[])):
                    with self.assertRaisesRegex(ValueError, "Cached CEg layer range"):
                        ceg.main()


if __name__ == "__main__":
    unittest.main()
