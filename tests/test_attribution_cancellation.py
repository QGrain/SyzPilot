"""Lifecycle cancellation tests for Integrated Gradients guidance."""

import importlib
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
FILTER_DIR = REPO_ROOT / "filter"


class AttributionCancellationTests(unittest.TestCase):
    def test_final_target_filter_and_internal_batch_size_reach_captum(self):
        module_names = (
            "attribution_guidance", "model_v2", "dataset_v2",
            "attribution_wrapped", "model", "dataset", "config", "utils",
        )
        saved_modules = {
            name: sys.modules.get(name) for name in module_names
        }
        original_path = list(sys.path)
        sys.path[:] = [
            path for path in sys.path if path != str(FILTER_DIR)
        ]
        sys.path.insert(0, str(FILTER_DIR))
        for name in module_names:
            sys.modules.pop(name, None)
        try:
            attribution_guidance = importlib.import_module(
                "attribution_guidance"
            )
            tokenizer = mock.Mock()
            tokenizer.return_value = {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }
            tokenizer.convert_ids_to_tokens.return_value = [
                "ioctl", "()"
            ]
            model = mock.Mock()
            model.base_model.embeddings = object()
            wrapped_model = mock.Mock(side_effect=(
                torch.tensor([[0.0, 1.0, 0.0]]),
                torch.tensor([[1.0, 0.0, 0.0]]),
                torch.tensor([[0.0, 0.0, 1.0]]),
            ))
            lig = mock.Mock()
            lig.attribute.return_value = (
                torch.ones((1, 2, 1)), torch.tensor([0.0])
            )

            def merge_scores(total, current):
                total.update(current)

            with (
                mock.patch.object(
                    attribution_guidance,
                    "load_canonical_records",
                    return_value={
                        "shallow": (
                            "openat()", [False, True, False], 1
                        ),
                        "misclassified-final": (
                            "read()", [False, False, True], 2
                        ),
                        "unreachable-final": (
                            "write()", [False, False, True], 2
                        ),
                        "final": (
                            "ioctl()", [False, False, True], 2
                        ),
                    },
                ),
                mock.patch.object(
                    attribution_guidance.AutoTokenizer,
                    "from_pretrained",
                    return_value=tokenizer,
                ),
                mock.patch.object(
                    attribution_guidance,
                    "TraceClassifierV2",
                    return_value=model,
                ),
                mock.patch.object(
                    attribution_guidance,
                    "AttributionProbabilityWrapper",
                    return_value=wrapped_model,
                ),
                mock.patch.object(
                    attribution_guidance,
                    "LayerIntegratedGradients",
                    return_value=lig,
                ),
                mock.patch.object(
                    attribution_guidance.torch,
                    "load",
                    return_value={},
                ),
                mock.patch.object(
                    attribution_guidance,
                    "replace_tokens",
                ),
                mock.patch.object(
                    attribution_guidance,
                    "split_invocations",
                    return_value=([["openat"]], [[1.0]]),
                ),
                mock.patch.object(
                    attribution_guidance,
                    "get_syscall_attr",
                    return_value={"ioctl": 1.0},
                ),
                mock.patch.object(
                    attribution_guidance,
                    "merge_syscall_attr",
                    side_effect=merge_scores,
                ),
            ):
                result = (
                    attribution_guidance.run_attribution_for_guidance(
                        model_path="checkpoint.pt",
                        base_model_path="encoder",
                        tokenizer_path="tokenizer",
                        data_dir="data",
                        data_indices=[1],
                        num_classes=3,
                        stage=2,
                        target_class=2,
                        max_length=2,
                        device="cpu",
                        internal_batch_size=7,
                    )
                )

            self.assertEqual(result, {"ioctl": 1.0})
            self.assertEqual(tokenizer.call_count, 3)
            self.assertEqual(
                tokenizer.call_args,
                mock.call(
                    "ioctl()", max_length=2, padding="max_length",
                    truncation=True, return_tensors="pt",
                ),
            )
            lig.attribute.assert_called_once()
            self.assertEqual(lig.attribute.call_args.kwargs["target"], 2)
            self.assertEqual(
                lig.attribute.call_args.kwargs["internal_batch_size"], 7
            )
            self.assertEqual(lig.attribute.call_args.kwargs["n_steps"], 50)
        finally:
            for name in module_names:
                sys.modules.pop(name, None)
                if saved_modules[name] is not None:
                    sys.modules[name] = saved_modules[name]
            sys.path[:] = original_path

    def test_internal_batch_size_must_be_positive(self):
        module_names = (
            "attribution_guidance", "model_v2", "dataset_v2",
            "attribution_wrapped", "model", "dataset", "config", "utils",
        )
        saved_modules = {
            name: sys.modules.get(name) for name in module_names
        }
        original_path = list(sys.path)
        sys.path[:] = [
            path for path in sys.path if path != str(FILTER_DIR)
        ]
        sys.path.insert(0, str(FILTER_DIR))
        for name in module_names:
            sys.modules.pop(name, None)
        try:
            attribution_guidance = importlib.import_module(
                "attribution_guidance"
            )
            with self.assertRaisesRegex(ValueError, "must be positive"):
                attribution_guidance.run_attribution_for_guidance(
                    model_path="checkpoint.pt",
                    base_model_path="encoder",
                    tokenizer_path="tokenizer",
                    data_dir="data",
                    data_indices=[1],
                    num_classes=3,
                    stage=2,
                    internal_batch_size=0,
                )
        finally:
            for name in module_names:
                sys.modules.pop(name, None)
                if saved_modules[name] is not None:
                    sys.modules[name] = saved_modules[name]
            sys.path[:] = original_path

    def test_pre_canceled_run_does_not_load_training_data(self):
        module_names = (
            "attribution_guidance", "model_v2", "dataset_v2",
            "attribution_wrapped", "model", "dataset", "config", "utils",
        )
        saved_modules = {
            name: sys.modules.get(name) for name in module_names
        }
        original_path = list(sys.path)
        sys.path[:] = [
            path for path in sys.path if path != str(FILTER_DIR)
        ]
        sys.path.insert(0, str(FILTER_DIR))
        for name in module_names:
            sys.modules.pop(name, None)
        try:
            attribution_guidance = importlib.import_module(
                "attribution_guidance"
            )
            cancel_event = threading.Event()
            cancel_event.set()

            with mock.patch.object(
                    attribution_guidance,
                    "load_canonical_records") as loader:
                with self.assertRaises(
                        attribution_guidance.AttributionCanceled):
                    attribution_guidance.run_attribution_for_guidance(
                        model_path="checkpoint.pt",
                        base_model_path="encoder",
                        tokenizer_path="tokenizer",
                        data_dir="data",
                        data_indices=[1],
                        num_classes=3,
                        stage=2,
                        cancel_event=cancel_event,
                    )

            loader.assert_not_called()
        finally:
            for name in module_names:
                sys.modules.pop(name, None)
                if saved_modules[name] is not None:
                    sys.modules[name] = saved_modules[name]
            sys.path[:] = original_path


if __name__ == "__main__":
    unittest.main()
