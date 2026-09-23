"""Regression tests for validation-selected training artifacts."""

import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
FILTER_DIR = REPO_ROOT / "filter"
sys.path.insert(0, str(FILTER_DIR))
# Production entry points use script-local top-level imports. Avoid inheriting
# brain/config.py when this test follows controller tests in the same process.
for module_name in ("config", "utils"):
    sys.modules.pop(module_name, None)

from train_v2 import (
    TrainerV2,
    binary_serving_logits,
    effective_eval_steps,
    write_training_manifest,
)


class TrainingManifestTest(unittest.TestCase):
    def test_binary_projection_matches_each_serving_stage(self):
        raw_logits = torch.tensor([[0.0, 1.0, -2.0]])

        stage_one = binary_serving_logits(raw_logits, 1)
        stage_two = binary_serving_logits(raw_logits, 2)

        self.assertEqual(torch.argmax(stage_one, dim=1).item(), 0)
        self.assertEqual(torch.argmax(stage_two, dim=1).item(), 1)
        self.assertEqual(stage_one[0, 1].item(), -0.5)
        self.assertEqual(stage_two[0, 1].item(), 1.0)

    def test_effective_eval_steps_covers_unique_records_once(self):
        self.assertEqual(effective_eval_steps(142, 512, 25), 1)
        self.assertEqual(effective_eval_steps(1025, 512, 25), 3)
        self.assertEqual(effective_eval_steps(20000, 512, 25), 25)
        with self.assertRaises(ValueError):
            effective_eval_steps(0, 512, 25)

    def test_eval_loss_weights_short_tail_by_example_count(self):
        class FakeModel:
            def eval(self):
                return None

            def __call__(self, input_ids, attention_mask):
                batch_size = input_ids.shape[0]
                loss = 0.1 if batch_size == 2 else 0.3
                other_logit = math.log(math.exp(loss) - 1.0)
                return torch.tensor(
                    [[0.0, other_logit]] * batch_size
                )

        trainer = TrainerV2.__new__(TrainerV2)
        trainer.model = FakeModel()
        trainer.accelerator = SimpleNamespace(is_main_process=True)
        trainer.eval_steps = 2
        trainer.test_dl = [
            (torch.zeros((2, 1)), torch.ones((2, 1)), torch.zeros((2, 2))),
            (torch.zeros((1, 1)), torch.ones((1, 1)), torch.zeros((1, 2))),
        ]
        trainer.config = {
            "test_only": False, "num_classes": 2, "train_stage": 1,
        }

        _, _, metrics = trainer.test()

        self.assertAlmostEqual(metrics["eval_loss"], 0.5 / 3.0, places=6)
        self.assertEqual(metrics["class_counts"], {0: 3, 1: 0})
        self.assertEqual(metrics["per_class_recall"], {0: 1.0, 1: 0.0})
        self.assertAlmostEqual(metrics["macro_f1"], 0.5)

    def test_manifest_points_to_best_checkpoint_and_records_hash(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)
            best_checkpoint = save_dir / "step-200.pt"
            best_checkpoint.write_bytes(b"best-model")
            (save_dir / "step-400.pt").write_bytes(b"worse-final-model")

            manifest_path = write_training_manifest(
                save_dir,
                best_checkpoint=best_checkpoint,
                best_step=200,
                best_eval_loss=0.25,
                final_step=400,
                config={
                    "train_stage": 1,
                    "num_classes": 4,
                    "data_idx": [1, 2, 3],
                    "test_data_idx": [4],
                    "load_path": "",
                    "batch_size": 64,
                    "grad_acc_steps": 2,
                    "mixed_precision": "bf16",
                    "total_steps": 400,
                    "test_interval": 100,
                    "min_steps": 100,
                    "patience": 3,
                    "num_warmup_steps": 100,
                    "is_first_train": True,
                    "assigned_physical_gpu": "1",
                    "validation_signature_count": 25,
                    "validation_signature_sha256": "a" * 64,
                    "initialization_wall_seconds": 3.5,
                    "training_wall_seconds": 12.5,
                    "trainer_process_wall_seconds": 16.0,
                    "cuda_peak_measurement_scope": (
                        "accelerator_initialized_through_training_completion"
                    ),
                    "max_rss_kib": 1234,
                    "cuda_peak_allocated_bytes": 1000,
                    "cuda_peak_reserved_bytes": 2000,
                },
                best_metrics={
                    "accuracy": 0.9,
                    "f1_score": 0.85,
                    "macro_f1": 0.75,
                    "class_counts": {0: 20, 1: 5},
                    "per_class_recall": {0: 0.95, 1: 0.6},
                    "binary_eval_loss": 0.25,
                    "binary_accuracy": 0.9,
                    "binary_macro_f1": 0.75,
                    "binary_class_counts": {0: 20, 1: 5},
                    "binary_per_class_recall": {0: 0.95, 1: 0.6},
                },
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["best_checkpoint"], str(best_checkpoint))
            self.assertEqual(manifest["best_step"], 200)
            self.assertEqual(manifest["final_step"], 400)
            self.assertEqual(manifest["best_eval_loss"], 0.25)
            self.assertEqual(manifest["best_eval_accuracy"], 0.9)
            self.assertEqual(manifest["best_eval_weighted_f1"], 0.85)
            self.assertEqual(manifest["best_eval_macro_f1"], 0.75)
            self.assertEqual(manifest["micro_batch_size"], 64)
            self.assertEqual(manifest["gradient_accumulation_steps"], 2)
            self.assertEqual(manifest["effective_batch_size"], 128)
            self.assertEqual(manifest["mixed_precision"], "bf16")
            self.assertEqual(manifest["training_profile"], "first")
            self.assertEqual(manifest["configured_total_steps"], 400)
            self.assertEqual(manifest["test_interval"], 100)
            self.assertEqual(manifest["min_steps"], 100)
            self.assertEqual(manifest["patience"], 3)
            self.assertEqual(manifest["num_warmup_steps"], 100)
            self.assertEqual(manifest["assigned_physical_gpu"], "1")
            self.assertEqual(manifest["validation_signature_count"], 25)
            self.assertEqual(manifest["validation_signature_sha256"], "a" * 64)
            self.assertEqual(manifest["initialization_wall_seconds"], 3.5)
            self.assertEqual(manifest["training_wall_seconds"], 12.5)
            self.assertEqual(manifest["trainer_process_wall_seconds"], 16.0)
            self.assertEqual(
                manifest["cuda_peak_measurement_scope"],
                "accelerator_initialized_through_training_completion",
            )
            self.assertEqual(manifest["max_rss_kib"], 1234)
            self.assertEqual(manifest["cuda_peak_allocated_bytes"], 1000)
            self.assertEqual(manifest["cuda_peak_reserved_bytes"], 2000)
            self.assertEqual(
                manifest["validation_class_counts"], {"0": 20, "1": 5}
            )
            self.assertEqual(
                manifest["validation_per_class_recall"],
                {"0": 0.95, "1": 0.6},
            )
            self.assertEqual(
                manifest["checkpoint_sha256"],
                hashlib.sha256(b"best-model").hexdigest(),
            )

            with self.assertRaisesRegex(ValueError, "must be finite"):
                write_training_manifest(
                    save_dir,
                    best_checkpoint=best_checkpoint,
                    best_step=200,
                    best_eval_loss=math.nan,
                    final_step=400,
                    config={
                        "train_stage": 1,
                        "num_classes": 4,
                        "data_idx": [1, 2, 3],
                        "test_data_idx": [4],
                        "load_path": "",
                        "total_steps": 400,
                        "test_interval": 100,
                        "min_steps": 100,
                        "patience": 3,
                        "num_warmup_steps": 100,
                    },
                )

    def test_manifest_records_same_split_baseline_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)
            active = save_dir / "active.pt"
            candidate = save_dir / "step-100.pt"
            active.write_bytes(b"active")
            candidate.write_bytes(b"candidate")
            common_metrics = {
                "accuracy": 0.9,
                "f1_score": 0.88,
                "macro_f1": 0.86,
                "class_counts": {0: 50, 1: 50},
                "per_class_recall": {0: 0.9, 1: 0.9},
                "binary_eval_loss": 0.2,
                "binary_accuracy": 0.9,
                "binary_macro_f1": 0.86,
                "binary_class_counts": {0: 50, 1: 50},
                "binary_per_class_recall": {0: 0.9, 1: 0.9},
            }
            manifest_path = write_training_manifest(
                save_dir,
                best_checkpoint=candidate,
                best_step=100,
                best_eval_loss=0.2,
                final_step=100,
                config={
                    "train_stage": 1,
                    "num_classes": 4,
                    "data_idx": [1],
                    "test_data_idx": [2],
                    "load_path": str(active),
                    "loaded_checkpoint_stage": 1,
                    "total_steps": 100,
                    "test_interval": 100,
                    "min_steps": 100,
                    "patience": 2,
                    "num_warmup_steps": 10,
                    "validation_signature_count": 100,
                    "validation_signature_sha256": "b" * 64,
                },
                best_metrics={**common_metrics, "eval_loss": 0.2},
                baseline_metrics={
                    **common_metrics,
                    "eval_loss": 0.3,
                    "binary_eval_loss": 0.3,
                },
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["loaded_checkpoint_stage"], 1)
            self.assertEqual(
                manifest["loaded_checkpoint_sha256"],
                hashlib.sha256(b"active").hexdigest(),
            )
            self.assertEqual(manifest["baseline_eval_loss"], 0.3)
            self.assertEqual(
                manifest["baseline_validation_class_counts"],
                manifest["validation_class_counts"],
            )


if __name__ == "__main__":
    unittest.main()
