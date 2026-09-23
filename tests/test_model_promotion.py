"""Model-promotion quality gate regression tests."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRAIN = ROOT / "brain"
sys.path.insert(0, str(BRAIN))

from model_promotion import decide_model_promotion


def manifest(stage, counts, recalls, *, accuracy=None, macro_f1=0.85,
             loss=0.2, baseline=None, loaded_stage=0,
             binary=None, signature_count=None):
    if accuracy is None:
        accuracy = sum(
            count * recall for count, recall in zip(counts, recalls)
        ) / sum(counts)
    result = {
        "best_eval_loss": loss,
        "best_eval_accuracy": accuracy,
        "best_eval_macro_f1": macro_f1,
        "validation_class_counts": {
            str(index): value for index, value in enumerate(counts)
        },
        "validation_per_class_recall": {
            str(index): value for index, value in enumerate(recalls)
        },
        "validation_signature_count": (
            sum(counts) if signature_count is None else signature_count
        ),
        "validation_signature_sha256": "a" * 64,
        "loaded_checkpoint": "active.pt" if baseline else None,
        "loaded_checkpoint_stage": loaded_stage,
    }
    if stage == 2:
        binary_counts = [counts[0], sum(counts[1:])]
        binary = binary or {
            "loss": 0.2,
            "macro_f1": 0.9,
            "recalls": [0.9, 0.9],
        }
        binary_accuracy = sum(
            count * recall
            for count, recall in zip(binary_counts, binary["recalls"])
        ) / sum(binary_counts)
        result.update({
            "binary_eval_loss": binary["loss"],
            "binary_eval_accuracy": binary_accuracy,
            "binary_eval_macro_f1": binary["macro_f1"],
            "binary_validation_class_counts": {
                "0": binary_counts[0], "1": binary_counts[1]
            },
            "binary_validation_per_class_recall": {
                "0": binary["recalls"][0], "1": binary["recalls"][1]
            },
        })
    if baseline:
        baseline_accuracy = sum(
            count * recall
            for count, recall in zip(counts, baseline["recalls"])
        ) / sum(counts)
        result.update({
            "baseline_eval_loss": baseline["loss"],
            "baseline_eval_accuracy": baseline_accuracy,
            "baseline_eval_macro_f1": baseline["macro_f1"],
            "baseline_validation_class_counts": {
                str(index): value for index, value in enumerate(counts)
            },
            "baseline_validation_per_class_recall": {
                str(index): value
                for index, value in enumerate(baseline["recalls"])
            },
        })
        if stage == 2:
            baseline_binary = baseline.get("binary", binary)
            binary_counts = [counts[0], sum(counts[1:])]
            baseline_binary_accuracy = sum(
                count * recall
                for count, recall in zip(
                    binary_counts, baseline_binary["recalls"]
                )
            ) / sum(binary_counts)
            result.update({
                "baseline_binary_eval_loss": baseline_binary["loss"],
                "baseline_binary_eval_accuracy": baseline_binary_accuracy,
                "baseline_binary_eval_macro_f1": baseline_binary["macro_f1"],
                "baseline_binary_validation_class_counts": {
                    "0": binary_counts[0], "1": binary_counts[1]
                },
                "baseline_binary_validation_per_class_recall": {
                    "0": baseline_binary["recalls"][0],
                    "1": baseline_binary["recalls"][1],
                },
            })
    return result


class ModelPromotionTest(unittest.TestCase):
    def test_stage_one_good_first_model_passes(self):
        decision = decide_model_promotion(
            manifest(1, [100, 100], [0.9, 0.9]), 1, 5
        )
        self.assertTrue(decision["accepted"])

    def test_majority_predictor_is_rejected(self):
        decision = decide_model_promotion(
            manifest(
                1, [100, 900], [0.0, 1.0],
                accuracy=0.9, macro_f1=0.47,
            ),
            1,
            5,
        )
        self.assertFalse(decision["accepted"])
        self.assertIn("below_majority_margin", decision["reason_codes"])
        self.assertIn("low_class_recall", decision["reason_codes"])

    def test_stage_two_requires_exact_class_support_and_recall(self):
        decision = decide_model_promotion(
            manifest(
                2, [100, 100, 12], [0.95, 0.8, 0.1],
                macro_f1=0.7,
            ),
            2,
            3,
        )
        self.assertFalse(decision["accepted"])
        self.assertIn("insufficient_class_support", decision["reason_codes"])
        self.assertIn("low_final_recall", decision["reason_codes"])

    def test_same_stage_regression_is_rejected(self):
        decision = decide_model_promotion(
            manifest(
                2, [100, 100, 100], [0.85, 0.72, 0.55],
                macro_f1=0.70, loss=0.3,
                baseline={
                    "loss": 0.2,
                    "macro_f1": 0.82,
                    "recalls": [0.92, 0.85, 0.72],
                },
                loaded_stage=2,
            ),
            2,
            3,
        )
        self.assertFalse(decision["accepted"])
        self.assertIn("relative_loss_regression", decision["reason_codes"])
        self.assertIn("relative_class_recall_regression", decision["reason_codes"])

    def test_same_stage_material_improvement_passes(self):
        decision = decide_model_promotion(
            manifest(
                2, [100, 100, 100], [0.92, 0.84, 0.72],
                macro_f1=0.84, loss=0.19,
                baseline={
                    "loss": 0.2,
                    "macro_f1": 0.82,
                    "recalls": [0.92, 0.83, 0.70],
                },
                loaded_stage=2,
            ),
            2,
            3,
        )
        self.assertTrue(decision["accepted"], decision["reason_codes"])

    def test_stage_two_rejects_binary_regression_from_stage_one(self):
        decision = decide_model_promotion(
            manifest(
                2,
                [100, 100],
                [0.8, 0.6],
                macro_f1=0.7,
                baseline={
                    "loss": 0.2,
                    "macro_f1": 0.95,
                    "recalls": [0.95, 0.95],
                    "binary": {
                        "loss": 0.1,
                        "macro_f1": 0.95,
                        "recalls": [0.95, 0.95],
                    },
                },
                loaded_stage=1,
                binary={
                    "loss": 0.4,
                    "macro_f1": 0.7,
                    "recalls": [0.8, 0.6],
                },
            ),
            2,
            2,
        )
        self.assertFalse(decision["accepted"])
        self.assertIn(
            "binary_relative_class_recall_regression",
            decision["reason_codes"],
        )

    def test_signature_count_must_match_evaluated_support(self):
        with self.assertRaisesRegex(ValueError, "metric support"):
            decide_model_promotion(
                manifest(
                    1,
                    [100, 100],
                    [0.9, 0.9],
                    signature_count=999,
                ),
                1,
                5,
            )


if __name__ == "__main__":
    unittest.main()
