"""Model-promotion quality gate regression tests."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRAIN = ROOT / "brain"
sys.path.insert(0, str(BRAIN))

from model_promotion import (
    decide_detailed_bootstrap,
    decide_model_promotion,
    decide_stage1_bootstrap,
    decide_stage2_bootstrap,
)


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
    if stage in (2, 3):
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
        if stage in (2, 3):
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
    def test_stage_one_bootstrap_reuses_production_gate(self):
        accepted = decide_stage1_bootstrap({
            "eval_loss": 0.2,
            "accuracy": 0.9,
            "macro_f1": 0.9,
            "class_counts": {0: 100, 1: 100},
            "per_class_recall": {0: 0.9, 1: 0.9},
        })
        rejected = decide_stage1_bootstrap({
            "eval_loss": 0.5,
            "accuracy": 0.9,
            "macro_f1": 0.47,
            "class_counts": {0: 100, 1: 900},
            "per_class_recall": {0: 0.0, 1: 1.0},
        })
        self.assertTrue(accepted["accepted"])
        self.assertFalse(rejected["accepted"])
        self.assertIn("low_class_recall", rejected["reason_codes"])

    def test_stage_one_good_first_model_passes(self):
        decision = decide_model_promotion(
            manifest(1, [100, 100], [0.9, 0.9]), 1, 5
        )
        self.assertTrue(decision["accepted"])

    def test_stage_two_bootstrap_reuses_absolute_production_gate(self):
        decision = decide_stage2_bootstrap({
            "eval_loss": 0.2,
            "accuracy": 0.85,
            "macro_f1": 0.8,
            "class_counts": {0: 100, 1: 100, 2: 100},
            "per_class_recall": {0: 0.8, 1: 0.85, 2: 0.9},
            "binary_eval_loss": 0.1,
            "binary_accuracy": 0.9,
            "binary_macro_f1": 0.9,
            "binary_class_counts": {0: 100, 1: 200},
            "binary_per_class_recall": {0: 0.8, 1: 0.95},
        }, num_classes=3)

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
        self.assertIn("low_deep_recall", decision["reason_codes"])

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
                [100, 100, 100],
                [0.8, 0.6, 0.6],
                macro_f1=0.7,
                baseline={
                    "loss": 0.2,
                    "macro_f1": 0.95,
                    "recalls": [0.95, 0.95, 0.95],
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
            3,
        )
        self.assertFalse(decision["accepted"])
        self.assertIn(
            "binary_relative_class_recall_regression",
            decision["reason_codes"],
        )

    def test_stage_transition_allows_small_binary_loss_tradeoff(self):
        decision = decide_model_promotion(
            manifest(
                3,
                [100, 100, 100, 50],
                [0.95, 0.9, 0.85, 0.8],
                macro_f1=0.9,
                baseline={
                    "loss": 0.6,
                    "macro_f1": 0.7,
                    "recalls": [0.9, 0.4, 0.4, 0.4],
                    "binary": {
                        "loss": 0.14,
                        "macro_f1": 0.95,
                        "recalls": [0.97, 0.94],
                    },
                },
                loaded_stage=2,
                binary={
                    "loss": 0.154,
                    "macro_f1": 0.95,
                    "recalls": [0.97, 0.94],
                },
            ),
            3,
            5,
            active_classes=(0, 1, 4),
        )
        self.assertTrue(decision["accepted"], decision["reason_codes"])
        self.assertEqual(
            decision["binary_loss_gate"]["mode"],
            "stage_transition_relative",
        )
        self.assertAlmostEqual(
            decision["binary_loss_gate"]["absolute_tolerance"], 0.014
        )
        self.assertAlmostEqual(
            decision["binary_loss_gate"]["maximum_candidate_loss"], 0.154
        )

    def test_stage_one_to_two_uses_transition_loss_gate(self):
        decision = decide_model_promotion(
            manifest(
                2,
                [100, 100, 100],
                [0.95, 0.9, 0.85],
                macro_f1=0.9,
                baseline={
                    "loss": 0.5,
                    "macro_f1": 0.7,
                    "recalls": [0.9, 0.5, 0.5],
                    "binary": {
                        "loss": 0.2,
                        "macro_f1": 0.9,
                        "recalls": [0.9, 0.9],
                    },
                },
                loaded_stage=1,
                binary={
                    "loss": 0.22,
                    "macro_f1": 0.9,
                    "recalls": [0.9, 0.9],
                },
            ),
            2,
            3,
        )
        self.assertTrue(decision["accepted"], decision["reason_codes"])
        self.assertEqual(
            decision["binary_loss_gate"]["mode"],
            "stage_transition_relative",
        )

    def test_same_stage_keeps_strict_binary_loss_gate(self):
        decision = decide_model_promotion(
            manifest(
                3,
                [100, 100, 100, 50],
                [0.95, 0.9, 0.85, 0.8],
                macro_f1=0.9,
                loss=0.18,
                baseline={
                    "loss": 0.2,
                    "macro_f1": 0.85,
                    "recalls": [0.94, 0.88, 0.82, 0.78],
                    "binary": {
                        "loss": 0.14,
                        "macro_f1": 0.95,
                        "recalls": [0.97, 0.94],
                    },
                },
                loaded_stage=3,
                binary={
                    "loss": 0.141,
                    "macro_f1": 0.95,
                    "recalls": [0.97, 0.94],
                },
            ),
            3,
            5,
            active_classes=(0, 1, 4),
        )
        self.assertFalse(decision["accepted"])
        self.assertIn(
            "binary_relative_loss_regression", decision["reason_codes"]
        )
        self.assertEqual(
            decision["binary_loss_gate"]["mode"], "same_stage_absolute"
        )

    def test_transition_loss_tolerance_does_not_relax_recall(self):
        decision = decide_model_promotion(
            manifest(
                3,
                [100, 100, 100, 50],
                [0.95, 0.9, 0.85, 0.8],
                macro_f1=0.9,
                baseline={
                    "loss": 0.6,
                    "macro_f1": 0.7,
                    "recalls": [0.9, 0.4, 0.4, 0.4],
                    "binary": {
                        "loss": 0.14,
                        "macro_f1": 0.95,
                        "recalls": [0.97, 0.94],
                    },
                },
                loaded_stage=2,
                binary={
                    "loss": 0.149,
                    "macro_f1": 0.91,
                    "recalls": [0.90, 0.92],
                },
            ),
            3,
            5,
            active_classes=(0, 1, 4),
        )
        self.assertFalse(decision["accepted"])
        self.assertIn(
            "binary_relative_class_recall_regression",
            decision["reason_codes"],
        )

    def test_non_adjacent_checkpoint_stage_fails_closed(self):
        evidence = manifest(
            3,
            [100, 100, 100, 50],
            [0.95, 0.9, 0.85, 0.8],
            macro_f1=0.9,
            baseline={
                "loss": 0.6,
                "macro_f1": 0.7,
                "recalls": [0.9, 0.4, 0.4, 0.4],
            },
            loaded_stage=1,
        )
        with self.assertRaisesRegex(ValueError, "immediate predecessor"):
            decide_model_promotion(
                evidence, 3, 5, active_classes=(0, 1, 4)
            )

    def test_checkpoint_stage_rollback_fails_closed(self):
        evidence = manifest(
            2,
            [100, 100, 100],
            [0.95, 0.9, 0.85],
            macro_f1=0.9,
            baseline={
                "loss": 0.6,
                "macro_f1": 0.7,
                "recalls": [0.9, 0.4, 0.4],
            },
            loaded_stage=3,
        )
        with self.assertRaisesRegex(ValueError, "immediate predecessor"):
            decide_model_promotion(evidence, 2, 3)

    def test_stage_transition_rejects_material_binary_loss_regression(self):
        decision = decide_model_promotion(
            manifest(
                3,
                [100, 100, 100, 50],
                [0.95, 0.9, 0.85, 0.8],
                macro_f1=0.9,
                baseline={
                    "loss": 0.6,
                    "macro_f1": 0.7,
                    "recalls": [0.9, 0.4, 0.4, 0.4],
                    "binary": {
                        "loss": 0.14,
                        "macro_f1": 0.95,
                        "recalls": [0.97, 0.94],
                    },
                },
                loaded_stage=2,
                binary={
                    "loss": 0.16,
                    "macro_f1": 0.95,
                    "recalls": [0.97, 0.94],
                },
            ),
            3,
            5,
            active_classes=(0, 1, 4),
        )
        self.assertFalse(decision["accepted"])
        self.assertIn(
            "binary_relative_loss_regression", decision["reason_codes"]
        )

    def test_stage_three_uses_compact_active_metric_classes(self):
        evidence = manifest(
            3,
            [100, 80, 100, 50],
            [0.9, 0.75, 0.7, 0.8],
            macro_f1=0.76,
        )
        decision = decide_model_promotion(
            evidence,
            3,
            5,
            active_classes=(0, 2, 4),
        )
        self.assertTrue(decision["accepted"], decision["reason_codes"])
        self.assertEqual(decision["active_exact_classes"], [0, 2, 4])

    def test_stage_three_bootstrap_counts_inactive_rows_as_binary_support(self):
        decision = decide_detailed_bootstrap(
            {
                "eval_loss": 0.2,
                "accuracy": 260 / 330,
                "macro_f1": 0.76,
                "class_counts": {0: 100, 1: 80, 2: 100, 3: 50},
                "per_class_recall": {
                    0: 0.9, 1: 0.75, 2: 0.7, 3: 0.8,
                },
                "binary_eval_loss": 0.2,
                "binary_accuracy": 0.9,
                "binary_macro_f1": 0.9,
                "binary_class_counts": {0: 100, 1: 230},
                "binary_per_class_recall": {0: 0.9, 1: 0.9},
            },
            stage=3,
            num_classes=5,
            active_classes=(0, 2, 4),
        )
        self.assertTrue(decision["accepted"], decision["reason_codes"])
        self.assertEqual(decision["validation_signature_count"], 330)

    def test_same_active_stage_three_regression_is_rejected(self):
        decision = decide_model_promotion(
            manifest(
                3, [100, 100, 100, 50], [0.85, 0.60, 0.52, 0.8],
                macro_f1=0.66, loss=0.3,
                baseline={
                    "loss": 0.2,
                    "macro_f1": 0.78,
                    "recalls": [0.92, 0.76, 0.70, 0.8],
                },
                loaded_stage=3,
            ),
            3,
            5,
            active_classes=(0, 2, 4),
        )
        self.assertFalse(decision["accepted"])
        self.assertIn("relative_loss_regression", decision["reason_codes"])

    def test_expanded_stage_three_material_improvement_passes(self):
        decision = decide_model_promotion(
            manifest(
                3, [100, 80, 100, 50], [0.92, 0.72, 0.75, 0.8],
                macro_f1=0.79, loss=0.18,
                baseline={
                    "loss": 0.24,
                    "macro_f1": 0.68,
                    "recalls": [0.90, 0.0, 0.72, 0.8],
                },
                loaded_stage=3,
            ),
            3,
            5,
            active_classes=(0, 2, 4),
        )
        self.assertTrue(decision["accepted"], decision["reason_codes"])

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
