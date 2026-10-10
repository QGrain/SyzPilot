"""Fail-closed quality gates for online reachability model promotion."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
import sys
from typing import Any, Dict, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.curriculum import curriculum_output_classes


@dataclass(frozen=True)
class PromotionThresholds:
    majority_margin: float = 0.05
    stage1_min_support: int = 32
    stage1_min_macro_f1: float = 0.75
    stage1_min_recall: float = 0.75
    stage2_min_support: int = 16
    stage2_min_macro_f1: float = 0.60
    stage2_min_unreachable_recall: float = 0.75
    stage2_min_reached_recall: float = 0.40
    stage2_min_final_recall: float = 0.50
    relative_loss_tolerance: float = 1e-4
    stage_transition_binary_loss_ratio: float = 0.10
    relative_macro_f1_tolerance: float = 0.02
    relative_mean_recall_tolerance: float = 0.02
    relative_class_recall_tolerance: float = 0.05
    material_loss_improvement: float = 1e-4
    material_macro_f1_improvement: float = 0.01
    material_min_recall_improvement: float = 0.02

    def validate(self) -> None:
        probability_fields = (
            self.majority_margin,
            self.stage1_min_macro_f1,
            self.stage1_min_recall,
            self.stage2_min_macro_f1,
            self.stage2_min_unreachable_recall,
            self.stage2_min_reached_recall,
            self.stage2_min_final_recall,
            self.relative_macro_f1_tolerance,
            self.relative_mean_recall_tolerance,
            self.relative_class_recall_tolerance,
            self.stage_transition_binary_loss_ratio,
            self.material_macro_f1_improvement,
            self.material_min_recall_improvement,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 1
               for value in probability_fields):
            raise ValueError("promotion probability thresholds must be in [0, 1]")
        if min(self.stage1_min_support, self.stage2_min_support) <= 0:
            raise ValueError("promotion support thresholds must be positive")
        if (not math.isfinite(self.relative_loss_tolerance) or
                not math.isfinite(self.material_loss_improvement) or
                min(self.relative_loss_tolerance,
                    self.material_loss_improvement) < 0):
            raise ValueError("promotion loss thresholds must be finite and non-negative")


def _metric_view(manifest: Mapping[str, Any], prefix: str,
                 expected_classes: int) -> Dict[str, Any]:
    scalar_prefix = f"{prefix}_" if prefix else ""
    collection_prefix = "" if prefix == "best" else scalar_prefix
    loss = float(manifest[f"{scalar_prefix}eval_loss"])
    accuracy = float(manifest[f"{scalar_prefix}eval_accuracy"])
    macro_f1 = float(manifest[f"{scalar_prefix}eval_macro_f1"])
    counts = {
        int(class_index): int(count)
        for class_index, count in manifest[
            f"{collection_prefix}validation_class_counts"
        ].items()
    }
    recalls = {
        int(class_index): float(recall)
        for class_index, recall in manifest[
            f"{collection_prefix}validation_per_class_recall"
        ].items()
    }
    expected = set(range(expected_classes))
    if set(counts) != expected or set(recalls) != expected:
        raise ValueError("promotion metrics do not cover the active classes")
    if any(count < 0 for count in counts.values()):
        raise ValueError("promotion class support must be non-negative")
    probabilities = [accuracy, macro_f1, *recalls.values()]
    if (not math.isfinite(loss) or
            any(not math.isfinite(value) or not 0 <= value <= 1
                for value in probabilities)):
        raise ValueError("promotion metrics must be finite and bounded")
    total_support = sum(counts.values())
    measured_accuracy = (
        sum(counts[index] * recalls[index] for index in counts) /
        total_support
        if total_support else 0.0
    )
    if total_support and not math.isclose(
            accuracy, measured_accuracy, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("promotion accuracy is inconsistent with class recall")
    return {
        "loss": loss,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "counts": counts,
        "recalls": recalls,
    }


def _binary_metric_view(
        manifest: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    key_prefix = f"{prefix}_binary_" if prefix else "binary_"
    synthetic = {
        "eval_loss": manifest[f"{key_prefix}eval_loss"],
        "eval_accuracy": manifest[f"{key_prefix}eval_accuracy"],
        "eval_macro_f1": manifest[f"{key_prefix}eval_macro_f1"],
        "validation_class_counts": manifest[
            f"{key_prefix}validation_class_counts"
        ],
        "validation_per_class_recall": manifest[
            f"{key_prefix}validation_per_class_recall"
        ],
    }
    return _metric_view(synthetic, "", 2)


def _append_relative_regressions(
        reasons: list, candidate: Mapping[str, Any],
        baseline: Mapping[str, Any], thresholds: PromotionThresholds,
        reason_prefix: str = "", class_indices=None,
        loss_tolerance: Optional[float] = None) -> None:
    if class_indices is None:
        class_indices = tuple(candidate["recalls"])
    else:
        class_indices = tuple(class_indices)
    candidate_mean_recall = sum(
        candidate["recalls"][index] for index in class_indices
    ) / len(class_indices)
    baseline_mean_recall = sum(
        baseline["recalls"][index] for index in class_indices
    ) / len(class_indices)
    if loss_tolerance is None:
        loss_tolerance = thresholds.relative_loss_tolerance
    if candidate["loss"] > baseline["loss"] + loss_tolerance:
        reasons.append(f"{reason_prefix}relative_loss_regression")
    if (candidate["macro_f1"] < baseline["macro_f1"] -
            thresholds.relative_macro_f1_tolerance):
        reasons.append(f"{reason_prefix}relative_macro_f1_regression")
    if (candidate_mean_recall < baseline_mean_recall -
            thresholds.relative_mean_recall_tolerance):
        reasons.append(f"{reason_prefix}relative_mean_recall_regression")
    if any(
            candidate["recalls"][class_index] <
            baseline["recalls"][class_index] -
            thresholds.relative_class_recall_tolerance
            for class_index in class_indices):
        reasons.append(f"{reason_prefix}relative_class_recall_regression")


def decide_model_promotion(
        manifest: Mapping[str, Any], stage: int, num_classes: int,
        thresholds: Optional[PromotionThresholds] = None,
        active_classes=None) -> Dict[str, Any]:
    """Return an auditable decision without mutating controller state."""
    thresholds = thresholds or PromotionThresholds()
    thresholds.validate()
    if stage not in (1, 2, 3):
        raise ValueError(f"invalid curriculum stage: {stage}")
    expected_classes = curriculum_output_classes(
        num_classes, stage, active_classes
    )
    if expected_classes < 2:
        raise ValueError("promotion requires at least two active classes")
    signature_count = int(manifest["validation_signature_count"])
    signature_digest = str(manifest["validation_signature_sha256"])
    if signature_count <= 0 or re.fullmatch(r"[0-9a-f]{64}", signature_digest) is None:
        raise ValueError("invalid validation signature fingerprint")

    reasons = []
    candidate = _metric_view(manifest, "best", expected_classes)
    total_support = sum(candidate["counts"].values())
    if signature_count != total_support:
        raise ValueError(
            "validation signature count does not match metric support"
        )
    if total_support <= 0:
        reasons.append("empty_validation")
        majority_accuracy = 1.0
    else:
        majority_accuracy = max(candidate["counts"].values()) / total_support
    if candidate["accuracy"] + 1e-12 < (
            majority_accuracy + thresholds.majority_margin):
        reasons.append("below_majority_margin")

    if stage == 1:
        if min(candidate["counts"].values()) < thresholds.stage1_min_support:
            reasons.append("insufficient_class_support")
        if candidate["macro_f1"] < thresholds.stage1_min_macro_f1:
            reasons.append("low_macro_f1")
        if min(candidate["recalls"].values()) < thresholds.stage1_min_recall:
            reasons.append("low_class_recall")
    else:
        exact_metric_classes = (
            tuple(range(len(active_classes)))
            if stage == 3 else tuple(range(expected_classes))
        )
        reached_metric_classes = exact_metric_classes[1:]
        final_metric_class = exact_metric_classes[-1]
        if min(
                candidate["counts"][index]
                for index in exact_metric_classes
        ) < thresholds.stage2_min_support:
            reasons.append("insufficient_class_support")
        if candidate["macro_f1"] < thresholds.stage2_min_macro_f1:
            reasons.append("low_macro_f1")
        if (candidate["recalls"][0] <
                thresholds.stage2_min_unreachable_recall):
            reasons.append("low_unreachable_recall")
        if any(
                candidate["recalls"][class_index] <
                thresholds.stage2_min_reached_recall
                for class_index in reached_metric_classes
        ):
            reasons.append("low_reached_recall")
        if (candidate["recalls"][final_metric_class] <
                thresholds.stage2_min_final_recall):
            reasons.append(
                "low_deep_recall" if stage == 2 else "low_final_recall"
            )

        binary_candidate = _binary_metric_view(manifest, "")
        active_positive_support = sum(
            candidate["counts"][class_index]
            for class_index in range(1, expected_classes)
        )
        collapsed_counts = {
            0: candidate["counts"][0],
            1: active_positive_support,
        }
        support_matches = binary_candidate["counts"] == collapsed_counts
        if (not support_matches or
                sum(binary_candidate["counts"].values()) != signature_count):
            raise ValueError(
                "binary validation support does not match the holdout"
            )
        if min(binary_candidate["counts"].values()) < (
                thresholds.stage1_min_support):
            reasons.append("binary_insufficient_class_support")
        if binary_candidate["macro_f1"] < thresholds.stage1_min_macro_f1:
            reasons.append("binary_low_macro_f1")
        if min(binary_candidate["recalls"].values()) < (
                thresholds.stage1_min_recall):
            reasons.append("binary_low_class_recall")

    baseline = None
    loaded_stage = int(manifest.get("loaded_checkpoint_stage", 0))
    allowed_loaded_stages = {stage}
    if stage > 1:
        allowed_loaded_stages.add(stage - 1)
    if (manifest.get("loaded_checkpoint") and
            loaded_stage not in allowed_loaded_stages):
        raise ValueError(
            "loaded checkpoint stage must match the candidate stage or its "
            "immediate predecessor"
        )
    if manifest.get("loaded_checkpoint") and loaded_stage == stage:
        baseline = _metric_view(manifest, "baseline", expected_classes)
        if baseline["counts"] != candidate["counts"]:
            raise ValueError(
                "baseline and candidate validation support do not match"
            )
        _append_relative_regressions(
            reasons,
            candidate,
            baseline,
            thresholds,
            class_indices=(
                range(len(active_classes)) if stage == 3 else None
            ),
        )

        material_classes = (
            range(len(active_classes)) if stage == 3
            else candidate["recalls"]
        )
        candidate_min_recall = min(
            candidate["recalls"][index] for index in material_classes
        )
        material_classes = (
            range(len(active_classes)) if stage == 3
            else baseline["recalls"]
        )
        baseline_min_recall = min(
            baseline["recalls"][index] for index in material_classes
        )
        materially_better = (
            candidate["loss"] <= baseline["loss"] -
            thresholds.material_loss_improvement or
            candidate["macro_f1"] >= baseline["macro_f1"] +
            thresholds.material_macro_f1_improvement or
            candidate_min_recall >= baseline_min_recall +
            thresholds.material_min_recall_improvement
        )
        if not materially_better:
            reasons.append("no_material_improvement")

    binary_baseline = None
    binary_loss_gate = None
    if stage in (2, 3) and manifest.get("loaded_checkpoint"):
        binary_baseline = _binary_metric_view(manifest, "baseline")
        if binary_baseline["counts"] != binary_candidate["counts"]:
            raise ValueError(
                "binary baseline and candidate support do not match"
            )
        binary_loss_tolerance = thresholds.relative_loss_tolerance
        gate_mode = "same_stage_absolute"
        if (loaded_stage, stage) in ((1, 2), (2, 3)):
            gate_mode = "stage_transition_relative"
            binary_loss_tolerance = max(
                binary_loss_tolerance,
                binary_baseline["loss"] *
                thresholds.stage_transition_binary_loss_ratio,
            )
        binary_loss_gate = {
            "mode": gate_mode,
            "loaded_stage": loaded_stage,
            "candidate_stage": stage,
            "baseline_loss": binary_baseline["loss"],
            "candidate_loss": binary_candidate["loss"],
            "absolute_tolerance": binary_loss_tolerance,
            "maximum_candidate_loss": (
                binary_baseline["loss"] + binary_loss_tolerance
            ),
        }
        _append_relative_regressions(
            reasons,
            binary_candidate,
            binary_baseline,
            thresholds,
            reason_prefix="binary_",
            loss_tolerance=binary_loss_tolerance,
        )

    return {
        "schema_version": 1,
        "accepted": not reasons,
        "reason_codes": sorted(set(reasons)),
        "stage": stage,
        "num_classes": num_classes,
        "majority_accuracy": majority_accuracy,
        "thresholds": asdict(thresholds),
        "candidate": candidate,
        "baseline": baseline,
        "active_exact_classes": (
            list(active_classes) if stage == 3 else []
        ),
        "binary_candidate": binary_candidate if stage in (2, 3) else None,
        "binary_baseline": binary_baseline,
        "binary_loss_gate": binary_loss_gate,
        "validation_signature_count": signature_count,
        "validation_signature_sha256": signature_digest,
    }


def decide_stage1_bootstrap(
        metrics: Mapping[str, Any],
        thresholds: Optional[PromotionThresholds] = None) -> Dict[str, Any]:
    """Apply the production Stage-1 gate to in-memory evaluation metrics."""
    counts = {
        int(class_index): int(count)
        for class_index, count in metrics["class_counts"].items()
    }
    recalls = {
        int(class_index): float(recall)
        for class_index, recall in metrics["per_class_recall"].items()
    }
    if set(counts) != {0, 1} or set(recalls) != {0, 1}:
        raise ValueError("Stage-1 bootstrap metrics require classes 0 and 1")
    manifest = {
        "validation_signature_count": sum(counts.values()),
        # Deployment independently verifies the real validation fingerprint.
        # This training-time probe only reuses the identical quality gate.
        "validation_signature_sha256": "0" * 64,
        "best_eval_loss": metrics["eval_loss"],
        "best_eval_accuracy": metrics["accuracy"],
        "best_eval_macro_f1": metrics["macro_f1"],
        "validation_class_counts": counts,
        "validation_per_class_recall": recalls,
        "loaded_checkpoint": None,
        "loaded_checkpoint_stage": 0,
    }
    return decide_model_promotion(
        manifest,
        stage=1,
        num_classes=2,
        thresholds=thresholds,
    )


def decide_stage2_bootstrap(
        metrics: Mapping[str, Any], num_classes: int,
        thresholds: Optional[PromotionThresholds] = None) -> Dict[str, Any]:
    """Apply absolute production Stage-2 gates to in-memory metrics."""
    manifest = {
        "validation_signature_count": sum(
            int(count) for count in metrics["class_counts"].values()
        ),
        "validation_signature_sha256": "0" * 64,
        "best_eval_loss": metrics["eval_loss"],
        "best_eval_accuracy": metrics["accuracy"],
        "best_eval_macro_f1": metrics["macro_f1"],
        "validation_class_counts": metrics["class_counts"],
        "validation_per_class_recall": metrics["per_class_recall"],
        "binary_eval_loss": metrics["binary_eval_loss"],
        "binary_eval_accuracy": metrics["binary_accuracy"],
        "binary_eval_macro_f1": metrics["binary_macro_f1"],
        "binary_validation_class_counts": metrics["binary_class_counts"],
        "binary_validation_per_class_recall": metrics[
            "binary_per_class_recall"
        ],
        "loaded_checkpoint": None,
        "loaded_checkpoint_stage": 0,
    }
    return decide_model_promotion(
        manifest,
        stage=2,
        num_classes=num_classes,
        thresholds=thresholds,
    )


def decide_detailed_bootstrap(
        metrics: Mapping[str, Any], stage: int, num_classes: int,
        active_classes=None,
        thresholds: Optional[PromotionThresholds] = None) -> Dict[str, Any]:
    """Apply absolute Stage-2/3 gates to in-memory validation metrics."""
    if stage not in (2, 3):
        raise ValueError("detailed bootstrap requires Stage 2 or Stage 3")
    exact_support = sum(
        int(count) for count in metrics["class_counts"].values()
    )
    binary_support = sum(
        int(count) for count in metrics["binary_class_counts"].values()
    )
    manifest = {
        "validation_signature_count": binary_support,
        "validation_signature_sha256": "0" * 64,
        "best_eval_loss": metrics["eval_loss"],
        "best_eval_accuracy": metrics["accuracy"],
        "best_eval_macro_f1": metrics["macro_f1"],
        "validation_class_counts": metrics["class_counts"],
        "validation_per_class_recall": metrics["per_class_recall"],
        "binary_eval_loss": metrics["binary_eval_loss"],
        "binary_eval_accuracy": metrics["binary_accuracy"],
        "binary_eval_macro_f1": metrics["binary_macro_f1"],
        "binary_validation_class_counts": metrics["binary_class_counts"],
        "binary_validation_per_class_recall": metrics[
            "binary_per_class_recall"
        ],
        "loaded_checkpoint": None,
        "loaded_checkpoint_stage": 0,
    }
    if exact_support != binary_support:
        raise ValueError("detailed bootstrap metric support does not match")
    return decide_model_promotion(
        manifest,
        stage=stage,
        num_classes=num_classes,
        thresholds=thresholds,
        active_classes=active_classes,
    )
