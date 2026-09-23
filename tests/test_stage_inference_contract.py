import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from brain.ts_operators import ServeOperator
from filter.model_v2 import (
    TraceClassifierServingWrapper,
    curriculum_labels,
)


class FixedLogitModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, input_ids, attention_mask=None):
        return self.fixed_logits[: input_ids.shape[0]]


class StageInferenceContractTest(unittest.TestCase):
    def test_stage_one_serving_matches_binary_training_reduction(self):
        core = FixedLogitModel([
            [4.0, 1.0, 3.0, 5.0],
            [0.0, 2.0, 4.0, 6.0],
        ])
        wrapper = TraceClassifierServingWrapper(core, stage=1)
        actual = wrapper(torch.zeros((2, 1), dtype=torch.long))
        expected = torch.tensor([[4.0, 3.0], [0.0, 4.0]])
        torch.testing.assert_close(actual, expected)

    def test_stage_two_serving_preserves_exact_waypoint_logits(self):
        logits = [[1.0, 2.0, 4.0, 6.0, 8.0]]
        wrapper = TraceClassifierServingWrapper(
            FixedLogitModel(logits), stage=2
        )
        actual = wrapper(torch.zeros((1, 1), dtype=torch.long))
        torch.testing.assert_close(actual, torch.tensor(logits))

    def test_stage_two_training_labels_preserve_exact_classes(self):
        labels = torch.tensor([0, 1, 2, 3, 4])
        actual = curriculum_labels(labels, num_classes=5, stage=2)
        torch.testing.assert_close(actual, labels)

    def test_stage_three_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "invalid training stage"):
            TraceClassifierServingWrapper(
                FixedLogitModel([[1.0, 2.0, 3.0]]), stage=3
            )

    def test_torchscript_preserves_stage_one_reduction(self):
        wrapper = TraceClassifierServingWrapper(
            FixedLogitModel([[2.0, 0.0, 3.0]]), stage=1
        )
        inputs = (
            torch.zeros((1, 1), dtype=torch.long),
            torch.ones((1, 1), dtype=torch.long),
        )
        traced = torch.jit.trace(wrapper, inputs)
        torch.testing.assert_close(traced(*inputs), torch.tensor([[2.0, 1.5]]))

    def test_index_mapping_is_stage_aware(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            operator = ServeOperator(temp_dir, disable_auth=True)
            stage_one = operator.create_index2name(
                4, str(Path(temp_dir) / "stage1"), stage=1
            )
            stage_two = operator.create_index2name(
                4, str(Path(temp_dir) / "stage2"), stage=2
            )
            self.assertEqual(
                json.loads(Path(stage_one).read_text(encoding="utf-8")),
                {"0": "Unreachable", "1": "Reachable"},
            )
            self.assertEqual(
                json.loads(Path(stage_two).read_text(encoding="utf-8")),
                {
                    "0": "Unreachable",
                    "1": "Reach_Func1",
                    "2": "Reach_Func2",
                    "3": "Reach_Func3",
                },
            )
            with self.assertRaisesRegex(ValueError, "invalid training stage"):
                operator.create_index2name(
                    4, str(Path(temp_dir) / "stage3"), stage=3
                )


if __name__ == "__main__":
    unittest.main()
