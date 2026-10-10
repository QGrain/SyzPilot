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
    curriculum_logits,
)


class FixedLogitModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(logits, dtype=torch.float32))
        self.num_classes = len(logits[0])

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

    def test_stage_two_serving_groups_shallow_and_deep_waypoints(self):
        logits = [[1.0, 2.0, 4.0, 6.0, 8.0]]
        wrapper = TraceClassifierServingWrapper(
            FixedLogitModel(logits), stage=2
        )
        actual = wrapper(torch.zeros((1, 1), dtype=torch.long))
        torch.testing.assert_close(
            actual, torch.tensor([[1.0, 3.0, 7.0]])
        )

    def test_stage_two_training_labels_form_three_groups(self):
        labels = torch.tensor([0, 1, 2, 3, 4])
        actual = curriculum_labels(labels, num_classes=5, stage=2)
        torch.testing.assert_close(actual, torch.tensor([0, 1, 1, 2, 2]))

    def test_stage_three_groups_inactive_reached_logits_as_other(self):
        wrapper = TraceClassifierServingWrapper(
            FixedLogitModel([[1.0, 2.0, 3.0, 4.0, 5.0]]),
            stage=3,
            active_classes=(0, 1, 3, 4),
        )
        actual = wrapper(torch.zeros((1, 1), dtype=torch.long))
        torch.testing.assert_close(
            actual, torch.tensor([[1.0, 2.0, 4.0, 5.0, 3.0]])
        )

    def test_sparse_reached_rows_train_the_other_output(self):
        logits = torch.tensor(
            [[2.0, 0.0, -1.0]], requires_grad=True
        )
        grouped_logits = curriculum_logits(
            logits, 3, active_classes=(0, 2)
        )
        grouped_labels = curriculum_labels(
            torch.tensor([1]), 3, 3, active_classes=(0, 2)
        )
        loss = torch.nn.functional.cross_entropy(
            grouped_logits, grouped_labels
        )
        loss.backward()

        self.assertGreater(loss.item(), 0)
        self.assertLess(logits.grad[0, 1].item(), 0)

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

    def test_torchscript_preserves_stage_two_grouping(self):
        wrapper = TraceClassifierServingWrapper(
            FixedLogitModel([[1.0, 2.0, 4.0, 6.0, 8.0]]), stage=2
        )
        inputs = (
            torch.zeros((1, 1), dtype=torch.long),
            torch.ones((1, 1), dtype=torch.long),
        )
        traced = torch.jit.trace(wrapper, inputs)
        torch.testing.assert_close(
            traced(*inputs), torch.tensor([[1.0, 3.0, 7.0]])
        )

    def test_stage_three_torchscript_round_trip_preserves_grouped_logits(self):
        wrapper = TraceClassifierServingWrapper(
            FixedLogitModel([
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
            ]),
            stage=3,
            active_classes=(0, 1, 3, 4),
        )
        inputs = (
            torch.zeros((1, 1), dtype=torch.long),
            torch.ones((1, 1), dtype=torch.long),
        )
        traced = torch.jit.trace(wrapper, inputs)
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = str(Path(temp_dir) / "stage3.pt")
            torch.jit.save(traced, model_path)
            loaded = torch.jit.load(model_path)
            actual = loaded(
                torch.zeros((2, 1), dtype=torch.long),
                torch.ones((2, 1), dtype=torch.long),
            )
        torch.testing.assert_close(
            actual,
            torch.tensor([
                [1.0, 2.0, 4.0, 5.0, 3.0],
                [6.0, 7.0, 9.0, 10.0, 8.0],
            ]),
        )

    @unittest.skipUnless(
        torch.cuda.device_count() >= 2,
        "requires two CUDA devices to verify cross-device loading",
    )
    def test_stage_three_torchscript_indices_follow_load_device(self):
        export_device = torch.device("cuda:1")
        serving_device = torch.device("cuda:0")
        wrapper = TraceClassifierServingWrapper(
            FixedLogitModel([[1.0, 2.0, 3.0, 4.0, 5.0]]),
            stage=3,
            active_classes=(0, 1, 3, 4),
        ).to(export_device)
        inputs = (
            torch.zeros((1, 1), dtype=torch.long, device=export_device),
            torch.ones((1, 1), dtype=torch.long, device=export_device),
        )
        traced = torch.jit.trace(wrapper, inputs)
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = str(Path(temp_dir) / "stage3-cross-device.pt")
            torch.jit.save(traced, model_path)
            loaded = torch.jit.load(
                model_path, map_location=serving_device
            )
            actual = loaded(
                torch.zeros((1, 1), dtype=torch.long, device=serving_device),
                torch.ones((1, 1), dtype=torch.long, device=serving_device),
            )
        torch.testing.assert_close(
            actual.cpu(), torch.tensor([[1.0, 2.0, 4.0, 5.0, 3.0]])
        )

    def test_index_mapping_is_stage_aware(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            operator = ServeOperator(temp_dir, disable_auth=True)
            stage_one = operator.create_index2name(
                4, str(Path(temp_dir) / "stage1"), stage=1
            )
            stage_two = operator.create_index2name(
                4, str(Path(temp_dir) / "stage2"), stage=2
            )
            stage_three = operator.create_index2name(
                4, str(Path(temp_dir) / "stage3"), stage=3,
                active_classes=(0, 2, 3),
            )
            self.assertEqual(
                json.loads(Path(stage_one).read_text(encoding="utf-8")),
                {"0": "Unreachable", "1": "Reachable"},
            )
            self.assertEqual(
                json.loads(Path(stage_two).read_text(encoding="utf-8")),
                {
                    "0": "Unreachable",
                    "1": "Reach_Shallow",
                    "2": "Reach_Deep",
                },
            )
            self.assertEqual(
                json.loads(Path(stage_three).read_text(encoding="utf-8")),
                {
                    "0": "Unreachable",
                    "1": "Reach_Func2",
                    "2": "Reach_Func3",
                    "3": "Reach_Other",
                },
            )


if __name__ == "__main__":
    unittest.main()
