"""Stage-aware cohort selection and hierarchical evidence tests."""

import threading
import unittest
from pathlib import Path
from unittest import mock

import torch

from brain.hierarchical_attribution import HierarchicalAttributionStore
from filter import attribution_guidance


def make_records(counts):
    records = {}
    for class_index, count in counts.items():
        for index in range(count):
            signature = f"class-{class_index}-sig-{index}"
            label = [False] * (max(counts) + 1)
            label[class_index] = True
            records[signature] = (
                f"call${class_index}_{index}()", label, class_index
            )
    return records


def snapshot(stage, objective, version, scores, *, members=(1,),
             active=(), analyzed=8, depth=1.0):
    return {
        "stage": stage,
        "objective": objective,
        "canonical_members": list(members),
        "active_exact_classes": list(active),
        "deployment_version": version,
        "selected_signature_digest": f"digest-{stage}-{objective}-{version}",
        "analyzed_count": analyzed,
        "depth_weight": depth,
        "scores": scores,
    }


class CohortSelectionTests(unittest.TestCase):
    def test_stage_one_keeps_binary_objective_and_depth_metadata(self):
        records = make_records({0: 3, 1: 30, 2: 30, 3: 30})
        cohorts = attribution_guidance.select_attribution_cohorts(
            records, num_classes=4, stage=1, deployment_version=7
        )
        self.assertEqual(len(cohorts), 1)
        cohort = cohorts[0]
        self.assertEqual(cohort.objective, "reachable")
        self.assertEqual(cohort.target_output, 1)
        self.assertEqual(cohort.max_samples, 48)
        self.assertEqual(cohort.canonical_members, (1, 2, 3))
        self.assertEqual(len(cohort.candidates), 90)
        self.assertTrue(all(
            candidate.canonical_class > 0 for candidate in cohort.candidates
        ))
        self.assertEqual(
            [candidate.signature for candidate in cohort.candidates],
            [candidate.signature for candidate in
             attribution_guidance.select_attribution_cohorts(
                 records, 4, 1, 7
             )[0].candidates],
        )
        final_depths = {
            candidate.depth_factor for candidate in cohort.candidates
            if candidate.canonical_class == 3
        }
        self.assertEqual(final_depths, {1.0})

    def test_stage_two_separates_shallow_and_deep_without_exact_labels(self):
        records = make_records({0: 2, 1: 10, 2: 10, 3: 10, 4: 10})
        cohorts = attribution_guidance.select_attribution_cohorts(
            records, num_classes=5, stage=2, deployment_version=2
        )
        self.assertEqual(
            [(item.objective, item.target_output, item.canonical_members,
              item.max_samples, item.depth_weight) for item in cohorts],
            [
                ("shallow", 1, (1, 2), 24, 0.60),
                ("deep", 2, (3, 4), 24, 1.00),
            ],
        )

    def test_sparse_stage_three_excludes_inactive_and_reach_other(self):
        records = make_records({0: 2, 1: 10, 2: 10, 3: 10, 4: 10})
        cohorts = attribution_guidance.select_attribution_cohorts(
            records, num_classes=5, stage=3, deployment_version=3,
            active_classes=(0, 1, 4),
        )
        self.assertEqual(
            [(item.objective, item.target_output) for item in cohorts],
            [("exact:1", 1), ("exact:4", 2)],
        )
        self.assertTrue(all(
            candidate.canonical_class in (1, 4)
            for cohort in cohorts for candidate in cohort.candidates
        ))

    def test_large_stage_three_rotates_but_always_includes_final(self):
        records = make_records({index: 1 for index in range(21)})
        active = tuple(range(21))
        first = attribution_guidance.select_attribution_cohorts(
            records, 21, 3, 1, active
        )
        second = attribution_guidance.select_attribution_cohorts(
            records, 21, 3, 2, active
        )
        first_classes = {item.canonical_members[0] for item in first}
        second_classes = {item.canonical_members[0] for item in second}
        self.assertEqual(len(first_classes), 16)
        self.assertEqual(len(second_classes), 16)
        self.assertIn(20, first_classes)
        self.assertIn(20, second_classes)
        self.assertNotEqual(first_classes, second_classes)


class HierarchicalStoreTests(unittest.TestCase):
    def test_replace_decay_expiry_and_specificity_first_publication(self):
        store = HierarchicalAttributionStore()
        store.begin_deployment(1, 1)
        store.replace([
            snapshot(1, "reachable", 1, {"stage1": 1.0, "shared": 1.0})
        ])
        store.begin_deployment(2, 2)
        store.replace([
            snapshot(2, "deep", 2, {"stage2": 1.0, "shared": 1.0})
        ])
        store.begin_deployment(3, 3, (0, 3))
        store.replace([
            snapshot(
                3, "exact:3", 3,
                {"stage3": 1.0, "shared": 1.0},
                members=(3,), active=(0, 3),
            )
        ])

        state = {
            (item["stage"], item["objective"]): item["effective_multiplier"]
            for item in store.audit_state()
        }
        self.assertEqual(state, {
            (1, "reachable"): 0.25,
            (2, "deep"): 0.5,
            (3, "exact:3"): 1.0,
        })
        published = list(store.publication_scores().keys())
        self.assertLess(published.index("stage3"), published.index("stage2"))
        self.assertLess(published.index("stage2"), published.index("stage1"))
        self.assertEqual(store.fuse().specificity["shared"], 3)

        store.begin_deployment(4, 3, (0, 3))
        self.assertNotIn(
            (1, "reachable"),
            {(item["stage"], item["objective"])
             for item in store.audit_state()},
        )

    def test_empty_update_retains_snapshot_and_replacement_is_per_objective(self):
        store = HierarchicalAttributionStore()
        store.begin_deployment(1, 2)
        store.replace([
            snapshot(2, "shallow", 1, {"read": 1.0}),
            snapshot(2, "deep", 1, {"write": 1.0}),
        ])
        store.begin_deployment(2, 2)
        self.assertEqual(store.replace([]), 0)
        store.replace([
            snapshot(2, "deep", 2, {"ioctl": 1.0})
        ])
        objectives = {
            item["objective"]: item["deployment_version"]
            for item in store.audit_state()
        }
        self.assertEqual(objectives, {"shallow": 1, "deep": 2})

    def test_stage_three_mapping_drops_only_removed_exact_bucket(self):
        store = HierarchicalAttributionStore()
        store.begin_deployment(1, 3, (0, 1, 2, 4))
        store.replace([
            snapshot(3, "exact:1", 1, {"read": 1.0},
                     members=(1,), active=(0, 1, 2, 4)),
            snapshot(3, "exact:2", 1, {"write": 1.0},
                     members=(2,), active=(0, 1, 2, 4)),
            snapshot(3, "exact:4", 1, {"ioctl": 1.0},
                     members=(4,), active=(0, 1, 2, 4)),
        ])
        store.begin_deployment(2, 3, (0, 1, 4))
        self.assertEqual(
            {item["objective"] for item in store.audit_state()},
            {"exact:1", "exact:4"},
        )

    def test_lower_stage_support_is_capped_numerically(self):
        store = HierarchicalAttributionStore()
        store.begin_deployment(1, 1)
        store.replace([
            snapshot(1, "reachable", 1, {"shared": 1.0})
        ])
        store.begin_deployment(2, 3, (0, 2))
        store.replace([
            snapshot(
                3, "exact:2", 2, {"shared": 0.01, "exact": 1.0},
                members=(2,), active=(0, 2),
            )
        ])

        fused = store.fuse().scores
        self.assertAlmostEqual(
            fused["shared"] / fused["exact"], 0.0125, places=6
        )

    def test_stage_freshness_changes_cross_stage_weight_ratio(self):
        store = HierarchicalAttributionStore()
        store.begin_deployment(1, 1)
        store.replace([
            snapshot(1, "reachable", 1, {"coarse": 1.0})
        ])
        store.begin_deployment(2, 3, (0, 2))
        store.replace([
            snapshot(
                3, "exact:2", 2, {"exact": 1.0},
                members=(2,), active=(0, 2),
            )
        ])
        ratio_at_age_one = (
            store.fuse().scores["coarse"] /
            store.fuse().scores["exact"]
        )
        store.begin_deployment(3, 3, (0, 2))
        store.replace([
            snapshot(
                3, "exact:2", 3, {"exact": 1.0},
                members=(2,), active=(0, 2),
            )
        ])
        ratio_at_age_two = (
            store.fuse().scores["coarse"] /
            store.fuse().scores["exact"]
        )

        self.assertAlmostEqual(ratio_at_age_one, 0.075)
        self.assertAlmostEqual(ratio_at_age_two, 0.0375)

    def test_store_rejects_wrong_stage_and_sparse_reach_other_atomically(self):
        store = HierarchicalAttributionStore()
        store.begin_deployment(1, 3, (0, 1, 4))
        valid = snapshot(
            3, "exact:4", 1, {"sendmsg$rds": 1.0},
            members=(4,), active=(0, 1, 4),
        )
        store.replace([valid])
        with self.assertRaisesRegex(ValueError, "deployed stage"):
            store.replace([
                snapshot(2, "deep", 1, {"read": 1.0}, members=(3, 4))
            ])
        with self.assertRaisesRegex(ValueError, "deployed objective"):
            store.replace([
                snapshot(
                    3, "Reach_Other", 1, {"write": 1.0},
                    members=(2, 3), active=(0, 1, 4),
                )
            ])
        self.assertEqual(
            {item["objective"] for item in store.audit_state()},
            {"exact:4"},
        )


class TokenizerRealisticTests(unittest.TestCase):
    tokenizer_path = Path.home() / "models" / "SyzTokenizer_224w"

    @unittest.skipUnless(tokenizer_path.is_dir(), "SyzTokenizer not installed")
    def test_exact_syzlang_calls_survive_real_token_boundaries(self):
        tokenizer = attribution_guidance.AutoTokenizer.from_pretrained(
            self.tokenizer_path
        )
        program = (
            "r0 = syz_open_dev$tty1(0x0, 0x0, 0x0)\n"
            "writev$auto(r0, 0x0, 0x0)\n"
            "r1 = syz_mount_image$f2fs(0x0, 0x0, 0x0, 0x0, 0x0, "
            "0x0, 0x0)\n"
            "r2 = socket$rds(0x15, 0x5, 0x0)\n"
            "sendmsg$rds(r2, 0x0, 0x0)\n"
        )
        token_ids = tokenizer(program, add_special_tokens=False)["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        attributions = [1.0] * len(tokens)
        attribution_guidance.replace_tokens(tokens)
        invocations, invocation_attrs = attribution_guidance.split_invocations(
            tokens, attributions
        )
        scores = attribution_guidance.get_syscall_attr(
            tokenizer, invocations, invocation_attrs
        )
        self.assertEqual(set(scores), {
            "syz_open_dev$tty1", "writev$auto", "syz_mount_image$f2fs",
            "socket$rds", "sendmsg$rds",
        })


class HierarchicalRunnerTests(unittest.TestCase):
    def _run_mocked(self, records, *, num_classes, stage,
                    active_classes=None, prediction=None, cancel_event=None):
        tokenizer = mock.Mock()
        tokenizer.eos_token = "<eos>"

        def encode(program, **_kwargs):
            class_index = int(program.split("$", 1)[1].split("_", 1)[0])
            return {
                "input_ids": torch.tensor([[class_index, 99]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        tokenizer.side_effect = encode
        tokenizer.convert_ids_to_tokens.return_value = ["read", "()"]
        model = mock.Mock()
        model.base_model.embeddings = object()

        def default_prediction(input_ids, _attention_mask):
            exact_class = int(input_ids[0, 0].item())
            if stage == 1:
                output = 1
                width = 2
            elif stage == 2:
                output = 1 if exact_class < 3 else 2
                width = 3
            else:
                output = tuple(active_classes).index(exact_class)
                width = len(active_classes) + int(
                    len(active_classes) < num_classes
                )
            probabilities = torch.zeros((1, width))
            probabilities[0, output] = 1.0
            return probabilities

        wrapped = mock.Mock(side_effect=prediction or default_prediction)
        lig = mock.Mock()
        lig.attribute.return_value = (
            torch.ones((1, 2, 1)), torch.tensor([0.0])
        )
        with (
            mock.patch.object(
                attribution_guidance, "load_canonical_records",
                return_value=records,
            ),
            mock.patch.object(
                attribution_guidance.AutoTokenizer, "from_pretrained",
                return_value=tokenizer,
            ),
            mock.patch.object(
                attribution_guidance, "TraceClassifierV2", return_value=model
            ),
            mock.patch.object(
                attribution_guidance, "AttributionProbabilityWrapper",
                return_value=wrapped,
            ),
            mock.patch.object(
                attribution_guidance, "LayerIntegratedGradients",
                return_value=lig,
            ),
            mock.patch.object(attribution_guidance.torch, "load", return_value={}),
            mock.patch.object(
                attribution_guidance, "get_syscall_attr",
                return_value={"read": 1.0},
            ),
        ):
            results = attribution_guidance.run_hierarchical_attribution_for_guidance(
                model_path="checkpoint.pt",
                base_model_path="encoder",
                tokenizer_path="tokenizer",
                data_dir="data",
                data_indices=[1],
                num_classes=num_classes,
                stage=stage,
                deployment_version=1,
                active_classes=active_classes,
                max_length=2,
                device="cpu",
                internal_batch_size=4,
                cancel_event=cancel_event,
            )
        return results, tokenizer, wrapped, lig

    def test_runner_uses_only_correct_reached_predictions(self):
        records = make_records({0: 1, 1: 9})
        tokenizer = mock.Mock()
        tokenizer.eos_token = "<eos>"
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        tokenizer.convert_ids_to_tokens.return_value = ["read", "()"]
        model = mock.Mock()
        model.base_model.embeddings = object()
        wrapped = mock.Mock(side_effect=(
            torch.tensor([[1.0, 0.0]]),
            *[torch.tensor([[0.0, 1.0]]) for _ in range(8)],
        ))
        lig = mock.Mock()
        lig.attribute.return_value = (
            torch.ones((1, 2, 1)), torch.tensor([0.0])
        )
        with (
            mock.patch.object(
                attribution_guidance, "load_canonical_records",
                return_value=records,
            ),
            mock.patch.object(
                attribution_guidance.AutoTokenizer, "from_pretrained",
                return_value=tokenizer,
            ),
            mock.patch.object(
                attribution_guidance, "TraceClassifierV2", return_value=model
            ),
            mock.patch.object(
                attribution_guidance, "AttributionProbabilityWrapper",
                return_value=wrapped,
            ),
            mock.patch.object(
                attribution_guidance, "LayerIntegratedGradients",
                return_value=lig,
            ),
            mock.patch.object(attribution_guidance.torch, "load", return_value={}),
            mock.patch.object(
                attribution_guidance, "get_syscall_attr",
                return_value={"read": 1.0},
            ),
        ):
            results = (
                attribution_guidance.run_hierarchical_attribution_for_guidance(
                    model_path="checkpoint.pt",
                    base_model_path="encoder",
                    tokenizer_path="tokenizer",
                    data_dir="data",
                    data_indices=[1],
                    num_classes=2,
                    stage=1,
                    deployment_version=1,
                    max_length=2,
                    device="cpu",
                    internal_batch_size=4,
                )
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["objective"], "reachable")
        self.assertEqual(results[0]["analyzed_count"], 8)
        self.assertEqual(results[0]["scores"], {"read": 1.0})
        self.assertEqual(lig.attribute.call_count, 8)
        self.assertTrue(all(
            call.kwargs["target"] == 1 for call in lig.attribute.call_args_list
        ))

    def test_stage_two_runner_uses_shallow_and_deep_targets(self):
        results, tokenizer, _, lig = self._run_mocked(
            make_records({1: 8, 3: 8}), num_classes=5, stage=2
        )
        self.assertEqual(
            [(result["objective"], result["analyzed_count"])
             for result in results],
            [("shallow", 8), ("deep", 8)],
        )
        self.assertEqual(
            {call.kwargs["target"] for call in lig.attribute.call_args_list},
            {1, 2},
        )
        first_four_classes = [
            int(call.args[0].split("$", 1)[1].split("_", 1)[0])
            for call in tokenizer.call_args_list[:4]
        ]
        self.assertEqual(first_four_classes, [1, 3, 1, 3])

    def test_sparse_stage_three_runner_never_scans_reach_other(self):
        results, tokenizer, _, lig = self._run_mocked(
            make_records({1: 8, 2: 8, 4: 8}),
            num_classes=5, stage=3, active_classes=(0, 1, 4),
        )
        self.assertEqual(
            [result["objective"] for result in results],
            ["exact:1", "exact:4"],
        )
        self.assertEqual(tokenizer.call_count, 16)
        self.assertEqual(
            {call.kwargs["target"] for call in lig.attribute.call_args_list},
            {1, 2},
        )

    def test_prediction_scan_uses_fixed_budget(self):
        def unreachable_prediction(_input_ids, _attention_mask):
            return torch.tensor([[1.0, 0.0]])

        results, tokenizer, _, lig = self._run_mocked(
            make_records({1: 100}), num_classes=2, stage=3,
            active_classes=(0, 1), prediction=unreachable_prediction,
        )
        self.assertEqual(results, [])
        self.assertEqual(tokenizer.call_count, 64)
        lig.attribute.assert_not_called()

    def test_cancellation_after_data_load_stops_before_model_setup(self):
        cancel_event = threading.Event()

        def load_then_cancel(*_args, **_kwargs):
            cancel_event.set()
            return make_records({1: 8})

        with (
            mock.patch.object(
                attribution_guidance, "load_canonical_records",
                side_effect=load_then_cancel,
            ),
            mock.patch.object(
                attribution_guidance.AutoTokenizer, "from_pretrained"
            ) as tokenizer_loader,
            self.assertRaises(attribution_guidance.AttributionCanceled),
        ):
            attribution_guidance.run_hierarchical_attribution_for_guidance(
                "checkpoint.pt", "encoder", "tokenizer", "data", [1],
                2, 1, 1, cancel_event=cancel_event,
            )
        tokenizer_loader.assert_not_called()

    def test_cancellation_after_prediction_stops_before_ig(self):
        cancel_event = threading.Event()

        def cancel_after_prediction(_input_ids, _attention_mask):
            cancel_event.set()
            return torch.tensor([[0.0, 1.0]])

        with self.assertRaises(attribution_guidance.AttributionCanceled):
            self._run_mocked(
                make_records({1: 8}), num_classes=2, stage=1,
                prediction=cancel_after_prediction,
                cancel_event=cancel_event,
            )


if __name__ == "__main__":
    unittest.main()
