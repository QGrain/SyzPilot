import sys
import threading
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
BRAIN_DIR = ROOT / "brain"
if str(BRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(BRAIN_DIR))

from guidance_engine import (  # noqa: E402
    ArgConstraint,
    GuidanceConfig,
    GuidanceEngine,
    MutationTemplate,
)


class GuidanceEngineTest(unittest.TestCase):
    def test_empty_attribution_update_clears_previous_snapshot(self):
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_attribution({"old$call": 1.0})

        engine.update_attribution({})
        guidance = engine.compute_guidance()

        self.assertEqual(engine.get_attribution_weights(), {})
        self.assertNotIn("old$call", guidance["syscall_weights"])

    def test_static_scores_share_a_fixed_source_budget(self):
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_static_analysis([
            {"name": "sendmsg", "weight": 0.125},
            {"name": "sendmsg", "weight": 0.01},
            {"name": "socket", "weight": 0.05},
        ])

        guidance = engine.compute_guidance()

        weights = guidance["syscall_weights"]
        self.assertAlmostEqual(sum(weights.values()), 0.3)
        self.assertAlmostEqual(
            weights["sendmsg"] / weights["socket"], 0.125 / 0.05
        )

    def test_attribution_candidate_count_does_not_multiply_budget(self):
        single = GuidanceEngine(GuidanceConfig())
        single.update_attribution({"read": 1.0})

        many = GuidanceEngine(GuidanceConfig())
        many.update_attribution({
            f"call${index}": 1.0 for index in range(30)
        })

        self.assertAlmostEqual(
            sum(single.compute_guidance()["syscall_weights"].values()),
            0.3,
        )
        self.assertAlmostEqual(
            sum(many.compute_guidance()["syscall_weights"].values()),
            0.3,
        )

    def test_threshold_is_source_relative_not_post_budget(self):
        engine = GuidanceEngine(GuidanceConfig(
            attribution_weight=0.05,
            min_weight_threshold=0.1,
        ))
        engine.update_attribution({"read": 1.0, "write": 0.05})

        weights = engine.compute_guidance()["syscall_weights"]

        self.assertEqual(set(weights), {"read"})
        self.assertAlmostEqual(weights["read"], 0.05)

    def test_same_syscall_adds_static_and_attribution_budgets(self):
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_static_analysis([{"name": "sendmsg$rds", "weight": 1.0}])
        engine.update_attribution({"sendmsg$rds": 1.0})

        weights = engine.compute_guidance()["syscall_weights"]

        self.assertEqual(set(weights), {"sendmsg$rds"})
        self.assertAlmostEqual(weights["sendmsg$rds"], 0.6)

    def test_top_k_reapplies_each_retained_source_budget(self):
        engine = GuidanceEngine(GuidanceConfig(max_syscall_weights=3))
        engine.update_static_analysis([
            {"name": "static$one", "weight": 1.0},
            {"name": "static$two", "weight": 1.0},
        ])
        engine.update_attribution({
            f"ig${index:02d}": 1.0 for index in range(30)
        })

        weights = engine.compute_guidance()["syscall_weights"]

        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(
            sum(weight for name, weight in weights.items()
                if name.startswith("static$")),
            0.3,
        )
        self.assertAlmostEqual(
            sum(weight for name, weight in weights.items()
                if name.startswith("ig$")),
            0.3,
        )

    def test_invalid_attribution_entries_clear_or_are_ignored(self):
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_attribution({"old$call": 1.0})

        engine.update_attribution({
            "nan$call": float("nan"),
            "inf$call": float("inf"),
            "negative$call": -1.0,
            "boolean$call": True,
            "valid$call": 2.0,
        })

        self.assertEqual(engine.get_attribution_weights(), {
            "valid$call": 1.0,
        })

        engine.update_attribution({"zero$call": 0.0})
        self.assertEqual(engine.get_attribution_weights(), {})

    def test_equal_weight_top_k_uses_syscall_name_tie_break(self):
        engine = GuidanceEngine(GuidanceConfig(max_syscall_weights=2))
        engine.update_static_analysis([
            {"name": "write", "weight": 1.0},
            {"name": "accept", "weight": 1.0},
            {"name": "bind", "weight": 1.0},
        ])

        guidance = engine.compute_guidance()

        self.assertEqual(
            list(guidance["syscall_weights"]), ["accept", "bind"]
        )

    def test_seed_only_static_candidate_is_a_hint_not_a_weight(self):
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_static_analysis([
            {"name": "mount", "weight": 1.0,
             "delivery_mode": "generatable"},
            {"name": "syz_mount_image$f2fs", "weight": 0.98,
             "delivery_mode": "seed_only"},
            {"name": "ioctl$DISABLED", "weight": 1.0,
             "delivery_mode": "disabled"},
        ])

        guidance = engine.compute_guidance()

        self.assertEqual(set(guidance["syscall_weights"]), {"mount"})
        self.assertEqual(
            guidance["generation_hints"]["preferred_syscalls"],
            ["mount", "syz_mount_image$f2fs"],
        )

    def test_seed_only_guidance_can_carry_no_choice_table_weights(self):
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_static_analysis([
            {"name": "syz_mount_image$f2fs", "weight": 1.0,
             "delivery_mode": "seed_only"},
        ])

        guidance = engine.compute_guidance()

        self.assertEqual(guidance["syscall_weights"], {})
        self.assertEqual(
            guidance["generation_hints"]["preferred_syscalls"],
            ["syz_mount_image$f2fs"],
        )

    def test_seed_only_score_does_not_suppress_generatable_weight(self):
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_static_analysis([
            {"name": "mount", "weight": 0.2,
             "delivery_mode": "generatable",
             "guidance_role": "primitive_fallback"},
            {"name": "syz_mount_image$f2fs", "weight": 1.0,
             "delivery_mode": "seed_only",
             "guidance_role": "entry_exact"},
        ])

        guidance = engine.compute_guidance()

        self.assertAlmostEqual(guidance["syscall_weights"]["mount"], 0.3)
        self.assertEqual(
            set(guidance["generation_hints"]["preferred_syscalls"]),
            {"mount", "syz_mount_image$f2fs"},
        )

    def test_generatable_exact_entry_dominates_primitive_and_peers(self):
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_static_analysis([
            {"name": "sendmsg", "weight": 1.0,
             "delivery_mode": "generatable",
             "guidance_role": "primitive_fallback"},
            {"name": "sendmsg$rds", "weight": 0.9,
             "delivery_mode": "generatable",
             "guidance_role": "entry_exact"},
            {"name": "setsockopt$RDS_GET_MR", "weight": 0.6,
             "delivery_mode": "generatable",
             "guidance_role": "subsystem_peer"},
        ])

        guidance = engine.compute_guidance()

        self.assertAlmostEqual(
            guidance["syscall_weights"]["sendmsg$rds"], 0.21
        )
        fallback_total = sum(
            weight for name, weight in guidance["syscall_weights"].items()
            if name != "sendmsg$rds"
        )
        self.assertAlmostEqual(fallback_total, 0.09)
        self.assertGreater(
            guidance["syscall_weights"]["sendmsg$rds"],
            guidance["syscall_weights"]["sendmsg"],
        )
        self.assertEqual(
            guidance["generation_hints"]["preferred_syscalls"][0],
            "sendmsg$rds",
        )

    def test_role_budget_is_reapplied_after_top_k_truncation(self):
        engine = GuidanceEngine(GuidanceConfig(max_syscall_weights=30))
        entries = [{
            "name": "sendmsg$rds",
            "weight": 1.0,
            "delivery_mode": "generatable",
            "guidance_role": "entry_exact",
        }]
        entries.extend({
            "name": f"peer${index:03d}",
            "weight": 1.0,
            "delivery_mode": "generatable",
            "guidance_role": "subsystem_peer",
        } for index in range(100))
        engine.update_static_analysis(entries)

        weights = engine.compute_guidance()["syscall_weights"]

        self.assertEqual(len(weights), 30)
        self.assertIn("sendmsg$rds", weights)
        self.assertAlmostEqual(weights["sendmsg$rds"], 0.21)
        self.assertAlmostEqual(
            sum(weight for name, weight in weights.items()
                if name != "sendmsg$rds"),
            0.09,
        )

    def test_top_k_reserves_exact_and_setup_slots(self):
        engine = GuidanceEngine(GuidanceConfig(max_syscall_weights=2))
        engine.update_static_analysis([
            {"name": "sendmsg$rds", "weight": 1.0,
             "guidance_role": "entry_exact"},
            {"name": "socket$rds", "weight": 1.0,
             "guidance_role": "resource_producer"},
        ])
        engine.update_attribution({"unrelated$high_ig": 1.0})

        weights = engine.compute_guidance()["syscall_weights"]

        self.assertEqual(set(weights), {"sendmsg$rds", "socket$rds"})
        self.assertAlmostEqual(weights["sendmsg$rds"], 0.21)
        self.assertAlmostEqual(weights["socket$rds"], 0.09)

    def test_attribution_remains_independent_of_static_role_budget(self):
        engine = GuidanceEngine(GuidanceConfig(max_syscall_weights=3))
        engine.update_static_analysis([
            {"name": "sendmsg$rds", "weight": 1.0,
             "guidance_role": "entry_exact"},
            {"name": "socket$rds", "weight": 1.0,
             "guidance_role": "resource_producer"},
        ])
        engine.update_attribution({"unrelated$high_ig": 1.0})

        weights = engine.compute_guidance()["syscall_weights"]

        self.assertAlmostEqual(weights["sendmsg$rds"], 0.21)
        self.assertAlmostEqual(weights["socket$rds"], 0.09)
        self.assertAlmostEqual(weights["unrelated$high_ig"], 0.3)
        self.assertAlmostEqual(sum(weights.values()), 0.6)

    def test_overlapping_exact_and_attribution_do_not_consume_two_slots(self):
        engine = GuidanceEngine(GuidanceConfig(max_syscall_weights=3))
        engine.update_static_analysis([
            {"name": "sendmsg$rds", "weight": 1.0,
             "guidance_role": "entry_exact"},
            {"name": "socket$rds", "weight": 1.0,
             "guidance_role": "resource_producer"},
        ])
        engine.update_attribution({
            "sendmsg$rds": 1.0,
            "unrelated$ig": 0.9,
        })

        weights = engine.compute_guidance()["syscall_weights"]

        self.assertEqual(
            set(weights), {"sendmsg$rds", "socket$rds", "unrelated$ig"}
        )
        self.assertAlmostEqual(sum(weights.values()), 0.6)
        self.assertAlmostEqual(weights["socket$rds"], 0.09)

    def test_small_top_k_never_returns_duplicate_syscall_slots(self):
        for capacity, expected_count in ((0, 0), (1, 1), (2, 2), (3, 3)):
            with self.subTest(capacity=capacity):
                engine = GuidanceEngine(GuidanceConfig(
                    max_syscall_weights=capacity
                ))
                engine.update_static_analysis([
                    {"name": "sendmsg$rds", "weight": 1.0,
                     "guidance_role": "entry_exact"},
                    {"name": "socket$rds", "weight": 1.0,
                     "guidance_role": "resource_producer"},
                ])
                engine.update_attribution({
                    "sendmsg$rds": 1.0,
                    "unrelated$ig": 0.9,
                })

                weights = engine.compute_guidance()["syscall_weights"]

                self.assertEqual(len(weights), expected_count)
                self.assertLessEqual(len(weights), capacity)

    def test_exact_role_wins_duplicate_weaker_evidence_for_same_call(self):
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_static_analysis([
            {"name": "sendmsg$rds", "weight": 0.1,
             "guidance_role": "entry_exact"},
            {"name": "sendmsg$rds", "weight": 1.0,
             "guidance_role": "primitive_fallback"},
            {"name": "socket$rds", "weight": 1.0,
             "guidance_role": "resource_producer"},
        ])

        weights = engine.compute_guidance()["syscall_weights"]

        self.assertAlmostEqual(weights["sendmsg$rds"], 0.21)
        self.assertAlmostEqual(weights["socket$rds"], 0.09)

    def test_static_templates_survive_sequence_refresh_and_are_weighted(self):
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_static_templates([{
            "type": "sequence",
            "syscalls": ["socket$rds", "sendmsg$rds"],
            "priority": 1.0,
            "insert_mode": "prefix",
        }])

        engine.update_sequence_patterns([])
        templates = engine.compute_guidance()["mutation_templates"]

        self.assertEqual(len(templates), 1)
        self.assertEqual(
            templates[0]["syscalls"], ["socket$rds", "sendmsg$rds"]
        )
        self.assertAlmostEqual(templates[0]["priority"], 0.3)

    def test_duplicate_static_and_sequence_template_is_sent_once(self):
        engine = GuidanceEngine(GuidanceConfig())
        template = {
            "type": "sequence",
            "syscalls": ["socket$rds", "sendmsg$rds"],
            "priority": 1.0,
            "insert_mode": "prefix",
        }
        engine.update_static_templates([template])
        engine.update_sequence_patterns([template])

        templates = engine.compute_guidance()["mutation_templates"]

        self.assertEqual(len(templates), 1)
        self.assertAlmostEqual(templates[0]["priority"], 0.5)

    def test_reordered_arg_hints_deduplicate_across_sources(self):
        base = {
            "type": "sequence",
            "syscalls": ["socket$rds", "sendmsg$rds"],
            "priority": 1.0,
            "insert_mode": "prefix",
        }
        first_hint = {
            "syscall": "socket$rds", "arg_idx": 0, "value": 21,
        }
        second_hint = {
            "syscall": "sendmsg$rds", "arg_idx": 1, "value": 4,
        }
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_static_templates([{
            **base, "arg_hints": [first_hint, second_hint],
        }])
        engine.update_sequence_patterns([{
            **base, "arg_hints": [second_hint, first_hint],
        }])

        templates = engine.compute_guidance()["mutation_templates"]

        self.assertEqual(len(templates), 1)
        self.assertAlmostEqual(templates[0]["priority"], 0.5)
        self.assertEqual(
            templates[0]["arg_hints"], [second_hint, first_hint]
        )

    def test_template_candidate_count_does_not_multiply_source_budget(self):
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_static_templates([{
            "type": "sequence",
            "syscalls": ["static$call"],
            "priority": 1.0,
        }])
        engine.update_sequence_patterns([{
            "type": "sequence",
            "syscalls": [f"sequence${index:02d}"],
            "priority": 1.0,
        } for index in range(15)])

        templates = engine.compute_guidance()["mutation_templates"]
        static_total = sum(
            template["priority"] for template in templates
            if template["syscalls"] == ["static$call"]
        )
        sequence_total = sum(
            template["priority"] for template in templates
            if template["syscalls"][0].startswith("sequence$")
        )

        self.assertEqual(len(templates), 16)
        self.assertAlmostEqual(static_total, 0.3)
        self.assertAlmostEqual(sequence_total, 0.2)

    def test_template_cap_preserves_each_retained_source_budget(self):
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_static_templates([{
            "type": "sequence",
            "syscalls": [f"static${index:02d}"],
            "priority": 1.0,
        } for index in range(10)])
        engine.update_sequence_patterns([{
            "type": "sequence",
            "syscalls": [f"sequence${index:02d}"],
            "priority": 1.0,
        } for index in range(20)])

        templates = engine.compute_guidance()["mutation_templates"]
        static_total = sum(
            template["priority"] for template in templates
            if template["syscalls"][0].startswith("static$")
        )
        sequence_total = sum(
            template["priority"] for template in templates
            if template["syscalls"][0].startswith("sequence$")
        )

        self.assertEqual(len(templates), 16)
        self.assertGreater(sum(
            1 for template in templates
            if template["syscalls"][0].startswith("static$")
        ), 0)
        self.assertGreater(sum(
            1 for template in templates
            if template["syscalls"][0].startswith("sequence$")
        ), 0)
        self.assertAlmostEqual(static_total, 0.3)
        self.assertAlmostEqual(sequence_total, 0.2)

    def test_malformed_static_templates_are_ignored(self):
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_static_templates([
            None,
            {"syscalls": [], "priority": 1.0},
            {"syscalls": ["sendmsg$rds"], "priority": float("nan")},
            {"syscalls": ["sendmsg$rds"], "priority": True},
            {"type": {}, "syscalls": ["sendmsg$rds"], "priority": 1.0},
            {"type": "unknown", "syscalls": ["sendmsg$rds"],
             "priority": 1.0},
            {"insert_mode": [], "syscalls": ["sendmsg$rds"],
             "priority": 1.0},
            {"insert_mode": "unknown", "syscalls": ["sendmsg$rds"],
             "priority": 1.0},
            {"syscalls": [f"call${index}" for index in range(33)],
             "priority": 2.0},
        ])

        self.assertEqual(
            engine.compute_guidance()["mutation_templates"], []
        )

    def test_poc_templates_accept_analyzer_dicts_and_dataclasses(self):
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_poc_patterns([
            {
                "type": "sequence",
                "syscalls": ["socket$rds", "sendmsg$rds"],
                "priority": 0.7,
                "insert_mode": "prefix",
                "arg_hints": [{
                    "syscall": "socket$rds", "arg_idx": 0, "value": 21,
                }],
            },
            MutationTemplate(
                type="prefix",
                syscalls=["socket$rds"],
                priority=0.3,
                insert_mode="prefix",
                arg_hints=[ArgConstraint(
                    syscall="socket$rds", arg_idx=1, value=5,
                )],
            ),
        ])

        templates = engine.compute_guidance()["mutation_templates"]

        self.assertEqual(len(templates), 2)
        self.assertAlmostEqual(sum(t["priority"] for t in templates), 0.2)
        self.assertEqual(sum(len(t.get("arg_hints", [])) for t in templates), 2)

    def test_malformed_poc_templates_are_ignored_before_budgeting(self):
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_poc_patterns([
            {"syscalls": ["read"], "priority": float("nan")},
            {"syscalls": ["read"], "priority": float("inf")},
            {"syscalls": ["read"], "priority": 0.0},
            {"syscalls": ["read"], "priority": -1.0},
            {"syscalls": [], "priority": 1.0},
            {"type": "unknown", "syscalls": ["read"], "priority": 1.0},
            {"insert_mode": "unknown", "syscalls": ["read"],
             "priority": 1.0},
            {"syscalls": ["read"], "priority": 1.0,
             "arg_hints": [{"syscall": "write", "arg_idx": 0,
                             "value": 1}]},
            MutationTemplate(
                type="bad", syscalls=["read"], priority=1.0
            ),
            MutationTemplate(
                type="sequence", syscalls=["read"], priority=float("nan")
            ),
        ])

        self.assertEqual(
            engine.compute_guidance()["mutation_templates"], []
        )

    def test_invalid_templates_do_not_displace_valid_static_template(self):
        engine = GuidanceEngine(GuidanceConfig())
        invalid = [{
            "type": {},
            "syscalls": [f"invalid${index}"],
            "priority": 100.0 - index,
        } for index in range(20)]
        engine.update_static_templates(invalid + [{
            "type": "sequence",
            "syscalls": ["socket$rds", "sendmsg$rds"],
            "priority": 1.0,
            "insert_mode": "prefix",
        }])

        templates = engine.compute_guidance()["mutation_templates"]

        self.assertEqual(len(templates), 1)
        self.assertEqual(
            templates[0]["syscalls"], ["socket$rds", "sendmsg$rds"]
        )

    def test_malformed_entries_do_not_crash_or_enable_role_suppression(self):
        engine = GuidanceEngine(GuidanceConfig())
        engine.update_static_analysis([
            None,
            {"name": "sendmsg$rds", "weight": "bad",
             "guidance_role": "entry_exact"},
            {"name": "sendmsg$rds", "weight": float("nan"),
             "guidance_role": "entry_exact"},
            {"name": "sendmsg$rds", "weight": float("inf"),
             "guidance_role": "entry_exact"},
            {"name": "sendmsg$rds", "weight": True,
             "guidance_role": "entry_exact"},
            {"name": "sendmsg", "weight": 0.5,
             "guidance_role": "primitive_fallback"},
        ])

        guidance = engine.compute_guidance()

        self.assertEqual(guidance["syscall_weights"], {"sendmsg": 0.3})

    def test_empty_guidance_is_sent_to_clear_stale_fuzzer_state(self):
        engine = GuidanceEngine(GuidanceConfig(fuzzer_callback_addr="localhost:1"))
        guidance = engine.compute_guidance()
        applied = mock.Mock()
        applied.raise_for_status.return_value = None
        applied.json.return_value = {
            "status": "applied",
            "version": guidance["version"],
        }

        with mock.patch("requests.post", return_value=applied) as post:
            self.assertTrue(engine.send_guidance(guidance=guidance))

        post.assert_called_once()

    def test_successful_ack_retains_seed_injection_counts(self):
        engine = GuidanceEngine(GuidanceConfig(fuzzer_callback_addr="localhost:1"))
        engine.update_static_analysis([
            {"name": "syz_mount_image$f2fs", "weight": 1.0,
             "delivery_mode": "seed_only"},
        ])
        guidance = engine.compute_guidance()
        applied = mock.Mock()
        applied.raise_for_status.return_value = None
        applied.json.return_value = {
            "status": "applied",
            "version": guidance["version"],
            "accepted_weights": 0,
            "rejected_weights": 0,
            "accepted_templates": 0,
            "rejected_templates": 0,
            "accepted_seeds": 16,
            "rejected_seeds": 0,
        }

        with mock.patch("requests.post", return_value=applied):
            self.assertTrue(engine.send_guidance(guidance=guidance))

        self.assertEqual(engine.get_last_ack()["accepted_seeds"], 16)

    def test_pending_guidance_waits_for_applied_acknowledgement(self):
        engine = GuidanceEngine(GuidanceConfig(fuzzer_callback_addr="localhost:1"))
        engine.update_static_analysis([{"name": "sendmsg", "weight": 1.0}])
        guidance = engine.compute_guidance()
        pending = mock.Mock()
        pending.raise_for_status.return_value = None
        pending.json.return_value = {
            "status": "pending",
            "version": guidance["version"],
        }
        applied = mock.Mock()
        applied.raise_for_status.return_value = None
        applied.json.return_value = {
            "status": "already_applied",
            "version": guidance["version"],
        }

        with mock.patch(
                "requests.post", side_effect=[pending, applied]) as post, \
                mock.patch("time.sleep"):
            self.assertTrue(engine.send_guidance(guidance=guidance))

        self.assertEqual(post.call_count, 2)

    def test_pending_guidance_timeout_is_not_success(self):
        engine = GuidanceEngine(GuidanceConfig(
            fuzzer_callback_addr="localhost:1",
            pending_ack_timeout_seconds=0.5,
        ))
        engine.update_static_analysis([{"name": "sendmsg", "weight": 1.0}])
        guidance = engine.compute_guidance()
        pending = mock.Mock()
        pending.raise_for_status.return_value = None
        pending.json.return_value = {
            "status": "pending",
            "version": guidance["version"],
        }

        with mock.patch("requests.post", return_value=pending) as post, \
                mock.patch(
                    "time.monotonic", side_effect=[0.0, 0.0, 1.0]
                ):
            self.assertFalse(engine.send_guidance(guidance=guidance))

        post.assert_called_once()

    def test_canceled_guidance_does_not_issue_http_request(self):
        engine = GuidanceEngine(GuidanceConfig(fuzzer_callback_addr="localhost:1"))
        engine.update_static_analysis([{"name": "sendmsg", "weight": 1.0}])
        guidance = engine.compute_guidance()
        cancel_event = threading.Event()
        cancel_event.set()

        with mock.patch("requests.post") as post:
            self.assertFalse(engine.send_guidance(
                guidance=guidance, cancel_event=cancel_event
            ))

        post.assert_not_called()

if __name__ == "__main__":
    unittest.main()
