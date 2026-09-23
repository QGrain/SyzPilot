"""Tests for the compiled Syzlang manifest trust boundary."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


BRAIN_DIR = Path(__file__).resolve().parents[1] / "brain"
sys.path.insert(0, str(BRAIN_DIR))

from syzlang_manifest import SyzlangIndex  # noqa: E402


def manifest_document():
    def call(name, call_name, delivery, **attrs):
        return {
            "name": name,
            "call_name": call_name,
            "nr": attrs.get("nr", "0x10"),
            "fixed_arguments": attrs.get("fixed_arguments", []),
            "delivery_mode": delivery,
            "attrs": {
                "disabled": attrs.get("disabled", False),
                "no_generate": attrs.get("no_generate", False),
                "automatic": attrs.get("automatic", False),
                "automatic_helper": attrs.get("automatic_helper", False),
            },
            "required_resources": attrs.get("required_resources", []),
            "declared_output_resources": attrs.get(
                "declared_output_resources", []
            ),
        }

    return {
        "schema": "syzpilot.compiled-syzlang-manifest",
        "schema_version": 2,
        "target": {"os": "linux", "arch": "amd64", "revision": "target-r1"},
        "producer": {"syzkaller_git_revision": "fuzzer-r1"},
        "calls": [
            call("ioctl", "ioctl", "generatable"),
            call("ioctl$AUTO", "ioctl", "generatable", automatic=True),
            call("ioctl$DISABLED", "ioctl", "disabled", disabled=True),
            call("syz_mount_image$f2fs", "syz_mount_image", "seed_only",
                 no_generate=True),
        ],
        "constants": [],
        "resource_dependencies": [],
    }


class SyzlangManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "manifest.json"
        self.write(manifest_document())

    def tearDown(self):
        self.temp_dir.cleanup()

    def write(self, document):
        self.path.write_text(
            json.dumps(document, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def load(self, **kwargs):
        return SyzlangIndex.load(
            self.path,
            expected_os="linux",
            expected_arch="amd64",
            **kwargs,
        )

    def test_indexes_exact_names_and_variants(self):
        index = self.load(expected_revision="target-r1")

        self.assertEqual(index.get("ioctl").delivery_mode, "generatable")
        self.assertEqual(
            [record.name for record in index.variants("ioctl")],
            ["ioctl", "ioctl$AUTO", "ioctl$DISABLED"],
        )
        self.assertEqual(index.identity["target_revision"], "target-r1")
        self.assertEqual(len(index.identity["sha256"]), 64)

    def test_filter_routes_seed_only_and_rejects_noncompiled_calls(self):
        index = self.load()
        result = index.filter_candidates([
            {"name": "ioctl", "weight": 1.0},
            {"name": "syz_mount_image$f2fs", "weight": 0.9},
            {"name": "ioctl$DISABLED", "weight": 0.8},
            {"name": "ioctl$NOT_COMPILED", "weight": 0.7},
        ])

        self.assertEqual(
            [(entry["name"], entry["delivery_mode"])
             for entry in result.entries],
            [("ioctl", "generatable"),
             ("syz_mount_image$f2fs", "seed_only")],
        )
        self.assertEqual(result.rejected, {"disabled": 1, "unknown": 1})
        self.assertTrue(all(
            entry["runtime_eligibility"] == "unknown"
            for entry in result.entries
        ))

    def test_description_mode_and_runtime_enabled_filters_are_explicit(self):
        index = self.load()
        result = index.filter_candidates(
            [{"name": "ioctl"}, {"name": "ioctl$AUTO"}],
            descriptions_mode="manual",
            enabled_names=frozenset({"ioctl$AUTO"}),
        )

        self.assertEqual(result.entries, ())
        self.assertEqual(
            result.rejected,
            {"description_mode": 1, "runtime_disabled": 1},
        )

    def test_generation_scores_keep_only_compiled_generatable_calls(self):
        index = self.load()

        result = index.filter_generation_scores({
            "ioctl": 3.0,
            "ioctl$AUTO": 2.0,
            "ioctl$DISABLED": 4.0,
            "syz_mount_image$f2fs": 5.0,
            "ioctl$NOT_COMPILED": 6.0,
            "zero": 0.0,
            "nan": float("nan"),
        }, descriptions_mode="manual")

        self.assertEqual(result.entries, ({
            "name": "ioctl",
            "weight": 3.0,
            "delivery_mode": "generatable",
        },))
        self.assertEqual(result.rejected, {
            "description_mode": 1,
            "disabled": 1,
            "malformed": 2,
            "seed_only": 1,
            "unknown": 1,
        })

    def test_rejects_delivery_attribute_contradiction(self):
        document = manifest_document()
        document["calls"][0]["delivery_mode"] = "seed_only"
        self.write(document)

        with self.assertRaisesRegex(ValueError, "contradicts"):
            self.load()

    def test_rejects_target_mismatch_unsorted_calls_and_symlink(self):
        with self.assertRaisesRegex(ValueError, "target mismatch"):
            SyzlangIndex.load(
                self.path, expected_os="linux", expected_arch="arm64"
            )

        with self.assertRaisesRegex(ValueError, "producer revision mismatch"):
            SyzlangIndex.load(
                self.path, expected_os="linux", expected_arch="amd64",
                expected_producer_revision="another-fuzzer",
            )

        document = manifest_document()
        document["calls"].reverse()
        self.write(document)
        with self.assertRaisesRegex(ValueError, "not sorted"):
            self.load()

        link = Path(self.temp_dir.name) / "manifest-link.json"
        os.symlink(self.path, link)
        with self.assertRaises(OSError):
            SyzlangIndex.load(
                link, expected_os="linux", expected_arch="amd64"
            )

    def test_rejects_file_over_size_limit(self):
        with self.assertRaisesRegex(ValueError, "size"):
            self.load(max_bytes=8)

    def test_builds_bounded_rds_resource_and_state_templates(self):
        document = manifest_document()
        document["calls"] = [
            {
                **document["calls"][0],
                "name": "bind$rds", "call_name": "bind",
                "required_resources": ["sock_rds"],
            },
            {
                **document["calls"][0],
                "name": "connect$rds", "call_name": "connect",
                "required_resources": ["sock_rds"],
            },
            {
                **document["calls"][0],
                "name": "sendmsg$rds", "call_name": "sendmsg",
                "required_resources": ["sock_rds"],
            },
            {
                **document["calls"][0],
                "name": "socket$rds", "call_name": "socket",
                "declared_output_resources": ["sock_rds"],
            },
        ]
        document["resource_dependencies"] = [{
            "name": "sock_rds",
            "precise_constructors": ["socket$rds"],
        }]
        self.write(document)
        index = self.load()

        result = index.build_report_static_templates([
            {"name": "sendmsg$rds", "source": "path_analysis",
             "guidance_role": "entry_exact", "weight": 1.0},
            {"name": "bind$rds", "source": "path_analysis",
             "guidance_role": "subsystem_peer", "weight": 0.8},
            {"name": "connect$rds", "source": "path_analysis",
             "guidance_role": "subsystem_peer", "weight": 0.7},
        ])

        self.assertEqual(
            [template["syscalls"] for template in result.templates],
            [
                ["socket$rds", "sendmsg$rds"],
                ["socket$rds", "bind$rds", "sendmsg$rds"],
                ["socket$rds", "connect$rds", "sendmsg$rds"],
            ],
        )
        self.assertEqual(
            [entry["name"] for entry in result.producer_entries],
            ["socket$rds"],
        )

    def test_orders_netlink_constructors_before_exact_entry(self):
        document = manifest_document()
        base = document["calls"][0]
        document["calls"] = [
            {
                **base,
                "name": "ioctl$sock_SIOCGIFINDEX_80211",
                "call_name": "ioctl",
                "required_resources": ["sock"],
                "declared_output_resources": ["nl80211_ifindex"],
            },
            {
                **base,
                "name": "sendmsg$NL80211_CMD_CONNECT",
                "call_name": "sendmsg",
                "required_resources": [
                    "nl80211_family_id", "nl80211_ifindex",
                    "sock_nl_generic",
                ],
            },
            {
                **base,
                "name": "socket$nl_generic", "call_name": "socket",
                "declared_output_resources": ["sock_nl_generic"],
            },
            {
                **base,
                "name": "syz_genetlink_get_family_id$nl80211",
                "call_name": "syz_genetlink_get_family_id",
                "required_resources": ["sock_nl_generic"],
                "declared_output_resources": ["nl80211_family_id"],
            },
        ]
        document["resource_dependencies"] = [
            {"name": "nl80211_family_id", "precise_constructors": [
                "syz_genetlink_get_family_id$nl80211",
            ]},
            {"name": "nl80211_ifindex", "precise_constructors": [
                "ioctl$sock_SIOCGIFINDEX_80211",
            ]},
            {"name": "sock", "precise_constructors": [
                "socket$nl_generic",
            ]},
            {"name": "sock_nl_generic", "precise_constructors": [
                "socket$nl_generic",
            ]},
        ]
        self.write(document)
        index = self.load()

        result = index.build_report_static_templates([{
            "name": "sendmsg$NL80211_CMD_CONNECT",
            "source": "path_analysis",
            "guidance_role": "entry_exact",
            "weight": 1.0,
        }])

        self.assertEqual(result.templates[0]["syscalls"], [
            "socket$nl_generic",
            "ioctl$sock_SIOCGIFINDEX_80211",
            "syz_genetlink_get_family_id$nl80211",
            "sendmsg$NL80211_CMD_CONNECT",
        ])

    def test_static_templates_require_report_exact_evidence(self):
        index = self.load()

        result = index.build_report_static_templates([
            {"name": "ioctl", "source": "kallgraph",
             "guidance_role": "entry_exact", "weight": 1.0},
            {"name": "ioctl$AUTO", "source": "path_analysis",
             "guidance_role": "subsystem_peer", "weight": 1.0},
        ])

        self.assertEqual(result.templates, ())
        self.assertEqual(result.producer_entries, ())

    def test_refines_path_candidate_with_unique_named_report_constant(self):
        document = manifest_document()
        base = document["calls"][0]
        document["calls"] = sorted([
            base,
            {
                **base,
                "name": "ioctl$FBIOGET_VSCREENINFO",
                "fixed_arguments": [{"index": 1, "value": "0x4600"}],
                "required_resources": ["fd_fb"],
            },
            {
                **base,
                "name": "ioctl$FBIOPUT_VSCREENINFO",
                "fixed_arguments": [{"index": 1, "value": "0x4601"}],
                "required_resources": ["fd_fb"],
            },
            {
                **base,
                "name": "openat$fb0", "call_name": "openat",
                "nr": "0x101",
                "declared_output_resources": ["fd_fb"],
            },
        ], key=lambda item: item["name"])
        document["constants"] = [
            {"name": "FBIOGET_VSCREENINFO", "value": "0x4600"},
            {"name": "FBIOPUT_VSCREENINFO", "value": "0x4601"},
        ]
        document["resource_dependencies"] = [{
            "name": "fd_fb", "precise_constructors": ["openat$fb0"],
        }]
        self.write(document)
        index = self.load()

        result = index.refine_candidates_by_report_constant([
            {"name": "ioctl", "source": "path_analysis",
             "guidance_role": "primitive_fallback", "weight": 1.0},
            {"name": "ioctl$FBIOGET_VSCREENINFO",
             "source": "path_analysis", "guidance_role": "subsystem_peer",
             "weight": 0.75},
            {"name": "ioctl$FBIOPUT_VSCREENINFO",
             "source": "path_analysis", "guidance_role": "subsystem_peer",
             "weight": 0.75},
            {"name": "openat$fb0", "source": "path_analysis",
             "guidance_role": "subsystem_peer", "weight": 0.75},
        ], call_name="ioctl", syscall_nr=0x10, fixed_arg_index=1,
            value=0x4601,
            descriptions_mode="manual")

        self.assertEqual(result.status, "matched")
        self.assertEqual(result.matched_name, "ioctl$FBIOPUT_VSCREENINFO")
        self.assertEqual(
            [entry["name"] for entry in result.entries],
            ["ioctl", "ioctl$FBIOPUT_VSCREENINFO", "openat$fb0"],
        )
        exact = result.entries[1]
        self.assertEqual(exact["guidance_role"], "entry_exact")
        self.assertEqual(exact["weight"], 0.99)
        self.assertEqual(exact["kernel_name"], "report_register:ioctl:0x4601")

        templates = index.build_report_static_templates(
            result.entries, descriptions_mode="manual"
        )
        self.assertEqual(
            templates.templates[0]["syscalls"],
            ["openat$fb0", "ioctl$FBIOPUT_VSCREENINFO"],
        )

    def test_report_constant_ambiguity_and_exact_conflict_fail_closed(self):
        document = manifest_document()
        base = document["calls"][0]
        document["calls"] = sorted([
            base,
            {**base, "name": "ioctl$CMD_A", "fixed_arguments": [
                {"index": 1, "value": "0x42"},
            ]},
            {**base, "name": "ioctl$CMD_B", "fixed_arguments": [
                {"index": 1, "value": "0x42"},
            ]},
        ], key=lambda item: item["name"])
        document["constants"] = [
            {"name": "CMD_A", "value": "0x42"},
            {"name": "CMD_B", "value": "0x42"},
        ]
        self.write(document)
        index = self.load()
        entries = [
            {"name": "ioctl$CMD_A", "source": "path_analysis",
             "guidance_role": "subsystem_peer"},
            {"name": "ioctl$CMD_B", "source": "path_analysis",
             "guidance_role": "subsystem_peer"},
        ]

        ambiguous = index.refine_candidates_by_report_constant(
            entries, call_name="ioctl", syscall_nr=0x10,
            fixed_arg_index=1, value=0x42
        )
        self.assertEqual(ambiguous.status, "ambiguous")
        self.assertEqual(ambiguous.entries, tuple(entries))

        conflict_entries = [
            {"name": "ioctl$CMD_A", "source": "path_analysis",
             "guidance_role": "subsystem_peer"},
            {"name": "ioctl$CMD_B", "source": "path_analysis",
             "guidance_role": "entry_exact"},
        ]
        document["constants"][1]["value"] = "0x43"
        self.write(document)
        index = self.load()
        conflict = index.refine_candidates_by_report_constant(
            conflict_entries, call_name="ioctl", syscall_nr=0x10,
            fixed_arg_index=1, value=0x42
        )
        self.assertEqual(conflict.status, "conflicting_exact_evidence")
        self.assertEqual(conflict.entries, tuple(conflict_entries))

    def test_report_constant_requires_matching_compiled_argument_position(self):
        document = manifest_document()
        base = document["calls"][0]
        document["calls"] = sorted([
            base,
            {
                **base,
                "name": "ioctl$CMD",
                "fixed_arguments": [{"index": 1, "value": "0x99"}],
            },
        ], key=lambda item: item["name"])
        document["constants"] = [{"name": "CMD", "value": "0x42"}]
        self.write(document)
        index = self.load()
        entries = [{
            "name": "ioctl$CMD", "source": "path_analysis",
            "guidance_role": "subsystem_peer",
        }]

        wrong_value = index.refine_candidates_by_report_constant(
            entries, call_name="ioctl", syscall_nr=0x10,
            fixed_arg_index=1, value=0x42,
        )
        document["calls"][1]["fixed_arguments"] = [
            {"index": 1, "value": "0x42"},
        ]
        self.write(document)
        index = self.load()
        wrong_index = index.refine_candidates_by_report_constant(
            entries, call_name="ioctl", syscall_nr=0x10,
            fixed_arg_index=0, value=0x42,
        )

        self.assertEqual(wrong_value.status, "unsupported_by_report_path")
        self.assertEqual(wrong_value.entries, tuple(entries))
        self.assertEqual(wrong_index.status, "unsupported_by_report_path")
        self.assertEqual(wrong_index.entries, tuple(entries))

    def test_rejects_invalid_compiled_scalar_metadata(self):
        document = manifest_document()
        document["calls"][0]["fixed_arguments"] = [
            {"index": 1, "value": "0x1"},
            {"index": 1, "value": "0x2"},
        ]
        self.write(document)

        with self.assertRaisesRegex(ValueError, "fixed_arguments"):
            self.load()

        document = manifest_document()
        document["calls"][0]["nr"] = "0x010"
        self.write(document)
        with self.assertRaisesRegex(ValueError, "syscall number"):
            self.load()

        document = manifest_document()
        document["constants"] = [{"name": "CMD", "value": "0X42"}]
        self.write(document)
        with self.assertRaisesRegex(ValueError, "constant CMD"):
            self.load()


if __name__ == "__main__":
    unittest.main()
