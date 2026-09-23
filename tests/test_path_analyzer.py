"""Regression tests for report-grounded exact Syzlang variant matching."""

import sys
import tempfile
import unittest
from pathlib import Path


BRAIN_DIR = Path(__file__).resolve().parents[1] / "brain"
sys.path.insert(0, str(BRAIN_DIR))

from path_analyzer import PathBasedAnalyzer


def report(*frames):
    return "Call Trace:\n" + "\n".join(frames) + "\n\n"


class PathBasedAnalyzerVariantTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        (root / "dev_kvm.txt").write_text(
            "ioctl$KVM_RUN(fd fd)\n"
            "ioctl$KVM_GET_VCPU_EVENTS(fd fd)\n"
            "ioctl$KVM_SET_VCPU_EVENTS(fd fd)\n",
            encoding="utf-8",
        )
        (root / "socket_netlink_generic_80211.txt").write_text(
            "sendmsg$NL80211_CMD_CONNECT(fd fd)\n"
            "sendmsg$NL80211_CMD_DISCONNECT(fd fd)\n",
            encoding="utf-8",
        )
        (root / "socket_rds.txt").write_text(
            "socket$rds(domain int32)\n"
            "bind$rds(fd fd)\n"
            "sendmsg$rds(fd fd)\n",
            encoding="utf-8",
        )
        (root / "socket_netlink.txt").write_text(
            "sendmsg$netlink(fd fd)\n",
            encoding="utf-8",
        )
        (root / "socket_netlink_route_sched.txt").write_text(
            "sendmsg$nl_route_sched(fd fd)\n",
            encoding="utf-8",
        )
        (root / "filesystem.txt").write_text(
            "syz_mount_image$f2fs(fs ptr[in, string[\"f2fs\"]])\n"
            "syz_mount_image$ext4(fs ptr[in, string[\"ext4\"]])\n",
            encoding="utf-8",
        )
        self.analyzer = PathBasedAnalyzer(str(root))

    def tearDown(self):
        self.temp_dir.cleanup()

    def names(self, report_text):
        return {entry["name"] for entry in self.analyzer.analyze_report(report_text)}

    def test_kvm_wrapper_refines_ioctl_to_matching_exact_variant(self):
        entries = self.analyzer.analyze_report(report(
            "kvm_vcpu_ioctl_x86_set_vcpu_events+0x1/0x2 arch/x86/kvm/x86.c:1",
            "__se_sys_ioctl+0x1/0x2 fs/ioctl.c:1",
        ))
        names = {entry["name"] for entry in entries}

        self.assertIn("ioctl", names)
        self.assertIn("ioctl$KVM_SET_VCPU_EVENTS", names)
        self.assertNotIn("ioctl$KVM_GET_VCPU_EVENTS", names)
        self.assertNotIn("ioctl$KVM_RUN", names)
        levels = {entry["name"]: entry["guidance_level"] for entry in entries}
        self.assertEqual(levels["ioctl"], "system_call")
        self.assertEqual(levels["ioctl$KVM_SET_VCPU_EVENTS"], "syz_call")
        roles = {entry["name"]: entry["guidance_role"] for entry in entries}
        self.assertEqual(roles["ioctl"], "primitive_fallback")
        self.assertEqual(roles["ioctl$KVM_SET_VCPU_EVENTS"], "entry_exact")

    def test_netlink_wrapper_selects_connect_not_disconnect(self):
        names = self.names(report(
            "nl80211_connect+0x1/0x2 net/wireless/sme.c:1",
            "____sys_sendmsg+0x1/0x2 net/socket.c:1",
        ))

        self.assertIn("sendmsg", names)
        self.assertIn("sendmsg$NL80211_CMD_CONNECT", names)
        self.assertNotIn("sendmsg$NL80211_CMD_DISCONNECT", names)

    def test_triple_underscore_wrapper_and_compiler_suffix_are_normalized(self):
        names = self.names(report(
            "nl80211_connect+0x1/0x2 net/wireless/sme.c:1",
            "___sys_sendmsg.constprop.7+0x1/0x2 net/socket.c:1",
        ))

        self.assertIn("sendmsg", names)
        self.assertIn("sendmsg$NL80211_CMD_CONNECT", names)

    def test_no_direct_primitive_does_not_infer_exact_variant(self):
        names = self.names(report(
            "kvm_vcpu_ioctl_x86_set_vcpu_events+0x1/0x2 arch/x86/kvm/x86.c:1",
        ))

        self.assertNotIn("ioctl$KVM_SET_VCPU_EVENTS", names)

    def test_mount_wrapper_and_fs_path_select_exact_image_variant(self):
        entries = self.analyzer.analyze_report(report(
            "f2fs_fill_super+0x1/0x2 fs/f2fs/super.c:1",
            "__se_sys_mount+0x1/0x2 fs/namespace.c:1",
        ))
        names = {entry["name"] for entry in entries}
        by_name = {entry["name"]: entry for entry in entries}

        self.assertIn("mount", names)
        self.assertIn("syz_mount_image$f2fs", names)
        self.assertNotIn("syz_mount_image$ext4", names)
        self.assertEqual(
            by_name["syz_mount_image$f2fs"]["guidance_level"], "syz_call"
        )
        self.assertEqual(
            by_name["syz_mount_image$f2fs"]["kernel_name"],
            "filesystem_variant:fs/f2fs/super.c",
        )

    def test_fs_path_without_mount_wrapper_does_not_select_image_variant(self):
        names = self.names(report(
            "f2fs_fill_super+0x1/0x2 fs/f2fs/super.c:1",
        ))

        self.assertNotIn("syz_mount_image$f2fs", names)

    def test_unique_source_family_refines_single_token_variant(self):
        entries = self.analyzer.analyze_report(report(
            "rds_sendmsg+0x1/0x2 net/rds/send.c:1",
            "____sys_sendmsg+0x1/0x2 net/socket.c:1",
        ))
        by_name = {entry["name"]: entry for entry in entries}

        self.assertIn("sendmsg", by_name)
        self.assertIn("sendmsg$rds", by_name)
        self.assertNotIn("sendmsg$inet", by_name)
        self.assertNotIn("sendmsg$inet6", by_name)
        self.assertEqual(by_name["sendmsg$rds"]["weight"], 0.9)
        self.assertEqual(
            by_name["sendmsg$rds"]["kernel_name"],
            "source_variant:sendmsg←net/rds/send.c→socket_rds",
        )
        self.assertEqual(
            by_name["sendmsg$rds"]["guidance_role"], "entry_exact"
        )

    def test_common_dispatcher_does_not_suppress_subsystem_variant(self):
        names = self.names(report(
            "qdisc_create+0x1/0x2 net/sched/sch_api.c:1",
            "netlink_sendmsg+0x1/0x2 net/netlink/af_netlink.c:1",
            "____sys_sendmsg+0x1/0x2 net/socket.c:1",
        ))

        self.assertIn("sendmsg", names)
        self.assertIn("sendmsg$netlink", names)
        self.assertIn("sendmsg$nl_route_sched", names)

    def test_extracts_abi_validated_ioctl_command_from_report_registers(self):
        report_text = """Call Trace:
 __x64_sys_ioctl+0x1/0x2 fs/ioctl.c:739
 do_syscall_64+0x1/0x2 arch/x86/entry/common.c:46
RIP: 0033:0x43fd49
Code: 0f 05
RSP: 002b:00007fff EFLAGS: 00000246 ORIG_RAX: 0000000000000010
RAX: ffffffffffffffda RBX: 0 RCX: 0
RDX: 0000000020000080 RSI: 0000000000004601 RDI: 0000000000000003
"""

        observation = self.analyzer.extract_report_dispatch_constant(
            report_text, "amd64"
        )

        self.assertIsNotNone(observation)
        self.assertEqual(observation.call_name, "ioctl")
        self.assertEqual(observation.syscall_nr, 0x10)
        self.assertEqual(observation.fixed_arg_index, 1)
        self.assertEqual(observation.value, 0x4601)
        self.assertEqual(observation.provenance, "report_register")

    def test_report_register_refinement_fails_closed_on_weak_evidence(self):
        base = """Call Trace:
 __x64_sys_ioctl+0x1/0x2 fs/ioctl.c:739
RIP: {rip}:0x43fd49
RSP: 002b:0 ORIG_RAX: {nr}
RDX: 0 RSI: {command} RDI: 3
"""
        kernel_rip = base.format(
            rip="0010", nr="0000000000000010",
            command="0000000000004601",
        )
        wrong_nr = base.format(
            rip="0033", nr="0000000000000001",
            command="0000000000004601",
        )
        no_wrapper = base.replace("__x64_sys_ioctl", "fb_ioctl").format(
            rip="0033", nr="0000000000000010",
            command="0000000000004601",
        )
        conflicting = (
            base.format(
                rip="0033", nr="0000000000000010",
                command="0000000000004601",
            ) + base.format(
                rip="0033", nr="0000000000000010",
                command="0000000000004602",
            )
        )
        cross_segment = (
            "Call Trace:\n"
            " __x64_sys_ioctl+0x1/0x2 fs/ioctl.c:739\n\n"
            "==================================================================\n"
            + base.replace("__x64_sys_ioctl", "__x64_sys_write").format(
                rip="0033", nr="0000000000000010",
                command="0000000000004601",
            )
        )

        for report_text, arch in (
                (kernel_rip, "amd64"),
                (wrong_nr, "amd64"),
                (no_wrapper, "amd64"),
                (conflicting, "amd64"),
                (cross_segment, "amd64"),
                (base.format(
                    rip="0033", nr="0000000000000010",
                    command="0000000000004601",
                ), "arm64")):
            with self.subTest(arch=arch, report_text=report_text[-80:]):
                self.assertIsNone(
                    self.analyzer.extract_report_dispatch_constant(
                        report_text, arch
                    )
                )

    def test_plain_sys_helper_is_not_treated_as_a_syscall_wrapper(self):
        names = self.names(report(
            "sys_imageblit+0x1/0x2 drivers/video/fbdev/core/sysimgblt.c:1",
        ))

        self.assertNotIn("imageblit", names)


if __name__ == "__main__":
    unittest.main()
