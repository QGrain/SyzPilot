"""Regression tests for Brain subprocess lifecycle helpers."""

import os
import select
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


BRAIN_DIR = Path(__file__).resolve().parents[1] / "brain"
sys.path.insert(0, str(BRAIN_DIR))

import utils as brain_utils


def read_child_pid(process):
    readable, _, _ = select.select([process.stdout], [], [], 5)
    if not readable:
        raise TimeoutError("launcher did not report its child PID")
    return int(process.stdout.readline().strip())


class ProcessLifecycleTest(unittest.TestCase):
    def test_kill_process_removes_real_parent_and_child_group(self):
        launcher_code = (
            "import subprocess,sys,time; "
            "child=subprocess.Popen([sys.executable,'-c',"
            "'import time; time.sleep(60)']); "
            "print(child.pid, flush=True); time.sleep(60)"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", launcher_code],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        process_group_id = process.pid
        try:
            child_pid = read_child_pid(process)
            self.assertTrue(
                brain_utils.kill_process(process, process_group=True)
            )
            with self.assertRaises(ProcessLookupError):
                os.killpg(process_group_id, 0)
            self.assertFalse(Path(f"/proc/{child_pid}").exists())
        finally:
            try:
                os.killpg(process_group_id, brain_utils.signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            process.stdout.close()

    def test_kill_process_terminates_isolated_process_group(self):
        process = mock.Mock(pid=12345)
        process.poll.return_value = None

        with (
            mock.patch.object(
                brain_utils.os, "getpgid", return_value=12345
            ),
            mock.patch.object(
                brain_utils.os,
                "killpg",
                side_effect=[None, ProcessLookupError],
            ) as killpg,
            mock.patch.object(brain_utils.time, "sleep"),
        ):
            self.assertTrue(
                brain_utils.kill_process(process, process_group=True)
            )

        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(12345, brain_utils.signal.SIGTERM),
                mock.call(12345, 0),
            ],
        )
        process.poll.assert_called_once_with()

    def test_kill_process_rejects_non_leader_process(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
        )
        try:
            self.assertNotEqual(os.getpgid(process.pid), process.pid)
            self.assertFalse(
                brain_utils.kill_process(process, process_group=True)
            )
            self.assertIsNone(process.poll())
        finally:
            process.kill()
            process.wait(timeout=5)

    def test_kill_process_escalates_real_group_ignoring_sigterm(self):
        launcher_code = (
            "import signal,subprocess,sys,time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "child=subprocess.Popen([sys.executable,'-c',"
            "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(60)']); "
            "print(child.pid, flush=True); time.sleep(60)"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", launcher_code],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        process_group_id = process.pid
        try:
            child_pid = read_child_pid(process)
            self.assertTrue(
                brain_utils.kill_process(process, process_group=True)
            )
            with self.assertRaises(ProcessLookupError):
                os.killpg(process_group_id, 0)
            self.assertFalse(Path(f"/proc/{child_pid}").exists())
        finally:
            try:
                os.killpg(process_group_id, brain_utils.signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
            process.stdout.close()

    def test_kill_process_escalates_lingering_process_group(self):
        process = mock.Mock(pid=12345)
        process.poll.return_value = None
        monotonic_values = iter((0, 6, 6, 6.1))

        with (
            mock.patch.object(
                brain_utils.os, "getpgid", return_value=12345
            ),
            mock.patch.object(
                brain_utils.os,
                "killpg",
                side_effect=[None, None, ProcessLookupError],
            ) as killpg,
            mock.patch.object(
                brain_utils.time,
                "monotonic",
                side_effect=lambda: next(monotonic_values),
            ),
        ):
            self.assertTrue(
                brain_utils.kill_process(process, process_group=True)
            )

        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(12345, brain_utils.signal.SIGTERM),
                mock.call(12345, brain_utils.signal.SIGKILL),
                mock.call(12345, 0),
            ],
        )
        process.poll.assert_called_once_with()

    def test_kill_process_reports_group_signal_failure(self):
        process = mock.Mock(pid=12345)

        with (
            mock.patch.object(
                brain_utils.os, "getpgid", return_value=12345
            ),
            mock.patch.object(
                brain_utils.os, "killpg", side_effect=PermissionError
            ),
        ):
            self.assertFalse(
                brain_utils.kill_process(process, process_group=True)
            )

    def test_kill_process_accepts_exit_between_poll_and_terminate(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.terminate.side_effect = ProcessLookupError

        self.assertTrue(brain_utils.kill_process(process))
        process.terminate.assert_called_once_with()

    def test_kill_process_reports_non_group_signal_failure(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.terminate.side_effect = PermissionError

        with self.assertRaises(PermissionError):
            brain_utils.kill_process(process)


if __name__ == "__main__":
    unittest.main()
