import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments import compile_kernel


class CompileKernelTest(unittest.TestCase):
    def test_selected_compiler_is_used_for_kernel_and_host_tools(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "linux"
            source.mkdir()
            (source / ".git").mkdir()
            config = root / "case.config"
            config.write_text("CONFIG_64BIT=y\n")
            compiler = "/opt/toolchains/gcc-11.4/bin/gcc"

            with mock.patch.object(
                compile_kernel, "select_source_repo", return_value=(source, "linux")
            ), mock.patch.object(
                compile_kernel, "run_command", return_value=(0, "", "")
            ) as run_command, mock.patch.object(
                compile_kernel.subprocess,
                "run",
                return_value=mock.Mock(returncode=0),
            ) as subprocess_run:
                result = compile_kernel.compile_single_kernel(
                    case_name="case_test",
                    commit_hash="deadbeef",
                    config_path=str(config),
                    workdir=root / "builds",
                    linux_git_master=source,
                    jobs=2,
                    compiler=compiler,
                )

            self.assertTrue(result.success)
            commands = [call.args[0] for call in run_command.call_args_list]
            self.assertIn(
                [
                    "make",
                    f"CC={compiler}",
                    f"HOSTCC={compiler}",
                    "olddefconfig",
                ],
                commands,
            )
            subprocess_run.assert_called_once_with(
                [
                    "make",
                    f"CC={compiler}",
                    f"HOSTCC={compiler}",
                    "-j2",
                ],
                cwd=str(root / "builds" / "case_test"),
                stdout=mock.ANY,
                stderr=compile_kernel.subprocess.STDOUT,
                text=True,
                check=False,
            )


if __name__ == "__main__":
    unittest.main()
