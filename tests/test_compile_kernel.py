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
                compile_kernel, "apply_kcov_remote_owner_fix", return_value=True
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

    def test_remote_kcov_fix_is_applied_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            kernel = Path(temp_dir)
            (kernel / "include/linux").mkdir(parents=True)
            (kernel / "include/uapi/linux").mkdir(parents=True)
            (kernel / "kernel").mkdir()
            (kernel / "include/uapi/linux/kcov.h").write_text(
                "#define KCOV_REMOTE_ENABLE _IOW('c', 102, unsigned long)\n"
            )
            (kernel / "include/linux/kcov.h").write_text(
                "enum kcov_mode {\n"
                "\tKCOV_MODE_TRACE_PC = 2,\n"
                "\t/* Collecting comparison operands mode. */\n"
                "\tKCOV_MODE_TRACE_CMP = 3,\n"
                "};\n\n"
                "#define KCOV_IN_CTXSW\t(1 << 30)\n"
            )
            (kernel / "kernel/kcov.c").write_text(
                "\t\tif (kcov->mode != KCOV_MODE_INIT || !t->kcov)\n"
                "\t\t\treturn -EINVAL;\n"
                "\t\tkcov->mode = mode;\n"
                "\t\tt->kcov = kcov;\n"
                "\t\tkcov->t = t;\n"
                "\t\tkcov->remote = true;\n"
                "\t\tkcov->remote_size = remote_arg->area_size;\n"
            )

            self.assertTrue(compile_kernel.apply_kcov_remote_owner_fix(kernel))
            self.assertFalse(compile_kernel.apply_kcov_remote_owner_fix(kernel))
            self.assertIn(
                "KCOV_MODE_REMOTE",
                (kernel / "include/linux/kcov.h").read_text(),
            )
            self.assertIn(
                "t->kcov_mode = KCOV_MODE_REMOTE",
                (kernel / "kernel/kcov.c").read_text(),
            )

    def test_remote_kcov_fix_skips_unsupported_kernel(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            kernel = Path(temp_dir)
            (kernel / "include/linux").mkdir(parents=True)
            (kernel / "include/uapi/linux").mkdir(parents=True)
            (kernel / "kernel").mkdir()
            (kernel / "include/linux/kcov.h").write_text("enum kcov_mode {};\n")
            (kernel / "include/uapi/linux/kcov.h").write_text("#define KCOV_INIT_TRACE 1\n")
            (kernel / "kernel/kcov.c").write_text("static int kcov_open(void) { return 0; }\n")

            with mock.patch.object(compile_kernel, "run_command") as run_command:
                self.assertFalse(compile_kernel.apply_kcov_remote_owner_fix(kernel))
            run_command.assert_not_called()

    def test_remote_kcov_fix_rejects_partial_backport(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            kernel = Path(temp_dir)
            (kernel / "include/linux").mkdir(parents=True)
            (kernel / "include/uapi/linux").mkdir(parents=True)
            (kernel / "kernel").mkdir()
            (kernel / "include/uapi/linux/kcov.h").write_text(
                "#define KCOV_REMOTE_ENABLE _IOW('c', 102, unsigned long)\n"
            )
            (kernel / "include/linux/kcov.h").write_text(
                "enum kcov_mode { KCOV_MODE_REMOTE };\n"
            )
            (kernel / "kernel/kcov.c").write_text("kcov->remote = true;\n")

            with self.assertRaisesRegex(RuntimeError, "inconsistent remote-KCOV"):
                compile_kernel.apply_kcov_remote_owner_fix(kernel)


if __name__ == "__main__":
    unittest.main()
