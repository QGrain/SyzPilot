import tempfile
import unittest
from pathlib import Path
from unittest import mock

from analyzer import extract_benchmark_waypoints


class ExtractBenchmarkWaypointsTests(unittest.TestCase):
    def test_discover_case_ids_is_sorted_and_ignores_other_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)
            for name in ('case_10.title', 'case_2.title', 'README.title'):
                (config_dir / name).write_text('title', encoding='utf-8')

            self.assertEqual(
                extract_benchmark_waypoints.discover_case_ids(config_dir),
                [2, 10],
            )

    def test_portable_target_removes_kernel_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            kernel_dir = Path(temp_dir) / 'case_1'
            source_path = kernel_dir / 'net' / 'demo.c'
            source_path.parent.mkdir(parents=True)
            source_path.write_text('int demo;\n', encoding='utf-8')

            target = extract_benchmark_waypoints.portable_target(
                f'demo@{source_path}:1', kernel_dir
            )

            self.assertEqual(target, 'demo@net/demo.c:1')

    def test_portable_target_rejects_source_outside_kernel_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            kernel_dir = root / 'case_1'
            kernel_dir.mkdir()

            with self.assertRaisesRegex(ValueError, 'outside the kernel tree'):
                extract_benchmark_waypoints.portable_target(
                    f'demo@{root.parent / "outside.c"}:1', kernel_dir
                )

    def test_extract_case_writes_portable_resolution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark_dir = root / 'benchmark'
            config_dir = benchmark_dir / 'configs'
            kernel_root = root / 'cases'
            kernel_dir = kernel_root / 'case_7'
            source_path = kernel_dir / 'fs' / 'demo.c'
            config_dir.mkdir(parents=True)
            source_path.parent.mkdir(parents=True)
            source_path.write_text('int demo;\n', encoding='utf-8')
            (kernel_dir / 'vmlinux').write_bytes(b'vmlinux')
            (config_dir / 'case_7.title').write_text(
                'WARNING in demo\n', encoding='utf-8'
            )
            (config_dir / 'case_7.report').write_text(
                'demo+0x10/0x20 fs/demo.c:1\n', encoding='utf-8'
            )
            (config_dir / 'case_7.commit').write_text(
                'deadbeef\n', encoding='utf-8'
            )
            extractor = mock.Mock()
            extractor.get_pcs.return_value = (
                [f'demo@{source_path}:1'],
                ['0xffffffff81000082'],
            )
            extractor.pc_resolution_methods = [
                'report-relative-outer-function-predecessor'
            ]
            extractor.report_target = f'demo@{source_path}:1'
            extractor.final_target_kind = 'report-frame'

            with mock.patch.object(
                extract_benchmark_waypoints,
                'WaypointsExtractor',
                return_value=extractor,
            ), mock.patch.object(
                extract_benchmark_waypoints,
                'get_vmlinux_cache_identity',
                return_value='build-id:1234',
            ):
                result = extract_benchmark_waypoints.extract_case(
                    7, benchmark_dir, kernel_root
                )

            extractor.extract.assert_called_once_with()
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(result['waypoints'][0]['target'], 'demo@fs/demo.c:1')
            self.assertEqual(result['final_target_kind'], 'report-frame')
            self.assertEqual(result['target_pcs'], ['0x81000082'])
            self.assertNotIn(temp_dir, str(result))

    def test_extract_case_skips_missing_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_dir = root / 'benchmark' / 'configs'
            kernel_dir = root / 'cases' / 'case_62'
            config_dir.mkdir(parents=True)
            kernel_dir.mkdir(parents=True)
            (config_dir / 'case_62.title').write_text('title', encoding='utf-8')
            (config_dir / 'case_62.commit').write_text('commit', encoding='utf-8')
            (kernel_dir / 'vmlinux').write_bytes(b'vmlinux')

            result = extract_benchmark_waypoints.extract_case(
                62, root / 'benchmark', root / 'cases'
            )

            self.assertEqual(result['status'], 'skipped')
            self.assertIn('case_62.report', result['reason'])

    def test_extract_case_rejects_nonportable_kernel_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark_dir = root / 'benchmark'
            config_dir = benchmark_dir / 'configs'
            kernel_root = root / 'cases'
            kernel_dir = kernel_root / 'case_7'
            source_path = kernel_dir / 'fs' / 'demo.c'
            config_dir.mkdir(parents=True)
            source_path.parent.mkdir(parents=True)
            source_path.write_text('int demo;\n', encoding='utf-8')
            (kernel_dir / 'vmlinux').write_bytes(b'vmlinux')
            for suffix, value in (
                ('title', 'title'),
                ('report', 'demo+0x1/0x2 fs/demo.c:1'),
                ('commit', 'deadbeef'),
            ):
                (config_dir / f'case_7.{suffix}').write_text(
                    value, encoding='utf-8'
                )
            extractor = mock.Mock()
            extractor.get_pcs.return_value = (
                [f'demo@{source_path}:1'],
                ['0xffffffff81000082'],
            )
            extractor.pc_resolution_methods = [
                'report-relative-outer-function-predecessor'
            ]
            extractor.report_target = f'demo@{source_path}:1'
            extractor.final_target_kind = 'report-frame'

            with mock.patch.object(
                extract_benchmark_waypoints,
                'WaypointsExtractor',
                return_value=extractor,
            ), mock.patch.object(
                extract_benchmark_waypoints,
                'get_vmlinux_cache_identity',
                return_value='stat:local-only',
            ):
                with self.assertRaisesRegex(ValueError, 'GNU ELF build ID'):
                    extract_benchmark_waypoints.extract_case(
                        7, benchmark_dir, kernel_root
                    )


if __name__ == '__main__':
    unittest.main()
