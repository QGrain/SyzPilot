import os
import json
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from analyzer import get_targets


class GetTargetsTreeSitterTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.kernel_dir = self._tmpdir.name
        get_targets.SOURCE_FUNCTION_CACHE.clear()
        get_targets.TREE_SITTER_C_PARSER = None

    def tearDown(self):
        self._tmpdir.cleanup()

    def write_source(self, rel_path: str, content: str) -> str:
        abs_path = os.path.join(self.kernel_dir, rel_path)
        Path(abs_path).parent.mkdir(parents=True, exist_ok=True)
        Path(abs_path).write_text(textwrap.dedent(content).lstrip('\n'))
        return abs_path

    def test_function_pointer_multiline_signature(self):
        self.write_source(
            'lib/idr.c',
            r'''
            int idr_for_each(const struct idr *idr,
                    int (*fn)(int id, void *p, void *data), void *data)
            {
                return fn(1, 0, data);
            }
            ''',
        )
        resolved = get_targets.resolve_function_for_location(self.kernel_dir, 'lib/idr.c:2')
        self.assertEqual(
            resolved,
            f'idr_for_each@{os.path.join(self.kernel_dir, "lib/idr.c")}:2',
        )

    def test_multiline_macro_braces_do_not_break_following_short_function(self):
        self.write_source(
            'drivers/video/fbdev/vga16fb.c',
            r'''
            #define BAD_MACRO(x) \
                do {             \
                    if (x) {     \
                        (x)++;   \
                    }            \
                } while (0)

            static void vga16fb_imageblit(struct fb_info *info, const struct fb_image *image)
            {
                if (image->depth == 1)
                    vga_imageblit_expand(info, image);
                else
                    vga_imageblit_color(info, image);
            }
            ''',
        )
        resolved = get_targets.resolve_function_for_location(
            self.kernel_dir, 'drivers/video/fbdev/vga16fb.c:10'
        )
        self.assertEqual(
            resolved,
            f'vga16fb_imageblit@{os.path.join(self.kernel_dir, "drivers/video/fbdev/vga16fb.c")}:10',
        )

    def test_strings_comments_and_adjacent_functions(self):
        self.write_source(
            'fs/demo.c',
            r'''
            static int first_fn(void)
            {
                const char *s = "{ not a block }";
                /* comment with } { */
                return 0;
            }

            static int second_fn(void) { return first_fn(); }
            ''',
        )
        ranges = get_targets.get_file_function_ranges(os.path.join(self.kernel_dir, 'fs/demo.c'))
        self.assertEqual(
            [(item['name'], item['start_line'], item['end_line']) for item in ranges],
            [('first_fn', 1, 6), ('second_fn', 8, 8)],
        )
        resolved = get_targets.resolve_function_for_location(self.kernel_dir, 'fs/demo.c:8')
        self.assertEqual(resolved, f'second_fn@{os.path.join(self.kernel_dir, "fs/demo.c")}:8')

    def test_nested_preproc_toplevel_functions(self):
        self.write_source(
            'kernel/demo.c',
            r'''
            #ifdef CONFIG_A
            static int enabled_only(void)
            {
                return 1;
            }
            #else
            static int disabled_only(void)
            {
                return 0;
            }
            #endif
            ''',
        )
        ranges = get_targets.get_file_function_ranges(os.path.join(self.kernel_dir, 'kernel/demo.c'))
        self.assertEqual(
            [(item['name'], item['start_line'], item['end_line']) for item in ranges],
            [('enabled_only', 2, 5), ('disabled_only', 7, 10)],
        )

    def test_toplevel_error_subtree_still_collects_later_functions(self):
        self.write_source(
            'net/core/dev.c',
            r'''
            const char *netdev_cmd_to_name(enum netdev_cmd cmd)
            {
            #define N(val)                         \
                case NETDEV_##val:                 \
                    return "NETDEV_" __stringify(val);
                switch (cmd) {
                N(UP) N(DOWN) N(REBOOT) N(CHANGE) N(REGISTER) N(UNREGISTER)
                N(CHANGEMTU) N(CHANGEADDR) N(GOING_DOWN) N(CHANGENAME) N(FEAT_CHANGE)
                N(BONDING_FAILOVER) N(PRE_UP) N(PRE_TYPE_CHANGE) N(POST_TYPE_CHANGE)
                N(POST_INIT) N(RELEASE) N(NOTIFY_PEERS) N(JOIN) N(CHANGEUPPER)
                N(RESEND_IGMP) N(PRECHANGEMTU) N(CHANGEINFODATA) N(BONDING_INFO)
                N(PRECHANGEUPPER) N(CHANGELOWERSTATE) N(UDP_TUNNEL_PUSH_INFO)
                N(UDP_TUNNEL_DROP_INFO) N(CHANGE_TX_QUEUE_LEN)
                N(CVLAN_FILTER_PUSH_INFO) N(CVLAN_FILTER_DROP_INFO)
                N(SVLAN_FILTER_PUSH_INFO) N(SVLAN_FILTER_DROP_INFO)
                N(PRE_CHANGEADDR)
                }
            #undef N
                return "UNKNOWN_NETDEV_EVENT";
            }
            EXPORT_SYMBOL_GPL(netdev_cmd_to_name);

            static int call_netdevice_notifier(struct notifier_block *nb, unsigned long val,
                                               struct net_device *dev)
            {
                struct netdev_notifier_info info = {
                    .dev = dev,
                };

                return nb->notifier_call(nb, val, &info);
            }
            ''',
        )
        resolved = get_targets.resolve_function_for_location(self.kernel_dir, 'net/core/dev.c:23')
        self.assertEqual(
            resolved,
            f'call_netdevice_notifier@{os.path.join(self.kernel_dir, "net/core/dev.c")}:23',
        )

    def test_error_subtree_recovers_fragmented_function_after_gnu_do_while_macro(self):
        self.write_source(
            'kernel/cgroup_demo.c',
            r'''
            int first_fn(int leader, int task)
            {
                do {
                    leader += task;
                } while_each_thread(leader, task);
                return leader;
            }

            static int recovered_fn(int x,
                                    int y)
            {
                return x + y;
            }
            ''',
        )
        resolved = get_targets.resolve_function_for_location(self.kernel_dir, 'kernel/cgroup_demo.c:11')
        self.assertEqual(
            resolved,
            f'recovered_fn@{os.path.join(self.kernel_dir, "kernel/cgroup_demo.c")}:11',
        )

    def test_nested_error_inside_function_definition_recovers_later_function(self):
        self.write_source(
            'kernel/cgroup_nested_demo.c',
            r'''
            int first_fn(int leader, int task)
            {
                do {
                    leader += task;
                } while_each_thread(leader, task);

                leader += 1;
                return leader;
            }

            static int recovered_fn(int x)
            {
                return x + 1;
            }
            ''',
        )
        resolved = get_targets.resolve_function_for_location(self.kernel_dir, 'kernel/cgroup_nested_demo.c:12')
        self.assertEqual(
            resolved,
            f'recovered_fn@{os.path.join(self.kernel_dir, "kernel/cgroup_nested_demo.c")}:12',
        )

    def test_error_recovery_does_not_treat_parameter_names_as_functions(self):
        self.write_source(
            'kernel/cgroup_param_demo.c',
            r'''
            int first_fn(int leader, int task)
            {
                do {
                    leader += task;
                } while_each_thread(leader, task);

                leader += 1;
                return leader;
            }

            static int cgroup_events_show(struct seq_file *seq, void *v)
            {
                seq_printf(seq, "%d\n", 1);
                return 0;
            }

            static ssize_t cgroup_io_pressure_write(struct kernfs_open_file *of,
                                                    char *buf, size_t nbytes,
                                                    loff_t off)
            {
                return nbytes;
            }
            ''',
        )
        events_target = get_targets.resolve_function_for_location(
            self.kernel_dir, 'kernel/cgroup_param_demo.c:13'
        )
        io_target = get_targets.resolve_function_for_location(
            self.kernel_dir, 'kernel/cgroup_param_demo.c:21'
        )
        self.assertEqual(
            events_target,
            f'cgroup_events_show@{os.path.join(self.kernel_dir, "kernel/cgroup_param_demo.c")}:13',
        )
        self.assertEqual(
            io_target,
            f'cgroup_io_pressure_write@{os.path.join(self.kernel_dir, "kernel/cgroup_param_demo.c")}:21',
        )

    def test_syscall_define_macro_fallback(self):
        self.write_source(
            'kernel/sys_demo.c',
            r'''
            SYSCALL_DEFINE3(landlock_create_ruleset,
                    const struct foo __user *const, attr,
                    const size_t, size, const __u32, flags)
            {
                return flags ? -EINVAL : 0;
            }
            ''',
        )
        resolved = get_targets.resolve_function_for_location(self.kernel_dir, 'kernel/sys_demo.c:4')
        self.assertEqual(
            resolved,
            f'__do_sys_landlock_create_ruleset@{os.path.join(self.kernel_dir, "kernel/sys_demo.c")}:4',
        )

    def test_non_function_location_raises(self):
        self.write_source(
            'kernel/not_func.c',
            r'''
            static int f(void)
            {
                return 0;
            }

            int global_value = 1;
            ''',
        )
        with self.assertRaisesRegex(ValueError, 'cannot resolve enclosing function'):
            get_targets.resolve_function_for_location(self.kernel_dir, 'kernel/not_func.c:6')

    def test_file_level_cache_reuses_same_list_object(self):
        path = self.write_source(
            'kernel/cache_demo.c',
            r'''
            static int cache_demo(void)
            {
                return 0;
            }
            ''',
        )
        first = get_targets.get_file_function_ranges(path)
        second = get_targets.get_file_function_ranges(path)
        self.assertIs(first, second)

    def test_fast_build_rebuilds_incomplete_existing_artifact(self):
        outdir = Path(self.kernel_dir) / get_targets.DEF_SETS['outdir']
        outdir.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(get_targets.ensure_cache_dir(self.kernel_dir))
        (Path(self.kernel_dir) / 'vmlinux').write_bytes(b'fake-vmlinux')

        demo_path = self.write_source(
            'kernel/demo.c',
            r'''
            static int demo_fn(void)
            {
                return 0;
            }
            ''',
        )
        target = 'demo_fn@kernel/demo.c:2'
        targets_hash = get_targets.gen_targets_hash(self.kernel_dir, [target])
        allias_path = outdir / f"{get_targets.DEF_SETS['allias']}_fast_{targets_hash}"
        instru_path = outdir / f"{get_targets.DEF_SETS['instru']}_fast_{targets_hash}"
        pcs2funcs_path = outdir / f"{get_targets.DEF_SETS['pcs2funcs']}_fast_{targets_hash}"
        allias_path.write_text('stale-allias')
        with instru_path.open('w') as f:
            json.dump({'other_fn': {}}, f)
        with pcs2funcs_path.open('w') as f:
            json.dump({}, f)

        expected_info = {
            'demo_fn': {
                f'{demo_path}:2': ['0xffffffff81000005'],
            }
        }
        func_key = 'demo_fn@kernel/demo.c'
        cache_path = Path(get_targets.get_cached_func_allias_path(str(cache_dir), func_key))
        cache_path.write_text(f'0xffffffff81000000\ndemo_fn\n{demo_path}:2\n')

        def fake_build_instru_info(outdir_arg, allias_path_arg=None, instru_path_arg=None):
            with open(instru_path_arg, 'w') as f:
                json.dump(expected_info, f)
            return expected_info

        with mock.patch.object(get_targets, 'build_instru_info', side_effect=fake_build_instru_info), mock.patch.object(
            get_targets,
            'build_pcs2funcs',
            return_value={},
        ):
            info = get_targets.action_fast_build(self.kernel_dir, [target], rebuild=False)

        self.assertEqual(info, expected_info)
        self.assertTrue(cache_path.is_file())

    def test_fast_build_reads_only_requested_canonical_cache(self):
        outdir = Path(self.kernel_dir) / get_targets.DEF_SETS['outdir']
        outdir.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(get_targets.ensure_cache_dir(self.kernel_dir))
        (Path(self.kernel_dir) / 'vmlinux').write_bytes(b'fake-vmlinux')

        target_a_path = self.write_source(
            'kernel/a.c',
            r'''
            static int target_a(void)
            {
                return 0;
            }
            ''',
        )
        target_b_path = self.write_source(
            'kernel/b.c',
            r'''
            static int target_b(void)
            {
                return 0;
            }
            ''',
        )
        target = 'target_a@kernel/a.c:2'
        target_hash = get_targets.gen_targets_hash(self.kernel_dir, [target])
        allias_path = outdir / f"{get_targets.DEF_SETS['allias']}_fast_{target_hash}"
        instru_path = outdir / f"{get_targets.DEF_SETS['instru']}_fast_{target_hash}"
        pcs2funcs_path = outdir / f"{get_targets.DEF_SETS['pcs2funcs']}_fast_{target_hash}"

        cache_a = Path(get_targets.get_cached_func_allias_path(str(cache_dir), 'target_a@kernel/a.c'))
        cache_b = Path(get_targets.get_cached_func_allias_path(str(cache_dir), 'target_b@kernel/b.c'))
        cache_a.write_text(f'0xffffffff81000000\ntarget_a\n{target_a_path}:2\n')
        cache_b.write_text(f'0xffffffff82000000\ntarget_b\n{target_b_path}:2\n')

        captured = {}

        def fake_build_instru_info(outdir_arg, allias_path_arg=None, instru_path_arg=None):
            captured['allias_text'] = Path(allias_path_arg).read_text()
            info = {'target_a': {f'{target_a_path}:2': ['0xffffffff81000005']}}
            with open(instru_path_arg, 'w') as f:
                json.dump(info, f)
            return info

        def fake_build_pcs2funcs(info_arg, outdir_arg, pcs2funcs_path_arg):
            with open(pcs2funcs_path_arg, 'w') as f:
                json.dump({}, f)
            return {}

        with mock.patch.object(get_targets, 'build_instru_info', side_effect=fake_build_instru_info), mock.patch.object(
            get_targets,
            'build_pcs2funcs',
            side_effect=fake_build_pcs2funcs,
        ):
            info = get_targets.action_fast_build(self.kernel_dir, [target], rebuild=True)

        self.assertEqual(info['target_a'][f'{target_a_path}:2'][0], '0xffffffff81000005')
        self.assertIn('target_a', captured['allias_text'])
        self.assertNotIn('target_b', captured['allias_text'])
        self.assertTrue(allias_path.is_file())
        self.assertTrue(instru_path.is_file())
        self.assertTrue(pcs2funcs_path.is_file())

    def test_fast_cache_changes_when_vmlinux_content_changes(self):
        cache_dir = Path(get_targets.ensure_cache_dir(self.kernel_dir))
        vmlinux = Path(self.kernel_dir) / 'vmlinux'
        vmlinux.write_bytes(b'first-kernel')
        original_mtime_ns = vmlinux.stat().st_mtime_ns
        first_hash = get_targets.gen_targets_hash(
            self.kernel_dir, ['target@kernel/demo.c:1']
        )
        first_path = get_targets.get_cached_func_allias_path(
            str(cache_dir), 'target@kernel/demo.c'
        )

        vmlinux.write_bytes(b'other-kernel')
        os.utime(vmlinux, ns=(original_mtime_ns, original_mtime_ns))
        second_hash = get_targets.gen_targets_hash(
            self.kernel_dir, ['target@kernel/demo.c:1']
        )
        second_path = get_targets.get_cached_func_allias_path(
            str(cache_dir), 'target@kernel/demo.c'
        )

        self.assertNotEqual(first_hash, second_hash)
        self.assertNotEqual(first_path, second_path)

    def test_fast_build_falls_back_to_full_build_for_missing_inline_target(self):
        outdir = Path(self.kernel_dir) / get_targets.DEF_SETS['outdir']
        outdir.mkdir(parents=True, exist_ok=True)
        (Path(self.kernel_dir) / 'vmlinux').write_bytes(b'fake-vmlinux')
        self.write_source(
            'kernel/inline_demo.c',
            r'''
            static inline int inline_only(void)
            {
                return 0;
            }
            ''',
        )
        target = 'kernel/inline_demo.c:2'

        full_build_info = {
            'inline_only': {
                f'{os.path.join(self.kernel_dir, "kernel/inline_demo.c")}:2': ['0xffffffff8100abcd'],
            }
        }

        def fake_build_instru_info(outdir_arg, allias_path_arg=None, instru_path_arg=None):
            with open(instru_path_arg, 'w') as f:
                json.dump({}, f)
            return {}

        def fake_build_pcs2funcs(info_arg, outdir_arg, pcs2funcs_path_arg):
            with open(pcs2funcs_path_arg, 'w') as f:
                json.dump({}, f)
            return {}

        with mock.patch.object(get_targets, 'build_instru_info', side_effect=fake_build_instru_info), mock.patch.object(
            get_targets,
            'build_pcs2funcs',
            side_effect=fake_build_pcs2funcs,
        ), mock.patch.object(
            get_targets,
            'get_funcs_start_stop_addrs',
            return_value={'inline_only@kernel/inline_demo.c': {'start_addr': [], 'stop_addr': []}},
        ), mock.patch.object(
            get_targets,
            'action_build',
            return_value=full_build_info,
        ) as mocked_action_build:
            info = get_targets.action_fast_build(self.kernel_dir, [target], rebuild=True)

        self.assertEqual(info, full_build_info)
        mocked_action_build.assert_called_once_with(self.kernel_dir, True)

    def test_fast_build_can_return_partial_info_without_full_fallback(self):
        outdir = Path(self.kernel_dir) / get_targets.DEF_SETS['outdir']
        outdir.mkdir(parents=True, exist_ok=True)
        (Path(self.kernel_dir) / 'vmlinux').write_bytes(b'fake-vmlinux')
        self.write_source(
            'kernel/inline_demo.c',
            r'''
            static inline int inline_only(void)
            {
                return 0;
            }
            ''',
        )
        target = 'kernel/inline_demo.c:2'

        def fake_build_instru_info(outdir_arg, allias_path_arg=None, instru_path_arg=None):
            with open(instru_path_arg, 'w') as f:
                json.dump({}, f)
            return {}

        def fake_build_pcs2funcs(info_arg, outdir_arg, pcs2funcs_path_arg):
            with open(pcs2funcs_path_arg, 'w') as f:
                json.dump({}, f)
            return {}

        with mock.patch.object(
            get_targets, 'build_instru_info', side_effect=fake_build_instru_info
        ), mock.patch.object(
            get_targets, 'build_pcs2funcs', side_effect=fake_build_pcs2funcs
        ), mock.patch.object(
            get_targets,
            'get_funcs_start_stop_addrs',
            return_value={
                'inline_only@kernel/inline_demo.c': {
                    'start_addr': [],
                    'stop_addr': [],
                }
            },
        ), mock.patch.object(get_targets, 'action_build') as mocked_action_build:
            info = get_targets.action_fast_build(
                self.kernel_dir, [target], rebuild=True, allow_partial=True
            )

        self.assertEqual(info, {})
        mocked_action_build.assert_not_called()

    def test_missing_inline_func_uses_same_source_visible_symbol_ranges(self):
        self.write_source(
            'kernel/inline_owner.c',
            r'''
            static inline int inline_only(int x)
            {
                return x + 1;
            }

            int visible_owner(int x)
            {
                return inline_only(x);
            }

            int other_file_func(void)
            {
                return 0;
            }
            ''',
        )
        d_funcs_addrs = {
            'inline_only@kernel/inline_owner.c': {
                'start_addr': [],
                'stop_addr': [],
            }
        }
        ordered_records = [
            ('0x1000', 'visible_owner'),
            ('0x1100', 'other_file_func'),
            ('0x1200', 'unrelated_next'),
        ]

        get_targets.fill_missing_func_addrs_from_same_source_file(
            self.kernel_dir, ordered_records, d_funcs_addrs
        )

        self.assertEqual(d_funcs_addrs['inline_only@kernel/inline_owner.c']['start_addr'], ['0x1000', '0x1100'])
        self.assertEqual(d_funcs_addrs['inline_only@kernel/inline_owner.c']['stop_addr'], ['0x1100', '0x1200'])

    def test_addr2line_output_must_contain_target_source(self):
        target_output = '\n'.join([
            '0xffffffff81000000',
            'inline_only',
            f'{self.kernel_dir}/kernel/inline_owner.c:2',
            'visible_owner',
            f'{self.kernel_dir}/kernel/inline_owner.c:7',
        ])
        unrelated_output = '\n'.join([
            '0xffffffff82000000',
            'same_name',
            f'{self.kernel_dir}/drivers/other.c:9',
        ])

        self.assertTrue(
            get_targets.addr2line_output_contains_target_source(
                self.kernel_dir, 'kernel/inline_owner.c', target_output
            )
        )
        self.assertFalse(
            get_targets.addr2line_output_contains_target_source(
                self.kernel_dir, 'kernel/inline_owner.c', unrelated_output
            )
        )
        self.assertTrue(
            get_targets.addr2line_output_contains_target_source(
                self.kernel_dir, 'kernel/inline_owner.c', 'inline_only\nkernel/inline_owner.c:2'
            )
        )

    def test_fast_build_filters_worker_segments_by_debug_source(self):
        outdir = Path(self.kernel_dir) / get_targets.DEF_SETS['outdir']
        outdir.mkdir(parents=True, exist_ok=True)
        cache_dir = Path(get_targets.ensure_cache_dir(self.kernel_dir))
        (Path(self.kernel_dir) / 'vmlinux').write_bytes(b'fake-vmlinux')
        target_path = self.write_source(
            'kernel/inline_owner.c',
            r'''
            static inline int inline_only(void)
            {
                return 0;
            }
            ''',
        )
        self.write_source(
            'drivers/other.c',
            r'''
            static int same_name(void)
            {
                return 0;
            }
            ''',
        )
        func_key = 'inline_only@kernel/inline_owner.c'
        target_segment = f'0xffffffff81000000\ninline_only\n{target_path}:2\n'
        unrelated_segment = f'0xffffffff82000000\nsame_name\n{self.kernel_dir}/drivers/other.c:2\n'

        class FakeFuture:
            def __init__(self, result):
                self._result = result

            def result(self):
                return self._result

        class FakeExecutor:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def submit(self, fn, task):
                task_func_key, start_addr, _, _ = task
                segment = target_segment if start_addr == '0x1000' else unrelated_segment
                return FakeFuture((task_func_key, segment))

        with mock.patch.object(
            get_targets,
            'get_funcs_start_stop_addrs',
            return_value={func_key: {'start_addr': ['0x1000', '0x2000'], 'stop_addr': ['0x1100', '0x2100']}},
        ), mock.patch.object(
            get_targets.concurrent.futures,
            'ProcessPoolExecutor',
            FakeExecutor,
        ), mock.patch.object(
            get_targets.concurrent.futures,
            'as_completed',
            side_effect=lambda futures: list(futures),
        ):
            info = get_targets.action_fast_build(self.kernel_dir, ['inline_only@kernel/inline_owner.c:2'], rebuild=True)

        cache_path = Path(get_targets.get_cached_func_allias_path(str(cache_dir), func_key))
        cache_text = cache_path.read_text()
        self.assertIn('kernel/inline_owner.c:2', cache_text)
        self.assertNotIn('drivers/other.c:2', cache_text)
        self.assertIn('inline_only', info)
        self.assertIn(f'{target_path}:2', info['inline_only'])


if __name__ == '__main__':
    unittest.main()
