"""Static callgraph parsing and exact symbol matching regression tests."""

import io
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from brain.static_analyzer import StaticAnalyzer


class StaticAnalyzerTest(unittest.TestCase):
    def setUp(self):
        StaticAnalyzer.clear_graph_cache()

    def tearDown(self):
        StaticAnalyzer.clear_graph_cache()

    def test_kallgraph_csv_uses_named_caller_and_callee_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path = Path(temp_dir) / "callgraph.csv"
            graph_path.write_text(
                "caller_name,callsite_file,callsite_line,callee_name,"
                "callee_file,callee_line,call_type,is_declaration\n"
                "__sys_sendto,net/socket.c,123,helper,net/socket.c,456,direct,false\n"
                "helper,net/core/dev.c,789,target,net/core/dev.c,900,direct,false\n",
                encoding="utf-8",
            )
            analyzer = StaticAnalyzer(temp_dir, "target")
            self.assertTrue(analyzer.load_callgraph())
            self.assertEqual(analyzer.graph["helper"], {"__sys_sendto"})
            self.assertEqual(analyzer.graph["target"], {"helper"})
            self.assertNotIn("net/socket.c", analyzer.name_to_vid)
            entries = analyzer.find_reachable_syscall_entries()
            self.assertTrue(any(entry["name"] == "sendto$inet" for entry in entries))
            self.assertTrue(all(
                entry["guidance_level"] == "syz_call" for entry in entries
            ))

    def test_matching_build_reuses_immutable_process_graph(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path = Path(temp_dir) / "complete_callgraph"
            graph_path.write_text(
                "__sys_read -> helper\nhelper -> target\n",
                encoding="utf-8",
            )
            first = StaticAnalyzer(temp_dir, "target")
            second = StaticAnalyzer(temp_dir, "helper")

            self.assertTrue(first.load_callgraph())
            self.assertFalse(first.graph_cache_hit)
            self.assertTrue(second.load_callgraph())
            self.assertTrue(second.graph_cache_hit)
            self.assertIs(second.graph, first.graph)
            self.assertIsInstance(first.graph["target"], frozenset)
            with self.assertRaises(TypeError):
                first.graph["target"] = frozenset()
            with self.assertRaises(TypeError):
                first.name_to_vid["injected"] = 99

    def test_graph_changed_during_read_is_not_published(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path = Path(temp_dir) / "complete_callgraph"
            graph_path.write_text(
                "__sys_read -> target\n", encoding="utf-8"
            )
            analyzer = StaticAnalyzer(temp_dir, "target")
            bounded_lines = analyzer._bounded_text_lines

            def mutate_then_read(stream):
                graph_path.write_text(
                    "__sys_write -> other_target\n", encoding="utf-8"
                )
                return bounded_lines(stream)

            with mock.patch.object(
                    analyzer, "_bounded_text_lines", side_effect=mutate_then_read):
                self.assertFalse(analyzer.load_callgraph())

            self.assertEqual(analyzer.graph, {})
            self.assertFalse(analyzer.graph_cache_hit)

    def test_concurrent_same_graph_load_is_single_flight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path = Path(temp_dir) / "complete_callgraph"
            graph_path.write_text(
                "__sys_read -> helper\nhelper -> target\n",
                encoding="utf-8",
            )
            first = StaticAnalyzer(temp_dir, "target")
            second = StaticAnalyzer(temp_dir, "helper")
            parse_started = threading.Event()
            release_parse = threading.Event()
            parse_count = 0
            parse_count_lock = threading.Lock()
            original_lines = first._bounded_text_lines

            def slow_lines(stream):
                nonlocal parse_count
                with parse_count_lock:
                    parse_count += 1
                parse_started.set()
                release_parse.wait(timeout=2)
                return original_lines(stream)

            results = []
            with (
                mock.patch.object(first, "_bounded_text_lines", slow_lines),
                mock.patch.object(second, "_bounded_text_lines", slow_lines),
            ):
                owner = threading.Thread(
                    target=lambda: results.append(first.load_callgraph())
                )
                follower = threading.Thread(
                    target=lambda: results.append(second.load_callgraph())
                )
                owner.start()
                self.assertTrue(parse_started.wait(timeout=2))
                follower.start()
                time.sleep(0.05)
                release_parse.set()
                owner.join(timeout=2)
                follower.join(timeout=2)

            self.assertEqual(results, [True, True])
            self.assertEqual(parse_count, 1)
            self.assertTrue(second.graph_cache_hit)
            self.assertIs(first.graph, second.graph)

    def test_failed_single_flight_load_can_be_retried(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path = Path(temp_dir) / "complete_callgraph"
            graph_path.write_text(
                "__sys_read -> target\n", encoding="utf-8"
            )
            first = StaticAnalyzer(temp_dir, "target")
            second = StaticAnalyzer(temp_dir, "target")
            parse_started = threading.Event()
            release_parse = threading.Event()

            def fail_lines(_stream):
                parse_started.set()
                release_parse.wait(timeout=2)
                raise ValueError("injected parse failure")

            results = []
            with mock.patch.object(first, "_bounded_text_lines", fail_lines):
                owner = threading.Thread(
                    target=lambda: results.append(first.load_callgraph())
                )
                follower = threading.Thread(
                    target=lambda: results.append(second.load_callgraph())
                )
                owner.start()
                self.assertTrue(parse_started.wait(timeout=2))
                follower.start()
                time.sleep(0.05)
                release_parse.set()
                owner.join(timeout=2)
                follower.join(timeout=2)

            self.assertEqual(results, [False, False])
            self.assertFalse(first.graph_cache_hit)
            self.assertFalse(second.graph_cache_hit)
            self.assertTrue(second.load_callgraph())
            self.assertFalse(second.graph_cache_hit)

    def test_clear_during_load_does_not_cancel_new_single_flight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path = Path(temp_dir) / "complete_callgraph"
            graph_path.write_text(
                "__sys_read -> target\n", encoding="utf-8"
            )
            old_owner = StaticAnalyzer(temp_dir, "target")
            new_owner = StaticAnalyzer(temp_dir, "target")
            new_follower = StaticAnalyzer(temp_dir, "target")
            old_started = threading.Event()
            release_old = threading.Event()
            new_started = threading.Event()
            release_new = threading.Event()
            old_lines = old_owner._bounded_text_lines
            new_lines = new_owner._bounded_text_lines

            def paused_old_lines(stream):
                old_started.set()
                release_old.wait(timeout=2)
                return old_lines(stream)

            def paused_new_lines(stream):
                new_started.set()
                release_new.wait(timeout=2)
                return new_lines(stream)

            results = {}
            with (
                mock.patch.object(
                    old_owner, "_bounded_text_lines", paused_old_lines
                ),
                mock.patch.object(
                    new_owner, "_bounded_text_lines", paused_new_lines
                ),
            ):
                old_thread = threading.Thread(
                    target=lambda: results.setdefault(
                        "old", old_owner.load_callgraph()
                    )
                )
                new_thread = threading.Thread(
                    target=lambda: results.setdefault(
                        "new", new_owner.load_callgraph()
                    )
                )
                follower_thread = threading.Thread(
                    target=lambda: results.setdefault(
                        "follower", new_follower.load_callgraph()
                    )
                )

                old_thread.start()
                self.assertTrue(old_started.wait(timeout=2))
                StaticAnalyzer.clear_graph_cache()
                new_thread.start()
                self.assertTrue(new_started.wait(timeout=2))
                follower_thread.start()
                time.sleep(0.05)

                release_old.set()
                old_thread.join(timeout=2)
                self.assertFalse(old_thread.is_alive())
                self.assertTrue(follower_thread.is_alive())

                release_new.set()
                new_thread.join(timeout=2)
                follower_thread.join(timeout=2)

            self.assertEqual(results["old"], False)
            self.assertEqual(results["new"], True)
            self.assertEqual(results["follower"], True)
            self.assertTrue(new_follower.graph_cache_hit)
            self.assertIs(new_owner.graph, new_follower.graph)

    def test_graph_cache_has_bounded_lru_eviction(self):
        original_capacity = StaticAnalyzer._graph_cache_capacity
        StaticAnalyzer._graph_cache_capacity = 2
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                analyzers = []
                for index in range(3):
                    graph_dir = Path(temp_dir) / str(index)
                    graph_dir.mkdir()
                    (graph_dir / "complete_callgraph").write_text(
                        f"__sys_read -> target_{index}\n", encoding="utf-8"
                    )
                    analyzer = StaticAnalyzer(str(graph_dir), f"target_{index}")
                    self.assertTrue(analyzer.load_callgraph())
                    analyzers.append(analyzer)

                self.assertEqual(len(StaticAnalyzer._graph_cache), 2)
                reloaded = StaticAnalyzer(
                    str(Path(temp_dir) / "0"), "target_0"
                )
                self.assertTrue(reloaded.load_callgraph())
                self.assertFalse(reloaded.graph_cache_hit)
        finally:
            StaticAnalyzer._graph_cache_capacity = original_capacity

    def test_text_callgraph_populates_names_and_maps_wrapper_alias(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path = Path(temp_dir) / "complete_callgraph"
            graph_path.write_text(
                "__do_sys_sendto -> helper\n"
                "helper -> target.constprop.3\n",
                encoding="utf-8",
            )
            analyzer = StaticAnalyzer(temp_dir, "target")
            self.assertTrue(analyzer.load_callgraph())
            self.assertIn("target.constprop.3", analyzer.name_to_vid)
            entries = analyzer.find_reachable_syscall_entries()
            self.assertTrue(any(entry["name"] == "sendto$inet" for entry in entries))

    def test_target_and_syscall_mapping_do_not_use_substrings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "complete_callgraph").write_text(
                "__sys_readiness -> foobar\n",
                encoding="utf-8",
            )
            analyzer = StaticAnalyzer(temp_dir, "foo")
            self.assertTrue(analyzer.load_callgraph())
            self.assertEqual(analyzer.find_reachable_syscall_entries(), [])
            self.assertEqual(analyzer._map_kernel_to_syzkaller("__sys_readiness"), [])

    def test_callgraph_size_limit_is_enforced(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "complete_callgraph").write_text(
                "__sys_read -> target\n", encoding="utf-8"
            )
            analyzer = StaticAnalyzer(
                temp_dir, "target", max_callgraph_bytes=1
            )
            self.assertFalse(analyzer.load_callgraph())

    def test_callgraph_directory_scan_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "callgraph.csv").write_text(
                "caller_name,callee_name\n__sys_read,target\n",
                encoding="utf-8",
            )
            (root / "extra").write_text("x", encoding="utf-8")
            analyzer = StaticAnalyzer(temp_dir, "target", max_scan_entries=1)
            self.assertIsNone(analyzer.callgraph_path)

    def test_callgraph_replacement_with_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph_root = root / "graphs"
            graph_root.mkdir()
            outside = root / "outside"
            outside.mkdir()
            graph_path = graph_root / "complete_callgraph"
            graph_path.write_text("__sys_read -> target\n", encoding="utf-8")
            outside_graph = outside / "complete_callgraph"
            outside_graph.write_text("__sys_write -> target\n", encoding="utf-8")
            analyzer = StaticAnalyzer(
                str(graph_root), "target", trusted_roots=(str(graph_root),)
            )

            graph_path.unlink()
            graph_path.symlink_to(outside_graph)
            self.assertFalse(analyzer.load_callgraph())
            self.assertEqual(analyzer.graph, {})

    def test_callgraph_stream_read_enforces_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph_path = Path(temp_dir) / "complete_callgraph"
            graph_path.write_text("__sys_read -> target\n", encoding="utf-8")
            analyzer = StaticAnalyzer(
                temp_dir, "target", max_callgraph_bytes=8
            )
            with self.assertRaisesRegex(ValueError, "grew beyond"):
                list(analyzer._bounded_text_lines(io.BytesIO(b"123456789")))


if __name__ == "__main__":
    unittest.main()
