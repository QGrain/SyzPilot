import os
import re
import json
import logging
import argparse
import numpy as np
from math import log2
from typing import Iterator
from get_targets import action_get, action_fast_build


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


STACK_FRAME_PATTERN = re.compile(
    r'^(?P<func>[\w\.]+)'
    r'(?:\+(?P<offset>0x[\da-fA-F]+(?:/0x[\da-fA-F]+)?))?'
    r'.*?\s+'
    r'(?P<loc>[^ \t\n]+:\d+)'
    r'(?:\s+(?P<inline>\[inline\]))?'
)

KASAN_TRACE_BOUNDARY_PREFIXES = (
    'Allocated by task',
    'Freed by task',
    'Last potentially related work creation:',
    'Second to last potentially related work creation:',
    'The buggy address',
    'Memory state around the buggy address:',
    'page last allocated',
    'page last free',
    'page:',
    'head:',
    'flags:',
    'raw:',
    'page dumped because:',
)


def match_stack_frame(line: str):
    """Return a parsed kernel stack frame, excluding report metadata pseudo-frames."""
    match = STACK_FRAME_PATTERN.search(line.strip())
    if match is None:
        return None
    source_path = match.group('loc').rsplit(':', 1)[0]
    if not re.search(r'\.(?:c|h|S|s|rs)$', source_path):
        return None
    return match


def deduplicate_target_pcs(
    full_targets: list[str], target_pcs: list[str]
) -> tuple[list[str], list[str]]:
    """Drop zero and duplicate fuzzer PCs while preserving the deepest target."""
    if len(full_targets) != len(target_pcs):
        raise ValueError("resolved targets and PCs have different lengths")
    retained_reversed = []
    seen_pc32 = set()
    for target, pc in reversed(list(zip(full_targets, target_pcs))):
        try:
            pc32 = int(pc, 16) & 0xffffffff
        except (TypeError, ValueError):
            pc32 = 0
        if pc32 == 0:
            logger.warning("Dropping waypoint with an invalid zero PC: %s", target)
            continue
        if pc32 in seen_pc32:
            logger.warning(
                "Dropping shallower waypoint %s with duplicate PC32 0x%08x",
                target,
                pc32,
            )
            continue
        seen_pc32.add(pc32)
        retained_reversed.append((target, pc))
    if not retained_reversed:
        raise ValueError("no nonzero unique target PCs remain")
    retained = list(reversed(retained_reversed))
    return [target for target, _ in retained], [pc for _, pc in retained]


def read_file(file_path: str) -> str:
    with open(file_path, 'r') as f:
        return f.read().strip()


def compute_robust_z_score(data):
    """Compute Robust Z-score (Modified Z-score)"""
    data = np.array(data, dtype=float)
    # 1. Compute median
    median = np.median(data)
    # 2. Compute MAD (Median Absolute Deviation)
    mad = np.median(np.abs(data - median)) + 1e-9
    # 3. Compute Robust Z-score
    # Coefficient 0.6745 is used to approximate the standard Z-score under the normal distribution
    robust_z = 0.6745 * (data - median) / mad

    return robust_z


class WaypointNode:
    """
    Represents a single frame in the stack trace.
    """
    def __init__(self, func_name: str, location: str, instruct_offset: str, is_inline: bool, original_index: int):
        self.func_name = func_name
        self.location = os.path.normpath(location)      # e.g., "sound/core/oss/pcm_oss.c:1121"
        self.instruct_offset = instruct_offset          # e.g., "0x164/0x1c0" or None
        self.is_inline = is_inline    # bool
        self.original_index = original_index
        self.bb_offset = None
        self.bb_count = None
        self.hot_entry_score = None
        self.value_score = None

        # Linked list pointers
        self.next = None
        self.prev = None

    def get_bb_offset_and_count(self, instrumentation_info: dict) -> bool:
        """
        Get the basic block offset and count for the function.
        type of instrumentation_info: dict[str, dict[str, list[str]]]
        return boolean success or failure
        """
        func_instru_info = instrumentation_info.get(self.func_name, {})
        if func_instru_info == {}:
            return False

        src_path, src_line = self.location.rsplit(':', 1)
        matched_instru_loc = None
        instru_sites = []
        for loc in func_instru_info.keys():
            # there should be only one unique function name in one source file
            # collect the instrumentation sites
            loc = os.path.normpath(loc)
            if src_path in loc:
                instru_sites.append(loc)
        assert len(instru_sites) == len(set(instru_sites))
        sorted_instru_sites = sorted(instru_sites, key=lambda s: int(s.rsplit(':', 1)[1]))
        # find the matched or previous closest location
        for instru_loc in sorted_instru_sites:
            _, instru_line = instru_loc.split(':')
            if int(instru_line) <= int(src_line):
                matched_instru_loc = instru_loc
            else:
                break
        if matched_instru_loc is None:
            return False
        # bb_offset starts from 0 to bb_count-1
        self.bb_offset = sorted_instru_sites.index(matched_instru_loc)
        self.bb_count = len(sorted_instru_sites)
        return True

    def calc_hot_entry_score(self) -> float:
        """
        Calculate the hot entry score for the function. The hot entry score ∈ [0, 1].
        The higher the score, the more likely the callsite bb is a hot entry bb.
        """
        hot_entry_score = 0
        if self.bb_offset is not None and self.bb_count is not None:
            hot_entry_score = 1 - (self.bb_offset / self.bb_count)
            self.hot_entry_score = hot_entry_score
        else:
            assert self.instruct_offset is not None, f"instruction offset and basic block count are not available: {repr(self)}"
            instruct_position, func_size = self.instruct_offset.split('/')
            instruct_position, func_size = int(instruct_position, 16), int(func_size, 16)
            hot_entry_score = 1 - (instruct_position / func_size)
            self.hot_entry_score = hot_entry_score
        return hot_entry_score

    def calc_value_score(self, same_src_file_cnt: int, alpha: float = 0.2, beta: float = 0.8, gamma: float = 1.0) -> float:
        """Calculate the value score for the function."""
        assert (
            self.bb_count is not None and
            same_src_file_cnt >= 1 and
            self.hot_entry_score is not None
        ), f"bb_count or same_src_file_cnt is not available: {repr(self)}"
        # value_score = log2(1 + self.bb_count)
        value_score = log2(2 + self.bb_offset)
        value_score -= alpha * log2(same_src_file_cnt)
        value_score -= beta * self.hot_entry_score
        value_score -= gamma * int(self.is_inline)
        self.value_score = value_score
        return value_score

    def __repr__(self) -> str:
        repr = f"{self.func_name}"
        if self.instruct_offset:
            repr += f"+{self.instruct_offset}"
        repr += f" {self.location}"
        if self.is_inline:
            repr += " [inline]"
        repr += f" (idx={self.original_index})"
        return repr


class Waypoints:
    """
    A linked-list container for WaypointNodes representing a kernel calltrace.
    """
    def __init__(self, raw_calltrace_str: str = None, bug_func: str = None):
        self.bug_func = bug_func
        self.head = None
        self.tail = None
        self.length = 0
        self._node_map = {} # Optional: quick lookup helper
        self.src_file_distribution = {}

        if raw_calltrace_str:
            self._parse_and_build(raw_calltrace_str)

    def _parse_and_build(self, raw_text: str):
        """
        Parses the raw calltrace string and builds the linked list.
        The input text usually has the leaf (callee) at the top, but we want
        the head to be the root (caller).
        """
        lines = raw_text.strip().split('\n')
        # Reverse lines so the entry point (syscall) becomes index 0/Head
        lines.reverse()

        for idx, line in enumerate(lines):
            line = line.strip()
            if not line: continue

            match = match_stack_frame(line)
            if match:
                data = match.groupdict()
                node = WaypointNode(
                    func_name=data['func'],
                    location=data['loc'],
                    instruct_offset=data['offset'],
                    is_inline=bool(data['inline']),
                    original_index=idx
                )
                self.append_node(node)

    def append_node(self, node: WaypointNode):
        """Adds a node to the end of the chain."""
        if self.head is None:
            self.head = node
            self.tail = node
            self.length += 1
        else:
            self.tail.next = node
            node.prev = self.tail
            self.tail = node
            self.length += 1
        # Keep a reference if needed for direct access, optional
        self._node_map[node.original_index] = node
        src_file = node.location.split(':')[0]
        self.src_file_distribution.setdefault(src_file, 0)
        self.src_file_distribution[src_file] += 1

    def delete_node(self, node: WaypointNode):
        """Removes a specific node from the chain."""
        if node is None:
            return

        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next # Node was head

        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev # Node was tail

        # Clear references to detach completely
        node.prev = None
        node.next = None
        self.length -= 1
        src_file = node.location.split(':')[0]
        self.src_file_distribution[src_file] -= 1
        assert self.src_file_distribution[src_file] >= 0, f"src_file_distribution[{src_file}] is negative"

    def get_distance(self, node_a: WaypointNode, node_b: WaypointNode) -> int:
        """
        Returns the original function distance between two nodes.
        Returns None if one of the nodes is invalid.
        """
        if not node_a or not node_b:
            return None
        return abs(node_a.original_index - node_b.original_index)

    def calc_waypoints_value_scores(self, alpha: float = 0.2, beta: float = 0.8, gamma: float = 1.0):
        """Calculate the value scores for current waypoints."""
        current_waypoint = self.head
        while current_waypoint is not None:
            next_waypoint = current_waypoint.next
            src_file = current_waypoint.location.split(':')[0]
            current_waypoint.calc_value_score(
                self.src_file_distribution.get(src_file, 1),
                alpha,
                beta,
                gamma
            )
            # increase the value score of the bug function waypoint to make it undeletable
            if current_waypoint.func_name == self.bug_func:
                current_waypoint.value_score += 100
            current_waypoint = next_waypoint
        logger.info(f"calc_waypoints_value_scores done, waypoints length: {self.length}")

    def serialize(self) -> str:
        """
        Serializes the current chain to the requested string format.
        Format: func1@file1:line1, func2@file2:line2, ...
        """
        result = []
        current = self.head
        while current:
            result.append(f"{current.func_name}@{current.location}")
            current = current.next
        return ", ".join(result)

    def recover_calltrace(self, debug: bool = False) -> str:
        calltrace_lines = []
        current = self.head
        while current:
            line = f"{current.func_name}@{current.location}"
            if debug == True:
                if current.is_inline:
                    line += " [inline]"
                if current.bb_count is not None:
                    line += f" (bb_offset={current.bb_offset}/{current.bb_count})"
                current_src_file = current.location.split(':')[0]
                if current_src_file in self.src_file_distribution:
                    line += f"(file_cnt={self.src_file_distribution[current_src_file]})"
                if current.hot_entry_score is not None:
                    line += f"(hot_entry_score={current.hot_entry_score:.3f})"
                if current.value_score is not None:
                    line += f"(value_score={current.value_score:.3f})"
                line += f"(idx={current.original_index})"
            calltrace_lines.append(line)
            current = current.next
        calltrace_lines.reverse()
        return "\n".join(calltrace_lines)

    def __iter__(self) -> Iterator[WaypointNode]:
        """Allows standard Python iteration: for node in waypoints: ..."""
        current = self.head
        while current:
            yield current
            current = current.next


class WaypointsExtractor:
    MIN_BB_COUNT = 5
    MIN_WAYPOINTS_LENGTH = 4
    MAX_WAYPOINTS_LENGTH = 10
    HOT_ENTRY_SCORE_THRESHOLD = 0.9
    # experimental parameters
    VALUE_SCORE_ALPHA = 0.2
    VALUE_SCORE_BETA = 0.8
    VALUE_SCORE_GAMMA = 1.0

    def __init__(self, kernel_dir: str, title_path: str, report_path: str, debug: bool = False):
        self.kernel_dir = kernel_dir
        self.bug_title = read_file(title_path)
        # bug_title looks like BUG_TYPE in BUG_FUNC (TIMES)
        self.bug_title = self.bug_title.split('(')[0].strip()
        self.bug_func = self.bug_title.split(' in ')[-1].strip()
        self.bug_report = read_file(report_path)
        self.report_type = self.detect_report_type()
        self.call_trace = self.extract_calltrace()
        self.waypoints = Waypoints(self.call_trace, self.bug_func)
        self.debug = debug

    def detect_report_type(self) -> str:
        if "KASAN" in self.bug_title and "Freed by" in self.bug_report:
            return "KASAN"
        return "Normal"

    def extract_calltrace(self) -> str:
        report_lines = self.bug_report.split('\n')
        logger.info(f"len(report_lines): {len(report_lines)}")
        if self.report_type == "Normal":
            return self.extract_normal_calltrace(report_lines)
        elif self.report_type == "KASAN":
            return self.extract_kasan_calltrace(report_lines)
        else:
            raise ValueError(f"Unknown report type: {self.report_type}")

    def extract_kasan_calltrace(self, report_lines: list[str]) -> str:
        """Call Trace = Allocated Task Call Trace +Freed Task Call Trace + Bug Callsite"""
        # TODO: insert allocate tail trace
        logger.info("KASAN report type")
        call_trace_lines = []
        calltrace_start = False
        freedtrace_start = False
        for line in report_lines:
            line = line.strip()
            if line == '':
                continue
            if line.startswith('Call Trace:'):
                calltrace_start = True
                continue
            if calltrace_start:
                if line.startswith(KASAN_TRACE_BOUNDARY_PREFIXES):
                    calltrace_start = False
                if line.startswith('entry_SYSCALL') or line.startswith('</TASK>'):
                    calltrace_start = False
                    continue
                # only add the line of bug callsite
                # its idx in the waypoints may be far away from the previous node, but it doesn't matter
                if match_stack_frame(line) is not None and self.bug_func in line:
                    call_trace_lines.append(line)
                    calltrace_start = False
                    continue
            if line.startswith("Freed by task"):
                freedtrace_start = True
                continue
            if freedtrace_start:
                if line.startswith('entry_SYSCALL') or line.startswith('</TASK>'):
                    freedtrace_start = False
                    break
                if line.startswith(KASAN_TRACE_BOUNDARY_PREFIXES):
                    freedtrace_start = False
                    break
                if match_stack_frame(line) is not None and line not in call_trace_lines:
                    call_trace_lines.append(line)
        return '\n'.join(call_trace_lines)

    def extract_normal_calltrace(self, report_lines: list[str]) -> str:
        logger.info("Normal report type")
        call_trace = ""
        region_start = 0
        for line in report_lines:
            line = line.strip()
            if line == '':
                continue
            if 'RIP:' in line:
                try:
                    # maxsplit=3 preserves the ':' in file:line
                    parts = line.split(':', maxsplit=3)
                    call_trace_line = parts[2] + ':' + parts[3]
                    assert ':' in call_trace_line, "extraction error for RIP or no debug info in location"
                    if call_trace_line not in call_trace:
                        call_trace += call_trace_line + '\n'
                except:
                    continue
            if line.startswith('Call Trace:'):
                region_start = 1
                continue
            if region_start == 1:
                if line.startswith('entry_SYSCALL') or line.startswith('</TASK>'):
                    region_start = 0
                    break
                if line.startswith(KASAN_TRACE_BOUNDARY_PREFIXES):
                    region_start = 0
                    break
                if line not in call_trace:
                    call_trace += line + '\n'
        call_trace = call_trace.strip()
        return call_trace

    def load_bb_info(self, instrumentation_info: dict):
        current_waypoint = self.waypoints.head
        while current_waypoint is not None:
            next_waypoint = current_waypoint.next
            success = current_waypoint.get_bb_offset_and_count(instrumentation_info)
            if not success:
                self.waypoints.delete_node(current_waypoint)
                current_waypoint = next_waypoint
                continue
            current_waypoint.calc_hot_entry_score()
            current_waypoint = next_waypoint
        logger.info(f"load_bb_info with hot_entry_score calculation done, waypoints length: {self.waypoints.length}")

    def trace_sanitization(self):
        """Rule 1: Remove sanitizer functions, boilerplate functions, and functions in black list, and stop at the bug function"""
        black_list_sanitizers = ["kasan_", "__kasan_", "mm/kasan", "mm/kfence", "kmsan", "kcsan", "dump_stack"]
        black_list_boilerplates = ["entry_SYSCALL", "do_syscall", "__x64_sys_", "__x32_sys_", "__se_sys", "__do_sys"]
        black_list_functions = ["printf", "kmalloc", "kfree", "krealloc", "kzalloc", "kcalloc", "kzalloc", "kcalloc", "include/linux/slab.h", "mm/slub.c", "mm/slab.c", "kernel/exit.c", "kernel/entry"]

        current_waypoint = self.waypoints.head
        reached_bug_func = False
        while current_waypoint is not None:
            next_waypoint = current_waypoint.next
            if any(one in repr(current_waypoint) for one in black_list_sanitizers) or \
                any(one in repr(current_waypoint) for one in black_list_boilerplates) or \
                any(one in repr(current_waypoint) for one in black_list_functions) or \
                reached_bug_func:
                self.waypoints.delete_node(current_waypoint)
            if current_waypoint.func_name == self.bug_func:
                reached_bug_func = True
            current_waypoint = next_waypoint
        if self.debug:
            print(self.waypoints.recover_calltrace(debug=True))
        logger.info(f"trace_sanitize done, waypoints length: {self.waypoints.length}")

    def complexity_based_filtering(self):
        """Rule 2: Remove functions with high complexity"""
        current_waypoint = self.waypoints.head
        while current_waypoint is not None:
            next_waypoint = current_waypoint.next
            if current_waypoint.bb_count < self.MIN_BB_COUNT and current_waypoint.func_name != self.bug_func:
                self.waypoints.delete_node(current_waypoint)
            current_waypoint = next_waypoint
            if self.waypoints.length <= self.MIN_WAYPOINTS_LENGTH:
                break
        if self.debug:
            print(self.waypoints.recover_calltrace(debug=True))
        logger.info(f"complexity_based_filtering done, waypoints length: {self.waypoints.length}")

    def non_trivial_stage_separation(self):
        """Rule 3: Separate the non-trivial stages"""
        current_waypoint = self.waypoints.head
        while current_waypoint is not None and current_waypoint.next is not None:
            next_waypoint = current_waypoint.next
            # we only consider to trim the adjacent waypoints
            if self.waypoints.get_distance(current_waypoint, next_waypoint) == 1:
                assert (
                    next_waypoint.hot_entry_score is not None and
                    0 <= next_waypoint.hot_entry_score <= 1
                ), f"hot_entry_score {next_waypoint.hot_entry_score} is not calculated or invalid"
                if next_waypoint.hot_entry_score >= self.HOT_ENTRY_SCORE_THRESHOLD and next_waypoint.func_name != self.bug_func:
                    self.waypoints.delete_node(next_waypoint)
            current_waypoint = current_waypoint.next
            if self.waypoints.length <= self.MIN_WAYPOINTS_LENGTH:
                break
        # Wrapper-chain collapsing:
        # Deleting the next implementation function which is semantically similar to the current one
        # Such as: geneve_xmit_skb to geneve_xmit, ____sys_sendmsg to __sys_sendmsg
        continue_to_check_wrapper = True
        while continue_to_check_wrapper:
            current_waypoint = self.waypoints.head
            continue_to_check_wrapper = False
            while current_waypoint is not None and current_waypoint.next is not None:
                if self.waypoints.length <= self.MIN_WAYPOINTS_LENGTH:
                    continue_to_check_wrapper = False
                    break
                next_waypoint = current_waypoint.next
                if current_waypoint.func_name in next_waypoint.func_name:
                    self.waypoints.delete_node(next_waypoint)
                    continue_to_check_wrapper = True
                current_waypoint = next_waypoint

        if self.debug:
            print(self.waypoints.recover_calltrace(debug=True))
        logger.info(f"non_trivial_stage_separation done, waypoints length: {self.waypoints.length}")

    def recursively_greedy_triming(self):
        """Rule 4: Greedy triming"""
        if self.waypoints.length <= self.MAX_WAYPOINTS_LENGTH:
            logger.info(f"len(waypoints)={self.waypoints.length}, no need to trim")
            return
        self.waypoints.calc_waypoints_value_scores(
            self.VALUE_SCORE_ALPHA,
            self.VALUE_SCORE_BETA,
            self.VALUE_SCORE_GAMMA
        )
        waypoint_to_delete = None
        lowest_value_score = float('inf')
        current_waypoint = self.waypoints.head
        while current_waypoint is not None:
            assert current_waypoint.value_score is not None, f"value_score is not calculated for {current_waypoint.func_name}@{current_waypoint.location}"
            if current_waypoint.value_score < lowest_value_score:
                lowest_value_score = current_waypoint.value_score
                waypoint_to_delete = current_waypoint
            current_waypoint = current_waypoint.next
        if waypoint_to_delete is not None:
            self.waypoints.delete_node(waypoint_to_delete)
            if self.debug:
                print(self.waypoints.recover_calltrace(debug=True))
            logger.info(f"one iteration of recursively_greedy_triming done, waypoints length: {self.waypoints.length}")
            self.recursively_greedy_triming()
        else:
            logger.info(f"cannot find the waypoint to delete, current len(waypoints)={self.waypoints.length}")
            return

    def remove_outliers(self):
        outlier_waypoints = []
        value_scores = []
        current_waypoint = self.waypoints.head
        while current_waypoint is not None:
            next_waypoint = current_waypoint.next
            value_scores.append(current_waypoint.value_score)
            current_waypoint = next_waypoint

        z_score = compute_robust_z_score(value_scores)
        lower_outlier_masks = z_score < -1.0
        for i, waypoint in enumerate(self.waypoints):
            if lower_outlier_masks[i]:
                outlier_waypoints.append([waypoint, z_score[i]])

        sorted_outlier_waypoints = sorted(outlier_waypoints, key=lambda x: x[1])
        for waypoint, _ in sorted_outlier_waypoints:
            self.waypoints.delete_node(waypoint)
            if self.waypoints.length <= self.MIN_WAYPOINTS_LENGTH:
                break
        if self.debug:
            print(self.waypoints.recover_calltrace(debug=True))
        logger.info(f"remove_outliers done, waypoints length: {self.waypoints.length}")

    def extract(self):
        self.trace_sanitization()
        instru_info = action_fast_build(
            self.kernel_dir,
            self.waypoints.recover_calltrace().split('\n'),
            False,
            allow_partial=True,
        )
        self.load_bb_info(instru_info)
        self.complexity_based_filtering()
        self.non_trivial_stage_separation()
        self.waypoints.calc_waypoints_value_scores(
            self.VALUE_SCORE_ALPHA,
            self.VALUE_SCORE_BETA,
            self.VALUE_SCORE_GAMMA
        )
        if self.debug:
            print(self.waypoints.recover_calltrace(debug=True))
        logger.info(f"Before recursively_greedy_triming, waypoints length: {self.waypoints.length}")
        self.recursively_greedy_triming()
        self.remove_outliers()
        print(self.waypoints.recover_calltrace())
        logger.info(f"Final waypoints extraction done, waypoints length: {self.waypoints.length}")

    def get_pcs(self):
        targets = self.waypoints.recover_calltrace().split('\n')
        targets.reverse()
        full_targets, target_pcs = action_get(
            kernel_dir=self.kernel_dir,
            targets=targets,
            api_mode=True,
            fast_mode=True
        )
        full_targets, target_pcs = deduplicate_target_pcs(full_targets, target_pcs)
        print(f"\n| %0-60s | %0-18s |"%("func_name@file_path:line_num", "pc_addr"))
        for i in range(len(target_pcs)):
            print(f"| {full_targets[i]} | {target_pcs[i]} |")
        print(f"\n[For SyzPilot-fuzzer:]")
        s = ",".join([f"\"0x{pc[-8:]}\"" for pc in target_pcs])
        print(f"[{s}]")
        print(f"\n[For bench_parser:]")
        s = " ".join([f"reachability-0x{pc[-8:]}" for pc in target_pcs])
        print(s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract waypoints from syzkaller-style report")
    parser.add_argument("-k", "--kernel_dir", type=str, required=True, help="directory of kernel object")
    parser.add_argument("-t", "--title_path", type=str, default="case_1.title")
    parser.add_argument("-r", "--report_path", type=str, default="case_1.report")
    parser.add_argument("-d", "--debug", action="store_true", help="print debug information")
    args = parser.parse_args()

    waypoints_extractor = WaypointsExtractor(args.kernel_dir, args.title_path, args.report_path, args.debug)
    waypoints_extractor.extract()

    logger.info(f"Start to extract pc addresses for the waypoints...")
    waypoints_extractor.get_pcs()

# python waypoints_extractor.py -k ~/kernels/SyzDirect-targets/case_1/ -t ~/kernels/SyzDirect-targets/configs/case_1.title -r ~/kernels/SyzDirect-targets/configs/case_1.report
