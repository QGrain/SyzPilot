"""
StaticAnalyzer: Bridge to KallGraph call graph analysis for syscall entry inference.

Given a target vulnerability function, performs backward BFS on the call graph
to find all syscall entry points that can reach the target. Maps kernel function
names to syzkaller syscall descriptions.
"""

import csv
import json
import logging
import os
import re
import stat
import threading
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import Future
from pathlib import Path
from types import MappingProxyType
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Kernel syscall entry prefixes
SYSCALL_ENTRY_PREFIXES = ("__sys_", "__do_sys_", "sys_")

# Default syscall mapping (kernel entry → syzkaller names)
# This is a curated subset; the full mapping is in syscall_mapping.json
DEFAULT_SYSCALL_MAPPING = {
    "__sys_socket": ["socket$inet", "socket$inet6", "socket$packet", "socket$netlink",
                     "socket$l2tp6", "socket$nl_generic", "socket$nl_netfilter"],
    "__sys_socketpair": ["socketpair$inet", "socketpair$unix"],
    "__sys_bind": ["bind", "bind$inet", "bind$inet6", "bind$packet", "bind$802154_dgram"],
    "__sys_listen": ["listen"],
    "__sys_accept": ["accept$inet", "accept$inet6", "accept$unix"],
    "__sys_connect": ["connect$inet", "connect$inet6", "connect$unix"],
    "__sys_sendto": ["sendto$inet", "sendto$inet6", "sendto$packet"],
    "__sys_sendmsg": ["sendmsg$inet", "sendmsg$inet6", "sendmsg$nl_route_sched",
                      "sendmsg$NFT_BATCH", "sendmsg$NBD_CMD_DISCONNECT"],
    "__sys_sendmmsg": ["sendmmsg$inet", "sendmmsg$inet6"],
    "__sys_recvfrom": ["recvfrom$inet", "recvfrom$inet6"],
    "__sys_recvmsg": ["recvmsg$inet", "recvmsg$inet6"],
    "__sys_setsockopt": ["setsockopt$inet_tcp", "setsockopt$inet_udp",
                         "setsockopt$packet_int", "setsockopt$pppl2tp_PPPOL2TP_SO_LNSMODE"],
    "__sys_getsockopt": ["getsockopt$inet_tcp", "getsockopt$inet_udp"],
    "__sys_getsockname": ["getsockname", "getsockname$packet"],
    "__sys_getpeername": ["getpeername", "getpeername$inet"],
    "__sys_shutdown": ["shutdown$inet"],
    "__sys_ioctl": ["ioctl$sock_SIOCGIFINDEX", "ioctl$sock_SIOCGIFFLAGS"],
    "__sys_read": ["read", "read$proc"],
    "__sys_write": ["write", "write$proc"],
    "__sys_openat": ["openat", "openat$dir", "openat$proc"],
    "__sys_close": ["close"],
    "__sys_mmap": ["mmap"],
    "__sys_mprotect": ["mprotect"],
    "__sys_munmap": ["munmap"],
    "__sys_poll": ["poll"],
    "__sys_select": ["select"],
    "__sys_epoll_create": ["epoll_create"],
    "__sys_epoll_ctl": ["epoll_ctl"],
    "__sys_epoll_wait": ["epoll_wait"],
    "__sys_dup": ["dup"],
    "__sys_dup2": ["dup2"],
    "__sys_fcntl": ["fcntl$dupfd", "fcntl$getflags", "fcntl$setflags"],
    "__sys_fstat": ["fstat"],
    "__sys_lstat": ["lstat"],
    "__sys_stat": ["stat"],
    "__sys_access": ["access"],
    "__sys_pipe": ["pipe"],
    "__sys_fork": ["fork", "clone"],
    "__sys_execve": ["execve"],
    "__sys_exit": ["exit"],
    "__sys_kill": ["kill"],
    "__sys_futex": ["futex"],
    "__sys_semop": ["semop"],
    "__sys_semget": ["semget"],
    "__sys_msgget": ["msgget"],
    "__sys_msgsnd": ["msgsnd"],
    "__sys_msgrcv": ["msgrcv"],
    "__sys_shmget": ["shmget"],
    "__sys_shmat": ["shmat"],
    "__sys_shmdt": ["shmdt"],
    "__sys_nanosleep": ["nanosleep"],
    "__sys_clock_gettime": ["clock_gettime"],
    "__sys_clock_getres": ["clock_getres"],
    "__sys_getpid": ["getpid"],
    "__sys_getuid": ["getuid"],
    "__sys_getgid": ["getgid"],
    "__sys_gettid": ["gettid"],
    "__sys_getrandom": ["getrandom"],
    "__sys_ptrace": ["ptrace"],
}


class StaticAnalyzer:
    """Bridge to KallGraph call graph analysis for syscall entry inference."""

    _graph_cache_lock = threading.Lock()
    _graph_cache = OrderedDict()
    _graph_load_futures = {}
    _graph_cache_capacity = 4

    def __init__(self, kallgraph_output_dir: str, target_func: str,
                 syscall_mapping_path: Optional[str] = None,
                 max_scan_entries: int = 256,
                 max_callgraph_bytes: int = 512 * 1024 * 1024,
                 trusted_roots: Optional[Tuple[str, ...]] = None):
        """
        Args:
            kallgraph_output_dir: Path to KallGraph output containing callgraph.csv
            target_func: Target vulnerability function name (e.g., 'validate_xmit_skb')
            syscall_mapping_path: Optional path to syscall_mapping.json
        """
        self.kallgraph_dir = kallgraph_output_dir
        self.target_func = target_func
        self.max_scan_entries = max_scan_entries
        self.max_callgraph_bytes = max_callgraph_bytes
        self.trusted_roots = tuple(trusted_roots or (kallgraph_output_dir,))
        self.callgraph_path = self._find_latest_callgraph(kallgraph_output_dir)

        # Graph structures
        self.graph: Dict[str, Set[str]] = defaultdict(set)  # callee → callers
        self.name_to_vid: Dict[str, int] = {}
        self.vid_to_name: Dict[str, str] = {}
        self.graph_cache_hit = False

        # Syscall mapping
        self.syscall_mapping = dict(DEFAULT_SYSCALL_MAPPING)
        if syscall_mapping_path and os.path.exists(syscall_mapping_path):
            with open(syscall_mapping_path) as f:
                self.syscall_mapping.update(json.load(f))

    def _find_latest_callgraph(self, output_dir: str) -> Optional[str]:
        """Find the latest callgraph.csv in KallGraph output directory."""
        output_path = Path(output_dir)
        if not output_path.exists():
            logger.warning(f"KallGraph output dir not found: {output_dir}")
            return None

        resolved_output = output_path.resolve()
        candidates = []
        entries = 0
        search_dirs = [resolved_output]
        try:
            for child in resolved_output.iterdir():
                entries += 1
                if entries > self.max_scan_entries:
                    logger.error(
                        "KallGraph directory exceeds scan-entry limit: %s",
                        output_dir,
                    )
                    return None
                if child.is_dir():
                    search_dirs.append(child.resolve())
        except OSError as error:
            logger.error("Failed to scan KallGraph directory %s: %s", output_dir, error)
            return None

        for search_dir in search_dirs:
            if not search_dir.is_relative_to(resolved_output):
                continue
            for filename in ("callgraph.csv", "complete_callgraph"):
                candidate = search_dir / filename
                if candidate.is_file() and candidate.resolve().is_relative_to(resolved_output):
                    candidates.append(candidate.resolve())
        if candidates:
            latest = max(candidates, key=lambda path: path.stat().st_mtime)
            logger.info(f"Found callgraph: {latest}")
            return str(latest)

        logger.warning(f"No callgraph found in {output_dir}")
        return None

    def load_callgraph(self) -> bool:
        """Load callgraph into memory. Returns True if successful."""
        if not self.callgraph_path:
            return False

        self.graph = defaultdict(set)
        self.name_to_vid = {}
        self.vid_to_name = {}
        self.graph_cache_hit = False
        cache_key = None
        load_future = None
        owns_load = False

        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        try:
            descriptor = os.open(self.callgraph_path, flags)
        except OSError as error:
            logger.error(f"Unable to open callgraph {self.callgraph_path}: {error}")
            return False

        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("callgraph path must be a regular file")
            if metadata.st_size > self.max_callgraph_bytes:
                raise ValueError(
                    f"callgraph exceeds size limit ({metadata.st_size} > "
                    f"{self.max_callgraph_bytes})"
                )

            opened_path = Path(f"/proc/self/fd/{descriptor}").resolve(strict=True)
            trusted_roots = [
                Path(root).expanduser().resolve(strict=True)
                for root in self.trusted_roots
            ]
            if not any(opened_path.is_relative_to(root) for root in trusted_roots):
                raise ValueError("opened callgraph is outside trusted roots")

            cache_key = (
                str(opened_path),
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
            with self._graph_cache_lock:
                cached = self._graph_cache.get(cache_key)
                if cached is not None:
                    self._graph_cache.move_to_end(cache_key)
                else:
                    load_future = self._graph_load_futures.get(cache_key)
                    if load_future is None:
                        load_future = Future()
                        self._graph_load_futures[cache_key] = load_future
                        owns_load = True
            if cached is not None:
                self.graph, self.name_to_vid, self.vid_to_name = cached
                self.graph_cache_hit = True
                logger.info(
                    "Reused cached callgraph: %d functions, %d edges",
                    len(self.name_to_vid),
                    sum(len(callers) for callers in self.graph.values()),
                )
                return True
            if not owns_load:
                cached = load_future.result()
                current_metadata = os.fstat(descriptor)
                current_identity = (
                    str(opened_path),
                    current_metadata.st_dev,
                    current_metadata.st_ino,
                    current_metadata.st_size,
                    current_metadata.st_mtime_ns,
                    current_metadata.st_ctime_ns,
                )
                if current_identity != cache_key:
                    raise ValueError(
                        "callgraph changed while waiting for a concurrent load"
                    )
                if cached is None:
                    raise ValueError("concurrent callgraph load failed")
                self.graph, self.name_to_vid, self.vid_to_name = cached
                self.graph_cache_hit = True
                logger.info(
                    "Reused concurrently loaded callgraph: %d functions, "
                    "%d edges",
                    len(self.name_to_vid),
                    sum(len(callers) for callers in self.graph.values()),
                )
                return True

            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                lines = self._bounded_text_lines(stream)
                if self.callgraph_path.endswith('.csv'):
                    reader = csv.DictReader(lines)
                    fieldnames = {
                        name.strip().lstrip("\ufeff"): name
                        for name in (reader.fieldnames or [])
                        if name
                    }
                    caller_field = fieldnames.get("caller_name")
                    callee_field = fieldnames.get("callee_name")
                    if not caller_field or not callee_field:
                        raise ValueError(
                            "KallGraph CSV must contain caller_name and callee_name columns"
                        )
                    for row in reader:
                        caller = (row.get(caller_field) or "").strip()
                        callee = (row.get(callee_field) or "").strip()
                        if caller and callee:
                            self.graph[callee].add(caller)
                            self._register_function(caller)
                            self._register_function(callee)
                else:
                    # Text format: "caller -> callee"
                    for line in lines:
                        line = line.strip()
                        if '->' in line:
                            parts = line.split('->')
                            if len(parts) == 2:
                                caller = parts[0].strip().strip('"')
                                callee = parts[1].strip().strip('"')
                                if caller and callee:
                                    self.graph[callee].add(caller)
                                    self._register_function(caller)
                                    self._register_function(callee)
            final_metadata = os.fstat(descriptor)
            final_identity = (
                str(opened_path),
                final_metadata.st_dev,
                final_metadata.st_ino,
                final_metadata.st_size,
                final_metadata.st_mtime_ns,
                final_metadata.st_ctime_ns,
            )
            if final_identity != cache_key:
                raise ValueError("callgraph changed while it was being read")
        except Exception as e:
            logger.error(f"Failed to load callgraph: {e}")
            if owns_load and load_future is not None:
                should_signal_failure = False
                with self._graph_cache_lock:
                    if self._graph_load_futures.get(cache_key) is load_future:
                        del self._graph_load_futures[cache_key]
                        should_signal_failure = True
                if should_signal_failure and not load_future.done():
                    load_future.set_result(None)
            self.graph = defaultdict(set)
            self.name_to_vid = {}
            self.vid_to_name = {}
            return False
        finally:
            if descriptor >= 0:
                os.close(descriptor)

        try:
            self.graph = MappingProxyType({
                callee: frozenset(callers)
                for callee, callers in self.graph.items()
            })
            self.name_to_vid = MappingProxyType(dict(self.name_to_vid))
            self.vid_to_name = MappingProxyType(dict(self.vid_to_name))
            cached_graph = (
                self.graph,
                self.name_to_vid,
                self.vid_to_name,
            )
            published = False
            with self._graph_cache_lock:
                if self._graph_load_futures.get(cache_key) is load_future:
                    self._graph_cache[cache_key] = cached_graph
                    self._graph_cache.move_to_end(cache_key)
                    capacity = max(1, self._graph_cache_capacity)
                    while len(self._graph_cache) > capacity:
                        self._graph_cache.popitem(last=False)
                    del self._graph_load_futures[cache_key]
                    published = True
            if not published:
                logger.info("Callgraph load was invalidated before publication")
                self.graph = defaultdict(set)
                self.name_to_vid = {}
                self.vid_to_name = {}
                return False
            if not load_future.done():
                load_future.set_result(cached_graph)
        except Exception as error:
            logger.error("Failed to publish callgraph cache: %s", error)
            if owns_load and load_future is not None:
                should_signal_failure = False
                with self._graph_cache_lock:
                    if self._graph_load_futures.get(cache_key) is load_future:
                        del self._graph_load_futures[cache_key]
                        should_signal_failure = True
                if should_signal_failure and not load_future.done():
                    load_future.set_result(None)
            self.graph = defaultdict(set)
            self.name_to_vid = {}
            self.vid_to_name = {}
            return False

        logger.info(f"Loaded callgraph: {len(self.name_to_vid)} functions, "
                    f"{sum(len(v) for v in self.graph.values())} edges")
        return True

    @classmethod
    def clear_graph_cache(cls):
        """Clear the process-local graph cache, primarily for bounded tests."""
        with cls._graph_cache_lock:
            cls._graph_cache.clear()
            pending_futures = list(cls._graph_load_futures.values())
            cls._graph_load_futures.clear()
        for future in pending_futures:
            if not future.done():
                future.set_result(None)

    def _bounded_text_lines(self, stream):
        """Yield UTF-8 lines while enforcing the byte limit during reads."""
        remaining = self.max_callgraph_bytes
        while True:
            chunk = stream.readline(remaining + 1)
            if not chunk:
                return
            if len(chunk) > remaining:
                raise ValueError("callgraph grew beyond the configured size limit")
            remaining -= len(chunk)
            yield chunk.decode("utf-8")

    def _register_function(self, name: str):
        if name in self.name_to_vid:
            return
        vid = len(self.name_to_vid)
        self.name_to_vid[name] = vid
        self.vid_to_name[str(vid)] = name

    @staticmethod
    def _canonical_symbol(name: str) -> str:
        """Remove compiler clone suffixes without using substring matching."""
        symbol = name.strip().strip('"')
        return re.sub(r"\.(?:isra|constprop|part|cold)(?:\.\d+)*$", "", symbol)

    def find_reachable_syscall_entries(self) -> List[Dict]:
        """Backward BFS from target to find all syscall entry points.

        Returns:
            List of dicts with 'name', 'kernel_name', 'path_length', 'weight'
        """
        if not self.graph:
            logger.error("Callgraph not loaded")
            return []

        # Find target function in graph
        target_symbol = self._canonical_symbol(self.target_func)
        target_names = [
            name for name in self.name_to_vid
            if self._canonical_symbol(name) == target_symbol
        ]

        if not target_names:
            logger.error(f"Target function '{self.target_func}' not found in callgraph")
            return []

        # Backward BFS
        visited: Dict[str, int] = {}  # function_name → shortest distance
        queue = deque()

        for target in target_names:
            if target in self.name_to_vid:
                visited[target] = 0
                queue.append(target)

        while queue:
            current = queue.popleft()
            current_dist = visited[current]

            for caller in self.graph.get(current, set()):
                if caller not in visited or visited[caller] > current_dist + 1:
                    visited[caller] = current_dist + 1
                    queue.append(caller)

        # Find syscall entries
        syscall_entries = []
        for func_name, dist in visited.items():
            if any(func_name.startswith(prefix) for prefix in SYSCALL_ENTRY_PREFIXES):
                # Map to syzkaller names
                syzkaller_names = self._map_kernel_to_syzkaller(func_name)
                weight = max(0.1, 1.0 / (dist + 1))  # inverse distance

                for syz_name in syzkaller_names:
                    syscall_entries.append({
                        "name": syz_name,
                        "kernel_name": func_name,
                        "path_length": dist,
                        "weight": weight,
                        "source": "static_analysis",
                        "guidance_level": (
                            "syz_call" if "$" in syz_name or
                            syz_name.startswith("syz_") else "system_call"
                        ),
                    })

        # Sort by weight (descending)
        syscall_entries.sort(key=lambda x: x["weight"], reverse=True)

        logger.info(f"Found {len(syscall_entries)} syscall entries reachable from {self.target_func}")
        return syscall_entries

    def _map_kernel_to_syzkaller(self, kernel_func: str) -> List[str]:
        """Map kernel function name to syzkaller syscall names."""
        kernel_func = self._canonical_symbol(kernel_func)
        # Direct mapping
        if kernel_func in self.syscall_mapping:
            return self.syscall_mapping[kernel_func]

        # Try without prefix
        for prefix in SYSCALL_ENTRY_PREFIXES:
            if kernel_func.startswith(prefix):
                base_name = kernel_func[len(prefix):]
                matches = set()
                for key, values in self.syscall_mapping.items():
                    canonical_key = self._canonical_symbol(key)
                    key_base = canonical_key
                    for key_prefix in SYSCALL_ENTRY_PREFIXES:
                        if canonical_key.startswith(key_prefix):
                            key_base = canonical_key[len(key_prefix):]
                            break
                    if key_base == base_name:
                        matches.update(values)
                if matches:
                    return sorted(matches)

        return []

    def get_injection_guidance(self) -> Dict:
        """Main entry point: returns complete guidance for fuzzer injection."""
        entries = self.find_reachable_syscall_entries()

        # Deduplicate and aggregate weights
        weight_map: Dict[str, float] = {}
        for entry in entries:
            name = entry["name"]
            w = entry["weight"]
            if name not in weight_map or w > weight_map[name]:
                weight_map[name] = w

        return {
            "target_func": self.target_func,
            "syscall_entries": entries,
            "syscall_weights": weight_map,
            "analysis_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }


# Import time for timestamp
import time
