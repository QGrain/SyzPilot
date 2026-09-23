"""
PathBasedAnalyzer: Infers relevant syzkaller syscalls from crash report source paths.

This is a lightweight alternative to KallGraph static analysis. It parses kernel
crash reports to extract function names and source file paths, then matches them
against syzkaller's syscall description files to infer relevant syscalls.

Four-layer matching strategy with two cold-start guidance levels:
1. Direct syscall entry extraction (__sys_*, __x64_sys_* prefixes in call trace)
2. Direct-entry-constrained Syzlang variant matching against trace functions,
   unique source-family evidence, and exact filesystem-image variants
3. Source path → syzkaller description file reverse mapping
4. Function keyword → syscall inference

This produces output compatible with GuidanceEngine.update_static_analysis().
"""

import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Kernel syscall entry prefixes
SYSCALL_ENTRY_PREFIXES = (
    "__arm64_sys_", "compat_sys_", "__x64_sys_", "__ia32_sys_",
    "____sys_", "___sys_", "__do_sys_", "__se_sys_", "__sys_",
)
_COMPILER_FUNCTION_SUFFIX = re.compile(
    r"\.(?:(?:isra|constprop|part)\.\d+|cold(?:\.\d+)?)$"
)

# Wrapper/dispatch words do not identify a Syzlang operation variant.  Removing
# them lets, for example, KVM_SET_VCPU_EVENTS match the kernel function
# kvm_vcpu_ioctl_x86_set_vcpu_events without accepting unrelated KVM ioctls.
_VARIANT_STOP_TOKENS = {
    "cmd", "compat", "ioc", "ioctl", "x64", "x86", "amd64", "64",
}


def _identifier_tokens(value: str) -> List[str]:
    """Return normalized identifier components used for conservative matching."""
    return [
        token for token in re.findall(r"[a-z]+\d*|\d+", value.lower())
        if token not in _VARIANT_STOP_TOKENS
    ]


def _is_ordered_subsequence(needles: List[str], haystack: List[str]) -> bool:
    """Return whether all needles occur in order in haystack."""
    position = 0
    for token in haystack:
        if position < len(needles) and needles[position] == token:
            position += 1
    return position == len(needles)


def _direct_syscall_base(function_name: str) -> Optional[str]:
    """Extract a primitive syscall name from a known kernel entry wrapper."""
    normalized = function_name
    while True:
        stripped = _COMPILER_FUNCTION_SUFFIX.sub("", normalized)
        if stripped == normalized:
            break
        normalized = stripped
    for prefix in SYSCALL_ENTRY_PREFIXES:
        if normalized.startswith(prefix):
            bare_name = normalized[len(prefix):]
            return bare_name or None
    return None


def _parse_call_trace(report_text: str) -> List[Tuple[str, str]]:
    """Parse crash report call trace to extract (function_name, source_path) pairs.

    Handles formats like:
        validate_xmit_skb+0xbd5/0xee0 net/core/dev.c:3763
        packet_sendmsg+0x22fc/0x52b0 net/packet/af_packet.c:3044
        __x64_sys_sendto+0xdd/0x1b0 net/socket.c:2027

    Returns:
        List of (function_name, source_file_path) tuples.
    """
    pairs = []
    in_trace = False

    for line in report_text.split('\n'):
        line = line.strip()

        # Detect call trace start
        if 'Call Trace:' in line:
            in_trace = True
            continue

        # Detect call trace end (empty line or register dump)
        if in_trace and (not line or line.startswith('RIP:') or line.startswith('Code:')):
            if pairs:  # Only stop if we've collected some entries
                break
            continue

        if not in_trace:
            continue

        # Skip inline markers
        line = line.replace('[inline]', '').strip()

        # Parse: func_name+0xoffset/0xsize filepath:lineno
        # Also handle: func_name filepath:lineno (without offset)
        m = re.match(r'(\S+?)(?:\+0x[0-9a-f]+/0x[0-9a-f]+)?\s+(\S+\.[ch])(?::\d+)?', line)
        if m:
            func_name = m.group(1).strip()
            source_path = m.group(2).strip()
            pairs.append((func_name, source_path))

    return pairs


def _extract_source_paths(report_text: str) -> List[str]:
    """Extract all unique kernel source file paths from a crash report."""
    paths = []
    seen = set()
    for func, path in _parse_call_trace(report_text):
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def _extract_function_names(report_text: str) -> List[str]:
    """Extract all unique function names from a crash report call trace."""
    names = []
    seen = set()
    for func, _ in _parse_call_trace(report_text):
        if func not in seen:
            seen.add(func)
            names.append(func)
    return names


@dataclass(frozen=True)
class ReportDispatchObservation:
    """One ABI-validated fixed dispatch argument observed in a report."""

    call_name: str
    arch: str
    syscall_nr: int
    fixed_arg_index: int
    value: int
    provenance: str = "report_register"


def _extract_x86_64_ioctl_observation(
        report_text: str) -> Optional[ReportDispatchObservation]:
    """Extract one report-backed ioctl command from an amd64 register dump.

    The direct native wrapper, a user-mode RIP, and ``ORIG_RAX=0x10`` are all
    required. Conflicting qualifying dumps fail closed. RDI and RDX are not
    consumed because they are execution-specific fd and pointer values.
    """
    lines = report_text.splitlines()
    observations = []
    for trace_index, line in enumerate(lines):
        if "Call Trace:" not in line:
            continue
        frame_names = []
        register_index = None
        for section_index in range(trace_index + 1, len(lines)):
            stripped = lines[section_index].strip()
            if "Call Trace:" in stripped or stripped.startswith("===="):
                break
            if stripped.startswith("RIP:"):
                register_index = section_index
                break
            if not stripped:
                break
            frame_match = re.match(
                r"(\S+?)(?:\+0x[0-9a-f]+/0x[0-9a-f]+)?(?:\s|$)",
                stripped,
            )
            if frame_match:
                frame_names.append(frame_match.group(1))
        if register_index is None or not any(
                _COMPILER_FUNCTION_SUFFIX.sub("", function_name) ==
                "__x64_sys_ioctl"
                for function_name in frame_names):
            continue

        register_lines = []
        for section_index in range(register_index, len(lines)):
            stripped = lines[section_index].strip()
            if (section_index > register_index and
                    (not stripped or "Call Trace:" in stripped or
                     stripped.startswith("===="))):
                break
            register_lines.append(lines[section_index])
        block = " ".join(register_lines)
        if re.search(r"\bRIP:\s*0033:", block) is None:
            continue
        registers = {
            name: int(encoded, 16)
            for name, encoded in re.findall(
                r"\b(ORIG_RAX|RSI):\s*([0-9a-fA-F]{1,16})\b", block
            )
        }
        if registers.get("ORIG_RAX") == 0x10 and "RSI" in registers:
            observations.append(registers["RSI"])
    if len(observations) != 1:
        return None
    return ReportDispatchObservation(
        call_name="ioctl",
        arch="amd64",
        syscall_nr=0x10,
        fixed_arg_index=1,
        value=observations[0],
    )


class PathBasedAnalyzer:
    """Infers relevant syscalls from crash report source paths.

    Builds a reverse mapping from kernel source paths to syzkaller
    syscall description files, then matches crash report paths.
    """

    def __init__(self, syzkaller_syslinux_dir: str, enabled_syscalls_path: Optional[str] = None):
        """
        Args:
            syzkaller_syslinux_dir: Path to syzkaller's sys/linux/ directory
            enabled_syscalls_path: Path to enabled syscalls file (optional)
        """
        self.syslinux_dir = syzkaller_syslinux_dir
        self.enabled_syscalls: Set[str] = set()

        # Load enabled syscalls if provided
        if enabled_syscalls_path and os.path.exists(enabled_syscalls_path):
            with open(enabled_syscalls_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        self.enabled_syscalls.add(parts[2])

        # Build the mappings
        self._desc_file_to_syscalls: Dict[str, List[str]] = {}
        self._path_to_desc_files: Dict[str, List[str]] = defaultdict(list)
        self._ioctl_constants: Dict[str, str] = {}
        self._base_to_variants: Dict[str, Set[str]] = defaultdict(set)

        self._build_mappings()

    @staticmethod
    def extract_report_dispatch_constant(
            report_text: str, target_arch: str,
    ) -> Optional[ReportDispatchObservation]:
        """Return a conservative direct-dispatch constant observation."""
        if target_arch != "amd64":
            return None
        return _extract_x86_64_ioctl_observation(report_text)

    def _build_mappings(self):
        """Build reverse mapping from kernel paths to syzkaller description files.

        Mapping strategy:
        1. Parse each .txt file to extract syscall names
        2. Infer kernel source paths from file naming conventions
        3. Build reverse index: kernel_path_prefix → [desc_file1, desc_file2, ...]
        """
        syslinux = Path(self.syslinux_dir)
        if not syslinux.exists():
            logger.error(f"sys/linux/ directory not found: {syslinux}")
            return

        desc_files = sorted(syslinux.glob('*.txt'))
        # Filter out .const and .warn files
        desc_files = [f for f in desc_files
                      if not f.name.endswith('.const') and not f.name.endswith('.warn')]

        for desc_file in desc_files:
            fname = desc_file.stem  # e.g., "socket_packet"

            # Parse syscall names from this file
            syscalls = self._extract_syscalls_from_file(str(desc_file))
            if not syscalls:
                continue

            self._desc_file_to_syscalls[fname] = syscalls

            # Infer kernel paths from file name
            kernel_paths = self._infer_kernel_paths(fname)
            for kpath in kernel_paths:
                self._path_to_desc_files[kpath].append(fname)

            # Build ioctl constant index
            for sc in syscalls:
                if '$' in sc:
                    base_name, _ = sc.split('$', 1)
                    self._base_to_variants[base_name].add(sc)
                if sc.startswith('ioctl$'):
                    const_name = sc.split('ioctl$', 1)[1].lower()
                    self._ioctl_constants[const_name] = sc

        logger.info(f"[PathAnalyzer] Built mappings: {len(self._desc_file_to_syscalls)} desc files, "
                     f"{len(self._path_to_desc_files)} kernel path entries, "
                     f"{len(self._ioctl_constants)} ioctl constants, "
                     f"{sum(map(len, self._base_to_variants.values()))} variants")

    def _extract_syscalls_from_file(self, filepath: str) -> List[str]:
        """Extract syscall names from a syzkaller description file."""
        syscalls = []
        try:
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('include'):
                        continue
                    if line.startswith('define') or line.startswith('type'):
                        continue
                    # Match: syscall_name(args) or syscall_name(args) return_type
                    m = re.match(r'^([a-zA-Z_][a-zA-Z0-9_\$]*)\s*\(', line)
                    if m:
                        syscalls.append(m.group(1))
        except Exception as e:
            logger.warning(f"Failed to parse {filepath}: {e}")
        return syscalls

    def _infer_kernel_paths(self, desc_name: str) -> List[str]:
        """Infer kernel source paths from syzkaller description file name.

        Naming conventions:
        - socket_<family> → net/<family>/  (e.g., socket_packet → net/packet/)
        - socket_netlink_<subsystem> → net/<subsystem>/ (handles nested netlink)
        - dev_<device> → drivers/<device>/ or drivers/<subsystem>/
        - <subsystem> → <subsystem>/ (top-level subsystem)

        Design principle: use SPECIFIC path prefixes only. Avoid broad directories
        like net/core/ which would cause false positives.
        """
        paths = []

        # socket_netlink_<subsystem> → net/<subsystem>/
        if desc_name.startswith('socket_netlink_'):
            subsystem = desc_name[len('socket_netlink_'):]
            subsystem = re.sub(r'_retired$', '', subsystem)
            netlink_path_map = {
                'route': 'net/ipv4/route.c',
                'route_sched': 'net/sched/',
                'route_newroute': 'net/sched/',
                'route_newqdisc': 'net/sched/',
                'route_delroute': 'net/sched/',
                'route_getroute': 'net/sched/',
                'route_act': 'net/sched/',
                'route_cls': 'net/sched/',
                'route_police': 'net/sched/',
                'generic': 'net/netlink/',
                'generic_80211': 'net/wireless/',
                'generic_batadv': 'net/batman-adv/',
                'generic_devlink': 'net/devlink/',
                'generic_ethtool': 'net/ethtool/',
                'generic_fou': 'net/ipv4/fou.c',
                'generic_gtp': 'net/gtp/',
                'generic_mptcp': 'net/mptcp/',
                'generic_nfc': 'net/nfc/',
                'generic_team': 'drivers/net/team/',
                'audit': 'kernel/audit/',
                'crypto': 'crypto/',
                'kobject': 'lib/kobject.c',
                'netfilter': 'net/netfilter/',
                'selinux': 'security/selinux/',
            }
            if subsystem in netlink_path_map:
                paths.append(netlink_path_map[subsystem])
            else:
                paths.append(f'net/{subsystem}/')
            return paths

        # socket_<family> → net/<family>/
        if desc_name.startswith('socket_'):
            family = desc_name[len('socket_'):]
            family_path_map = {
                'inet': 'net/ipv4/af_inet.c',
                'inet6': 'net/ipv6/af_inet6.c',
                'inet_tcp': 'net/ipv4/tcp.c',
                'inet_udp': 'net/ipv4/udp.c',
                'inet6_tcp': 'net/ipv6/tcp_ipv6.c',
                'inet6_udp': 'net/ipv6/udp.c',
                'netlink': 'net/netlink/af_netlink.c',
                'unix': 'net/unix/af_unix.c',
                'packet': 'net/packet/af_packet.c',
                'vnet': 'drivers/net/vnet.c',
                'can_raw': 'net/can/raw.c',
                'tipc': 'net/tipc/',
                'tipc_netlink': 'net/tipc/',
                'bluetooth': 'net/bluetooth/',
                'caif': 'net/caif/',
                'nfc': 'net/nfc/',
                'pppox': 'drivers/net/ppp/',
                'rxrpc': 'net/rxrpc/',
                'x25': 'net/x25/',
                'rose': 'net/rose/',
                'ax25': 'net/ax25/',
                'rds': 'net/rds/',
                'l2tp': 'net/l2tp/',
                'qrtr': 'net/qrtr/',
                'xdp': 'net/xdp/',
            }
            if family in family_path_map:
                paths.append(family_path_map[family])
            return paths

        # dev_<device> → drivers/<device>/
        if desc_name.startswith('dev_'):
            device = desc_name[len('dev_'):]
            device_path_map = {
                'binder': 'drivers/android/',
                'binderfs': 'drivers/android/',
                'block': 'block/',
                'bus_usb': 'drivers/usb/',
                'dri': 'drivers/gpu/drm/',
                'fb': 'drivers/video/fbdev/',
                'floppy': 'drivers/block/',
                'hidraw': 'drivers/hid/',
                'i2c': 'drivers/i2c/',
                'i915': 'drivers/gpu/drm/i915/',
                'infiniband_rdma': 'drivers/infiniband/',
                'infiniband_rdma_cm': 'drivers/infiniband/',
                'input': 'drivers/input/',
                'kvm': 'virt/kvm/',
                'kvm_amd64': 'virt/kvm/',
                'kvm_arm64': 'virt/kvm/',
                'loop': 'drivers/block/',
                'media': 'drivers/media/',
                'mali': 'drivers/gpu/arm/',
                'net_tun': 'drivers/net/tun.c',
                'ppp': 'drivers/net/ppp/',
                'sg': 'drivers/scsi/',
                'snd': 'sound/',
                'snd_timer': 'sound/core/',
                'vhci': 'drivers/bluetooth/',
                'vhost_net': 'drivers/vhost/',
                'video': 'drivers/media/',
            }
            if device in device_path_map:
                paths.append(device_path_map[device])
            return paths

        # Subsystem-level files — use SPECIFIC file paths, not broad directories
        subsystem_path_map = {
            'bpf': ['kernel/bpf/'],
            'bpf_prog': ['kernel/bpf/'],
            'bpf_trace': ['kernel/trace/', 'kernel/bpf/'],
            'cgroup': ['kernel/cgroup/'],
            'epoll': ['fs/eventpoll.c'],
            'fcntl': ['fs/fcntl.c'],
            'fuse': ['fs/fuse/'],
            'io_uring': ['io_uring/'],
            'keyring': ['security/keys/'],
            'kill': ['kernel/signal.c'],
            'landlock': ['security/landlock/'],
            'mmap': ['mm/mmap.c'],
            'openat': ['fs/open.c'],
            'perf_event': ['kernel/events/'],
            'pidfd': ['kernel/pid.c'],
            'ptp': ['drivers/ptp/'],
            'seccomp': ['kernel/seccomp.c'],
            'sys': ['kernel/sys.c'],
            'timer': ['kernel/time/'],
            'uffd': ['mm/userfaultfd.c'],
            'xattr': ['fs/xattr.c'],
        }
        if desc_name in subsystem_path_map:
            paths.extend(subsystem_path_map[desc_name])

        return paths

    def analyze_report(self, report_text: str) -> List[Dict]:
        """Main entry point: analyze a crash report and return syscall entries.

        Args:
            report_text: Full text of the kernel crash report

        Returns:
            List of dicts with 'name', 'kernel_name', 'path_length', 'weight', 'source' keys.
            Compatible with GuidanceEngine.update_static_analysis().
        """
        trace_pairs = _parse_call_trace(report_text)
        source_paths = _extract_source_paths(report_text)
        function_names = _extract_function_names(report_text)

        logger.info(f"[PathAnalyzer] Parsed call trace: {len(trace_pairs)} entries, "
                     f"{len(source_paths)} unique paths, {len(function_names)} unique functions")

        # Collect all matched syscall names with their evidence
        syscall_scores: Dict[str, float] = {}  # syscall_name → best_score
        syscall_sources: Dict[str, str] = {}   # syscall_name → match_source
        direct_bases: Set[str] = set()
        matched_variants: Dict[str, Set[str]] = defaultdict(set)
        source_variants: Dict[str, Dict[str, Tuple[float, str]]] = defaultdict(dict)

        # Layer 1: Direct syscall entry extraction
        for func_name in function_names:
            bare_name = _direct_syscall_base(func_name)
            if bare_name:
                direct_bases.add(bare_name)
                syscall_scores[bare_name] = max(
                    syscall_scores.get(bare_name, 0), 1.0
                )
                syscall_sources[bare_name] = 'direct_entry'

        # Layer 2: Refine only directly observed primitive calls into exact
        # Syzlang variants.  Requiring at least two discriminative suffix
        # tokens avoids expanding a broad primitive into every variant in its
        # family while still matching KVM_SET_VCPU_EVENTS and
        # NL80211_CMD_CONNECT from report function names.
        for base_name in direct_bases:
            for variant in sorted(self._base_to_variants.get(base_name, ())):
                suffix = variant.split('$', 1)[1]
                variant_tokens = _identifier_tokens(suffix)
                if len(variant_tokens) < 2:
                    continue
                for func_name in function_names:
                    if _is_ordered_subsequence(
                            variant_tokens, _identifier_tokens(func_name)):
                        score = 0.95
                        if score > syscall_scores.get(variant, 0):
                            syscall_scores[variant] = score
                            syscall_sources[variant] = (
                                f'variant_match:{base_name}←{func_name}'
                            )
                        matched_variants[base_name].add(variant)
                        break

        # A mount wrapper plus a concrete fs/<type>/ trace directory identifies
        # the corresponding synthetic Syzlang image-mount call without PoC
        # input.  Require an exact normalized suffix match in the compiled
        # descriptions; never expand a generic fs/ frame to every filesystem.
        if "mount" in direct_bases:
            image_variants = sorted(
                self._base_to_variants.get("syz_mount_image", ())
            )
            for source_path in source_paths:
                path_parts = source_path.strip("/").split("/")
                if len(path_parts) < 3 or path_parts[0] != "fs":
                    continue
                filesystem_tokens = _identifier_tokens(path_parts[1])
                if not filesystem_tokens:
                    continue
                for variant in image_variants:
                    suffix = variant.split("$", 1)[1]
                    if _identifier_tokens(suffix) != filesystem_tokens:
                        continue
                    score = 0.98
                    if score > syscall_scores.get(variant, 0):
                        syscall_scores[variant] = score
                        syscall_sources[variant] = (
                            f"filesystem_variant:{source_path}"
                        )

        # Layer 3: Source path → syzkaller description file matching
        for source_path in source_paths:
            matched_descs = self._match_source_path(source_path)
            for desc_name, match_score in matched_descs:
                syscalls = self._desc_file_to_syscalls.get(desc_name, [])
                for sc in syscalls:
                    if sc not in syscall_scores or match_score > syscall_scores[sc]:
                        syscall_scores[sc] = match_score
                        syscall_sources[sc] = f'path_match:{source_path}→{desc_name}'
                    if '$' not in sc:
                        continue
                    base_name, _ = sc.split('$', 1)
                    if base_name not in direct_bases:
                        continue
                    previous = source_variants[base_name].get(sc)
                    if previous is None or match_score > previous[0]:
                        source_variants[base_name][sc] = (
                            match_score, f'{source_path}→{desc_name}'
                        )

        # A one-token variant such as sendmsg$rds cannot satisfy the
        # discriminative token rule above. Refine it only when *all*
        # report-matched description evidence leaves exactly one variant for
        # the directly observed primitive. A high-scoring common dispatcher
        # (for example net/netlink/af_netlink.c) must not suppress a lower-
        # scoring subsystem family such as net/sched. Ambiguous families remain
        # broad instead of guessing.
        for base_name in direct_bases:
            if matched_variants.get(base_name):
                continue
            candidates = source_variants.get(base_name, {})
            if len(candidates) != 1:
                continue
            variant, (_, evidence) = next(iter(candidates.items()))
            score = 0.9
            if score > syscall_scores.get(variant, 0):
                syscall_scores[variant] = score
                syscall_sources[variant] = (
                    f'source_variant:{base_name}←{evidence}'
                )
            matched_variants[base_name].add(variant)

        # Layer 4: Function keyword inference
        for func_name in function_names:
            keyword_matches = self._infer_from_function_name(func_name)
            for sc, score in keyword_matches:
                if sc not in syscall_scores or score > syscall_scores[sc]:
                    syscall_scores[sc] = score
                    syscall_sources[sc] = f'func_inference:{func_name}'

        # Exact trace evidence is more discriminative than a source-directory
        # match.  Once one or more variants of an observed primitive match, do
        # not retain unrelated variants of that same primitive merely because
        # their description file maps to the same broad kernel directory.
        for syscall_name in list(syscall_scores):
            if '$' not in syscall_name:
                continue
            base_name, _ = syscall_name.split('$', 1)
            if (base_name in matched_variants and
                    syscall_name not in matched_variants[base_name]):
                del syscall_scores[syscall_name]
                syscall_sources.pop(syscall_name, None)

        # Filter to enabled syscalls if list is available
        if self.enabled_syscalls:
            syscall_scores = {k: v for k, v in syscall_scores.items()
                             if k in self.enabled_syscalls}

        # Build output
        entries = []
        for sc_name, score in sorted(syscall_scores.items(), key=lambda x: x[1], reverse=True):
            evidence = syscall_sources.get(sc_name, "unknown")
            if evidence == "direct_entry":
                guidance_role = "primitive_fallback"
            elif evidence.startswith((
                    "variant_match:", "source_variant:",
                    "filesystem_variant:")):
                guidance_role = "entry_exact"
            elif evidence.startswith("path_match:"):
                guidance_role = "subsystem_peer"
            else:
                guidance_role = "heuristic_peer"
            entries.append({
                "name": sc_name,
                "kernel_name": evidence,
                "path_length": int(1.0 / max(score, 0.01)),  # inverse of score
                "weight": score,
                "source": "path_analysis",
                "guidance_level": (
                    "syz_call" if "$" in sc_name or
                    sc_name.startswith("syz_") else "system_call"
                ),
                "guidance_role": guidance_role,
            })

        logger.info(f"[PathAnalyzer] Found {len(entries)} relevant syscalls")
        return entries

    def _match_source_path(self, source_path: str) -> List[Tuple[str, float]]:
        """Match a kernel source path to syzkaller description files.

        Returns list of (desc_file_name, match_score) sorted by score.
        """
        matches = []
        # Normalize path
        source_path = source_path.strip('/')

        for kpath_prefix, desc_files in self._path_to_desc_files.items():
            kpath_prefix = kpath_prefix.strip('/')
            if source_path.startswith(kpath_prefix):
                # Score based on prefix specificity (longer match = higher score)
                specificity = len(kpath_prefix.split('/'))
                score = min(0.9, 0.3 + specificity * 0.15)
                for df in desc_files:
                    matches.append((df, score))

        # Also try matching by directory component
        path_parts = source_path.split('/')
        if len(path_parts) >= 2:
            # Try matching first two components (e.g., "net/core")
            dir_prefix = '/'.join(path_parts[:2])
            for kpath_prefix, desc_files in self._path_to_desc_files.items():
                if kpath_prefix.strip('/') == dir_prefix:
                    for df in desc_files:
                        if df not in [m[0] for m in matches]:
                            matches.append((df, 0.4))

        return sorted(matches, key=lambda x: x[1], reverse=True)

    def _infer_from_function_name(self, func_name: str) -> List[Tuple[str, float]]:
        """Infer syscalls from function name keywords.

        Handles cases like:
        - packet_sendmsg → sendto$packet, sendmsg$packet
        - geneve_xmit → (network tunnel, needs netlink config)
        - sch_direct_xmit → (qdisc, needs netlink config)
        """
        matches = []
        func_lower = func_name.lower()

        # Function name → syscall keyword patterns
        keyword_patterns = {
            # Socket family operations
            'packet_send': [('sendto$packet', 0.7), ('sendmsg$packet', 0.6), ('socket$packet', 0.5)],
            'packet_recv': [('recvfrom$packet', 0.7), ('recvmsg$packet', 0.6)],
            'inet6_send': [('sendto$inet6', 0.7), ('sendmsg$inet6', 0.6)],
            'inet_send': [('sendto$inet', 0.7), ('sendmsg$inet', 0.6)],
            'unix_send': [('sendto$unix', 0.6), ('sendmsg$unix', 0.5)],
            'netlink_send': [('sendmsg$netlink', 0.7)],
            'raw_send': [('sendto$packet', 0.6)],

            # Network configuration (needs netlink)
            'geneve_xmit': [('sendmsg$nl_route_sched', 0.5), ('socket$packet', 0.4)],
            'sch_direct': [('sendmsg$nl_route_sched', 0.6)],
            'qdisc': [('sendmsg$nl_route_sched', 0.5)],
            'tc_modify': [('sendmsg$nl_route_sched', 0.5)],
            'dev_queue_xmit': [('sendto$packet', 0.4)],

            # Socket operations
            'sock_sendmsg': [],  # too generic
            'do_sock_sendmsg': [],
            '__sys_sendto': [('sendto', 0.9), ('sendto$inet6', 0.6), ('sendto$inet', 0.6), ('sendto$packet', 0.6)],
            '__sys_sendmsg': [('sendmsg', 0.9), ('sendmsg$inet6', 0.6), ('sendmsg$inet', 0.6)],
            '__sys_socket': [('socket', 0.9), ('socket$inet6', 0.6), ('socket$inet', 0.6)],
            '__sys_bind': [('bind', 0.9), ('bind$inet6', 0.6), ('bind$inet', 0.6)],
            '__sys_connect': [('connect', 0.9), ('connect$inet6', 0.6), ('connect$inet', 0.6)],
            '__sys_setsockopt': [('setsockopt', 0.9)],
            '__sys_getsockopt': [('getsockopt', 0.9)],
            '__sys_ioctl': [('ioctl', 0.9)],
        }

        for pattern, syscalls in keyword_patterns.items():
            if pattern in func_lower:
                matches.extend(syscalls)

        return matches

    def get_injection_guidance(self, report_text: str) -> Dict:
        """Analyze report and return guidance dict (compatible with StaticAnalyzer interface)."""
        entries = self.analyze_report(report_text)

        # Deduplicate and aggregate weights
        weight_map: Dict[str, float] = {}
        for entry in entries:
            name = entry["name"]
            w = entry["weight"]
            if name not in weight_map or w > weight_map[name]:
                weight_map[name] = w

        return {
            "target_func": "",
            "syscall_entries": entries,
            "syscall_weights": weight_map,
            "analysis_timestamp": "",
        }
