"""
PoCPatternAnalyzer: Extract structural patterns from PoC syz-programs.

Analyzes the PoC program to identify required syscalls, key arguments,
and sequence structure. Generates mutation templates for the fuzzer.

Note: In real scenarios, we don't have the PoC. This module is used as
an oracle baseline for experimental evaluation.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SyscallInfo:
    """Information about a syscall in the PoC."""
    name: str
    position: int  # order in the program
    arguments: Dict[str, str] = field(default_factory=dict)
    is_required: bool = True


def extract_syscalls_ordered(prog_text: str) -> List[SyscallInfo]:
    """Extract ordered syscall list from a syz-program."""
    syscalls = []
    for i, line in enumerate(prog_text.split('\n')):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '(' not in line:
            continue

        # Extract syscall name
        if '=' in line:
            # r0 = socket$packet(...)
            sc_part = line.split('=')[1].strip()
        else:
            sc_part = line.strip()

        sc_name = sc_part.split('(')[0].strip()
        if sc_name:
            syscalls.append(SyscallInfo(
                name=sc_name,
                position=i,
            ))

    return syscalls


class PoCPatternAnalyzer:
    """Analyze PoC syz-program to extract structural patterns."""

    def __init__(self, poc_path: str):
        with open(poc_path) as f:
            self.poc_text = f.read()
        self.poc_syscalls = extract_syscalls_ordered(self.poc_text)

    def extract_pattern(self) -> Dict:
        """Extract mutation-friendly pattern from PoC."""
        required_syscalls = [sc.name for sc in self.poc_syscalls]

        # Build sequence template
        sequence_template = []
        for i, sc in enumerate(self.poc_syscalls):
            position = "prefix" if i == 0 else f"after:{self.poc_syscalls[i-1].name}"
            sequence_template.append({
                "syscall": sc.name,
                "position": position,
                "required": True,
            })

        return {
            "required_syscalls": required_syscalls,
            "sequence_template": sequence_template,
            "num_syscalls": len(required_syscalls),
        }

    def generate_mutation_templates(self) -> List[Dict]:
        """Generate partial templates for fuzzer mutation.

        Templates at multiple granularities:
        1. Full sequence (rarely useful directly)
        2. Prefix templates: first N syscalls
        3. Key syscall pairs
        """
        templates = []

        syscalls = [sc.name for sc in self.poc_syscalls]
        if not syscalls:
            return templates

        # Full sequence template (low priority - too specific)
        templates.append({
            "type": "sequence",
            "syscalls": syscalls,
            "priority": 0.3,
            "insert_mode": "prefix",
        })

        # Prefix templates (higher priority - more flexible)
        for length in range(2, min(len(syscalls) + 1, 6)):
            templates.append({
                "type": "prefix",
                "syscalls": syscalls[:length],
                "priority": 0.5 + 0.1 * (length - 2),
                "insert_mode": "prefix",
            })

        # Key syscall pairs (medium priority)
        # Socket creation + first operation is often critical
        if len(syscalls) >= 2:
            templates.append({
                "type": "sequence",
                "syscalls": syscalls[:2],
                "priority": 0.7,
                "insert_mode": "prefix",
            })

        # Sort by priority
        templates.sort(key=lambda x: x["priority"], reverse=True)

        logger.info(f"[PoCAnalyzer] Generated {len(templates)} templates from PoC")
        return templates

    def get_required_syscalls(self) -> List[str]:
        """Get list of required syscalls from the PoC."""
        return [sc.name for sc in self.poc_syscalls]

    def get_socket_types(self) -> Dict[str, str]:
        """Extract socket type information from PoC.

        Returns:
            {syscall_name: socket_type_info}
        """
        socket_info = {}
        for sc in self.poc_syscalls:
            if 'socket' in sc.name.lower():
                # Try to extract AF_* and SOCK_* from the program
                # This is a simplified extraction
                socket_info[sc.name] = "raw"  # placeholder
        return socket_info
