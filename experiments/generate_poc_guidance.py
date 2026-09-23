#!/usr/bin/env python3
"""
Generate PoC-based guidance JSON for case_1 (kernel BUG in validate_xmit_skb).

Extracts the syscall sequence from case_1.poc.syz and generates a GuidancePayload
that can be injected into the fuzzer via POST /guidance.

Usage:
    python generate_poc_guidance.py --output case_1_guidance.json
    python generate_poc_guidance.py --fuzzer_addr localhost:12630  # send directly
"""

import argparse
import json
import sys

# case_1 PoC syscall sequence (in execution order)
POC_SYSCALLS = [
    "socket$packet",
    "setsockopt$packet_int",
    "socket",                       # netlink socket (generic)
    "socket",                       # netlink socket (generic)
    "sendmsg$NBD_CMD_DISCONNECT",   # netlink operation
    "getsockname$packet",
    "sendmsg$nl_route_sched",       # setup qdisc
    "bind$packet",
    "sendto$inet6",                 # trigger: send crafted IPv6 packet
]

# Syscall weights — higher = more likely to be selected in guided mode
# Weighted by importance to the PoC execution path
SYSCALL_WEIGHTS = {
    # Core PoC syscalls (must appear)
    "socket$packet":           10.0,
    "setsockopt$packet_int":    8.0,
    "getsockname$packet":       8.0,
    "bind$packet":              9.0,
    "sendto$inet6":             9.0,
    "sendmsg$nl_route_sched":   8.0,
    "sendmsg$NBD_CMD_DISCONNECT": 5.0,

    # Supporting syscalls (create socket infrastructure)
    "socket":                   5.0,
    "socket$inet6":             4.0,
    "socket$netlink":           5.0,

    # Related syscalls that may help reach the target
    "sendto$packet":            6.0,
    "sendmsg$packet":           5.0,
    "sendmsg$netlink":          5.0,
    "setsockopt$inet_tcp":      3.0,
    "setsockopt$inet_udp":      3.0,
    "ioctl$sock_SIOCGIFINDEX":  4.0,
}

# Argument constraints extracted from case_1 PoC.
# These guide the fuzzer to generate syscalls with correct domain/type/protocol values.
ARG_HINTS_SOCKET_PACKET = [
    {"syscall": "socket$packet", "arg_idx": 0, "value": 0x11},  # domain = AF_PACKET
    {"syscall": "socket$packet", "arg_idx": 1, "value": 0x3},   # type = SOCK_RAW
    {"syscall": "socket$packet", "arg_idx": 2, "value": 0x300}, # protocol = ETH_P_ALL
]
ARG_HINTS_SOCKET_NETLINK = [
    {"syscall": "socket", "arg_idx": 0, "value": 0x10},  # domain = AF_NETLINK
    {"syscall": "socket", "arg_idx": 1, "value": 0x3},   # type = SOCK_RAW
    {"syscall": "socket", "arg_idx": 2, "value": 0x0},   # protocol = NETLINK_ROUTE
]

# Mutation templates at multiple granularities
MUTATION_TEMPLATES = [
    # Template 1: Socket creation + configuration prefix
    {
        "type": "prefix",
        "syscalls": ["socket$packet", "setsockopt$packet_int"],
        "priority": 0.9,
        "insert_mode": "prefix",
        "arg_hints": ARG_HINTS_SOCKET_PACKET,
    },
    # Template 2: Core PoC sequence (socket → bind → send)
    {
        "type": "sequence",
        "syscalls": ["socket$packet", "bind$packet", "sendto$inet6"],
        "priority": 0.85,
        "insert_mode": "prefix",
        "arg_hints": ARG_HINTS_SOCKET_PACKET,
    },
    # Template 3: Netlink setup + qdisc configuration
    {
        "type": "sequence",
        "syscalls": ["socket", "sendmsg$nl_route_sched"],
        "priority": 0.7,
        "insert_mode": "prefix",
        "arg_hints": ARG_HINTS_SOCKET_NETLINK,
    },
    # Template 4: Interface discovery sequence
    {
        "type": "sequence",
        "syscalls": ["socket", "getsockname$packet"],
        "priority": 0.7,
        "insert_mode": "prefix",
        "arg_hints": ARG_HINTS_SOCKET_NETLINK,
    },
    # Template 5: Full PoC prefix (first 5 syscalls)
    {
        "type": "prefix",
        "syscalls": ["socket$packet", "setsockopt$packet_int", "socket", "socket",
                     "sendmsg$NBD_CMD_DISCONNECT"],
        "priority": 0.6,
        "insert_mode": "prefix",
    },
    # Template 6: Complete PoC sequence (all 9 syscalls)
    {
        "type": "sequence",
        "syscalls": POC_SYSCALLS,
        "priority": 0.4,
        "insert_mode": "prefix",
    },
    # Template 7: Late-stage prefix (bind + send)
    {
        "type": "sequence",
        "syscalls": ["bind$packet", "sendto$inet6"],
        "priority": 0.8,
        "insert_mode": "replace",
    },
]


def generate_guidance_payload():
    """Generate the complete GuidancePayload for case_1 PoC."""
    return {
        "version": 1,
        "timestamp": "",
        "syscall_weights": SYSCALL_WEIGHTS,
        "mutation_templates": MUTATION_TEMPLATES,
        "generation_hints": {
            "preferred_syscalls": [
                "socket$packet", "setsockopt$packet_int", "bind$packet",
                "sendto$inet6", "sendmsg$nl_route_sched", "getsockname$packet",
            ],
            "preferred_ratio": 0.3,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate PoC guidance for case_1")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--fuzzer_addr", help="Send directly to fuzzer (e.g., localhost:12630)")
    args = parser.parse_args()

    payload = generate_guidance_payload()

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"Guidance payload written to {args.output}")
        print(f"  {len(payload['syscall_weights'])} syscall weights")
        print(f"  {len(payload['mutation_templates'])} mutation templates")

    if args.fuzzer_addr:
        import urllib.request
        url = f"http://{args.fuzzer_addr}/guidance"
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"Sent to {url}: {resp.status} {resp.read().decode()}")
        except Exception as e:
            print(f"Failed to send to {url}: {e}", file=sys.stderr)
            sys.exit(1)

    if not args.output and not args.fuzzer_addr:
        # Default: print to stdout
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
