"""
Automatic hyperparameter search script: SyzTokenizer v2 Hyperparameter Search

Search dimensions:
  - 2 datasets: 224w, 300w
  - 3 sample sizes: 1w, 5w, 10w
  - 2 approaches: base (no special tokens), special (with new_special_tokens)
  - Parameter grid: repeat_compress_len, max_token_length, compress_hex_runs, hex_run_compress_len

All experiments run in parallel on CPU, with final results aggregated into a leaderboard JSON.

Usage example:
    python run_tokenizer_search.py \\
        --data_root /tmp/tokenizer_search_data \\
        --base_model /opt/syzpilot/models/starencoder \\
        --vocab_size 49152 \\
        --output_dir /tmp/tokenizer_search_results \\
        --max_workers 8
"""

import os
import re
import json
import argparse
import subprocess
import concurrent.futures
from pathlib import Path
from itertools import product
from datetime import datetime
from collections import Counter

from evaluate_tokenizer import compute_composite_score


def build_param_grid():
    """Build the parameter search grid."""
    repeat_compress_lens = [16, 32, 64]
    max_token_lengths = [32, 64, 128]
    compress_hex_runs_opts = [True, False]
    hex_run_compress_lens = [8, 16]
    grid = []
    for rcl, mtl, chr_flag, hcl in product(
        repeat_compress_lens,
        max_token_lengths,
        compress_hex_runs_opts,
        hex_run_compress_lens,
    ):
        # If compress_hex_runs=False, hex_run_compress_len is meaningless; keep only one
        if not chr_flag and hcl != 16:
            continue
        grid.append({
            "repeat_compress_len": rcl,
            "max_token_length": mtl,
            "compress_hex_runs": chr_flag,
            "hex_run_compress_len": hcl,
        })
    return grid


def discover_special_tokens(data_root: str, top_n_syscalls: int = 50, top_n_args: int = 20) -> list:
    """
    Read pre-generated high-frequency special tokens list. If /tmp/special_tokens.json exists,
    read it directly; otherwise return a conservative default list.
    """
    prebuilt_path = Path("/tmp/special_tokens.json")
    if prebuilt_path.exists():
        with open(prebuilt_path, "r") as f:
            tokens = json.load(f)
        return tokens[:100]
    # Conservative default list
    return [
        "nl_generic", "syz_io_uring_setup", "unix", "packet", "nl_route",
        "nl80211", "PROG_LOAD_XDP", "inet", "syz_io_uring_submit",
        "socketpair", "BPF_PROG_RAW_TRACEPOINT_LOAD", "MAP_CREATE",
        "fuse", "FUSE", "MAP_CREATE_CONST_STR", "inet6", "l2tp",
        "PROG_LOAD", "syz_clone", "can_j1939", "tipc", "binfmt",
        "x86", "close", "KVM_CREATE_VM", "kvm", "inet_sctp", "ext4",
        "mkdirat", "dir", "openat", "xfs", "write", "btrfs",
        "nl_netfilter", "l2tp6", "truncate", "ioctl", "mmap",
    ]


def run_single_experiment(exp_id: str, dataset: str, sample_size: int,
                          use_special: bool, params: dict, special_tokens: list,
                          base_model: str, vocab_size: int,
                          data_root: str, output_dir: str) -> dict:
    """Run a single train+eval experiment and return result dict."""
    sample_dir = Path(data_root) / dataset / f"sample_{sample_size}"
    exp_out = Path(output_dir) / exp_id
    exp_out.mkdir(parents=True, exist_ok=True)

    tokenizer_out = exp_out / "tokenizer"
    report_path = exp_out / "report.json"

    # 1. Train tokenizer
    train_cmd = [
        "python", "train_tokenizer_v2.py",
        "--dataset", str(sample_dir),
        "--base_model", base_model,
        "--vocab_size", str(vocab_size),
        "--max_token_length", str(params["max_token_length"]),
        "--repeat_compress_len", str(params["repeat_compress_len"]),
        "--hex_run_compress_len", str(params["hex_run_compress_len"]),
        "--batch_size", "5000",
        "--out", str(tokenizer_out),
    ]
    if params["compress_hex_runs"]:
        train_cmd.append("--compress_hex_runs")

    if use_special and special_tokens:
        # Write a JSON file to be read by train_tokenizer_v2, but currently
        # train_tokenizer_v2.py doesn't support reading special tokens from file,
        # so we concatenate them directly on the command line, comma-separated.
        # Ensure train_tokenizer_v2.py supports --special_tokens
        train_cmd.extend(["--special_tokens", ",".join(special_tokens)])

    try:
        with open(exp_out / "train.log", "w") as logf:
            subprocess.run(train_cmd, stdout=logf, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        return {
            "exp_id": exp_id,
            "dataset": dataset,
            "sample_size": sample_size,
            "use_special": use_special,
            "params": params,
            "status": "train_failed",
            "error": str(e),
        }

    # 2. Evaluate tokenizer
    eval_cmd = [
        "python", "evaluate_tokenizer.py",
        "--tokenizer", str(tokenizer_out),
        "--test_dataset", str(sample_dir),
        "--max_length", "1024",
        "--sample_n", "500",
        "--seed", "42",
        "--output", str(report_path),
    ]
    try:
        with open(exp_out / "eval.log", "w") as logf:
            subprocess.run(eval_cmd, stdout=logf, stderr=subprocess.STDOUT, check=True)
    except subprocess.CalledProcessError as e:
        return {
            "exp_id": exp_id,
            "dataset": dataset,
            "sample_size": sample_size,
            "use_special": use_special,
            "params": params,
            "status": "eval_failed",
            "error": str(e),
        }

    # 3. Load report
    try:
        with open(report_path, "r") as f:
            report = json.load(f)
    except Exception as e:
        return {
            "exp_id": exp_id,
            "dataset": dataset,
            "sample_size": sample_size,
            "use_special": use_special,
            "params": params,
            "status": "report_load_failed",
            "error": str(e),
        }

    return {
        "exp_id": exp_id,
        "dataset": dataset,
        "sample_size": sample_size,
        "use_special": use_special,
        "params": params,
        "status": "success",
        "report": report,
    }


def main():
    parser = argparse.ArgumentParser("Hyperparameter search for SyzTokenizer v2")
    parser.add_argument("--data_root", type=str, default="/tmp/tokenizer_search_data",
                        help="Root dir containing 224w/ and 300w/ subdirs with sample_N folders")
    parser.add_argument("--base_model", type=str, default="/opt/syzpilot/models/starencoder")
    parser.add_argument("--vocab_size", type=int, default=49152)
    parser.add_argument("--output_dir", type=str, default="/tmp/tokenizer_search_results")
    parser.add_argument("--max_workers", type=int, default=8,
                        help="Number of parallel training jobs")
    parser.add_argument("--top_n_syscalls", type=int, default=50)
    parser.add_argument("--top_n_args", type=int, default=20)
    parser.add_argument("--skip_special", action="store_true",
                        help="Skip the special-tokens route to save time")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Auto-discover special tokens
    print("Discovering high-frequency special token candidates...")
    special_tokens = discover_special_tokens(args.data_root, args.top_n_syscalls, args.top_n_args)
    print(f"Selected {len(special_tokens)} special tokens: {special_tokens[:20]}{'...' if len(special_tokens)>20 else ''}")
    with open(Path(args.output_dir) / "special_tokens.json", "w") as f:
        json.dump(special_tokens, f, indent=2)

    param_grid = build_param_grid()
    print(f"Parameter grid size: {len(param_grid)}")

    datasets = ["224w", "300w"]
    sample_sizes = [10000, 50000, 100000]
    use_special_flags = [False] if args.skip_special else [False, True]

    experiments = []
    for dataset in datasets:
        for sample_size in sample_sizes:
            for use_special in use_special_flags:
                for params in param_grid:
                    exp_id = f"{dataset}_n{sample_size}_sp{int(use_special)}_rcl{params['repeat_compress_len']}_mtl{params['max_token_length']}_chr{int(params['compress_hex_runs'])}_hcl{params['hex_run_compress_len']}"
                    experiments.append((exp_id, dataset, sample_size, use_special, params))

    print(f"Total experiments: {len(experiments)}")
    print(f"Running with up to {args.max_workers} parallel workers...")

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_exp = {}
        for exp_id, dataset, sample_size, use_special, params in experiments:
            future = executor.submit(
                run_single_experiment,
                exp_id, dataset, sample_size,
                use_special, params,
                special_tokens if use_special else [],
                args.base_model, args.vocab_size,
                args.data_root, args.output_dir,
            )
            future_to_exp[future] = exp_id

        for future in concurrent.futures.as_completed(future_to_exp):
            exp_id = future_to_exp[future]
            try:
                res = future.result()
                results.append(res)
                score_str = "N/A"
                if res["status"] == "success":
                    score = compute_composite_score(res["report"])
                    score_str = f"{score:.4f}"
                print(f"[{len(results)}/{len(experiments)}] {exp_id} -> {res['status']} (score={score_str})")
            except Exception as exc:
                print(f"[{len(results)+1}/{len(experiments)}] {exp_id} -> EXCEPTION: {exc}")
                results.append({
                    "exp_id": exp_id,
                    "status": "exception",
                    "error": str(exc),
                })

    # Aggregate leaderboard
    leaderboard = []
    for res in results:
        if res.get("status") != "success":
            continue
        composite = compute_composite_score(res["report"])
        leaderboard.append({
            "exp_id": res["exp_id"],
            "dataset": res["dataset"],
            "sample_size": res["sample_size"],
            "use_special": res["use_special"],
            "params": res["params"],
            "composite_score": round(composite, 4),
            "vocab_file_size_mb": res["report"]["vocabulary_health"]["vocab_file_size_mb"],
            "avg_token_length": res["report"]["vocabulary_health"]["avg_token_length"],
            "chars_per_token": res["report"]["reconstruction_efficiency"]["chars_per_token"],
            "truncation_rate": res["report"]["reconstruction_efficiency"]["truncation_rate_at_1024"],
            "subword_syscall_recall": res["report"]["syntax_awareness"]["subword_syscall_recall"],
            "punct_boundary_accuracy": res["report"]["syntax_awareness"]["punct_boundary_accuracy"],
            "hex_literal_score": res["report"]["syntax_awareness"]["hex_literal_score"],
        })

    leaderboard.sort(key=lambda x: x["composite_score"], reverse=True)

    # Save full results
    summary_path = Path(args.output_dir) / "leaderboard.json"
    with open(summary_path, "w") as f:
        json.dump({
            "special_tokens": special_tokens,
            "total_experiments": len(experiments),
            "successful": len(leaderboard),
            "leaderboard": leaderboard,
        }, f, indent=2)

    print(f"\nLeaderboard saved to {summary_path}")
    print("\nTop 10 configs across all settings:")
    for i, entry in enumerate(leaderboard[:10], 1):
        print(f"{i}. [{entry['composite_score']:.4f}] {entry['exp_id']}  "
              f"CPT={entry['chars_per_token']:.2f} "
              f"Trunc={entry['truncation_rate']:.4f} "
              f"PunctAcc={entry['punct_boundary_accuracy']:.4f}")


if __name__ == "__main__":
    main()
