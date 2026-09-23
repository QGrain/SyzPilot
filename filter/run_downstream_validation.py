"""
Downstream validation pipeline for SyzTokenizer v2.

Trains 4 candidate tokenizers and runs small-scale MLM pretraining on each,
then compares MLM val loss to determine the best tokenizer configuration.

Candidates:
  A: 224w + best params  (rcl=32, mtl=64, chr=1, hcl=8)
  B: 224w + worst params (rcl=32, mtl=64, chr=1, hcl=16)
  C: 300w + best params  (rcl=32, mtl=64, chr=1, hcl=8)
  D: 300w + worst params (rcl=16, mtl=64, chr=0, hcl=16)

All experiments run sequentially to avoid GPU contention.
"""

import os
import re
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

EXPERIMENTS = [
    {
        "name": "A_224w_best",
        "dataset": "/artifact/datasets/dataset_224w/programs",
        "tokenizer_params": {
            "repeat_compress_len": 32,
            "max_token_length": 64,
            "compress_hex_runs": True,
            "hex_run_compress_len": 8,
        },
    },
    {
        "name": "B_224w_worst",
        "dataset": "/artifact/datasets/dataset_224w/programs",
        "tokenizer_params": {
            "repeat_compress_len": 32,
            "max_token_length": 64,
            "compress_hex_runs": True,
            "hex_run_compress_len": 16,
        },
    },
    {
        "name": "C_300w_best",
        "dataset": "/artifact/datasets/prog_dataset_300w",
        "tokenizer_params": {
            "repeat_compress_len": 32,
            "max_token_length": 64,
            "compress_hex_runs": True,
            "hex_run_compress_len": 8,
        },
    },
    {
        "name": "D_300w_worst",
        "dataset": "/artifact/datasets/prog_dataset_300w",
        "tokenizer_params": {
            "repeat_compress_len": 16,
            "max_token_length": 64,
            "compress_hex_runs": False,
            "hex_run_compress_len": 16,
        },
    },
]

BASE_MODEL = "/opt/syzpilot/models/starencoder"
VOCAB_SIZE = 49152
PRETRAIN_CONFIG = {
    "epochs": 2,
    "batch_size": 16,       # reduced from 32 to avoid OOM during eval on 2xA800
    "lr": 2e-5,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "mlm_probability": 0.15,
    "save_interval": 500,
    "test_steps": 50,
    "gradient_acc_step": 1,
    "small_test": 0.01,
}


def load_existing_results(root):
    summary_path = Path(root) / "summary.json"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            return json.load(f)
    return []


def _parse_val_losses_from_log(log_path):
    """Parse all Val Loss records from pretrain log"""
    val_losses = []
    if not Path(log_path).exists():
        return val_losses
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"Val Loss = ([\d.]+)", line)
            if m:
                val_losses.append(float(m.group(1)))
    return val_losses


def _has_valid_pretrain_output(pretrain_out):
    """Check if pretrain output directory contains valid results (best_model or final_syzencoder)"""
    pretrain_out = Path(pretrain_out)
    return (pretrain_out / "best_model").exists() or (pretrain_out / "final_syzencoder").exists()


def run_tokenizer_train(name, dataset, params, out_root):
    out_dir = Path(out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_out = out_dir / "tokenizer"

    # skip if tokenizer already exists and looks valid
    if (Path(tokenizer_out) / "tokenizer.json").exists():
        print(f"  [TrainTokenizer] {name} -> tokenizer already exists, skipping")
        return str(tokenizer_out)

    cmd = [
        "python", "train_tokenizer_v2.py",
        "--dataset", dataset,
        "--base_model", BASE_MODEL,
        "--vocab_size", str(VOCAB_SIZE),
        "--max_token_length", str(params["max_token_length"]),
        "--repeat_compress_len", str(params["repeat_compress_len"]),
        "--hex_run_compress_len", str(params["hex_run_compress_len"]),
        "--batch_size", "5000",
        "--out", str(tokenizer_out),
    ]
    if params["compress_hex_runs"]:
        cmd.append("--compress_hex_runs")

    log_path = out_dir / "tokenizer_train.log"
    print(f"  [TrainTokenizer] {name} -> {tokenizer_out}")
    with open(log_path, "w") as logf:
        subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, check=True)
    return str(tokenizer_out)


def run_pretrain(name, dataset, tokenizer_path, out_root):
    out_dir = Path(out_root) / name
    pretrain_out = out_dir / "syzencoder"
    pretrain_out.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "pretrain.log"

    # 1. If valid pretrain results already exist, skip (avoid redundant execution)
    existing_losses = _parse_val_losses_from_log(log_path)
    if existing_losses and _has_valid_pretrain_output(pretrain_out):
        print(f"  [Pretrain] {name} -> valid output already exists ({len(existing_losses)} val losses), skipping")
        return existing_losses

    # 2. If old invalid log exists, back it up and retrain
    if log_path.exists():
        backup_path = out_dir / f"pretrain.log.old.{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        shutil.move(str(log_path), str(backup_path))
        print(f"  [Pretrain] {name} -> backed up old log to {backup_path}")

    cmd = [
        "accelerate", "launch", "pretrain_encoder_mlm.py",
        "--prog_dataset", dataset,
        "--base_model", BASE_MODEL,
        "--tokenizer", tokenizer_path,
        "--output_dir", str(pretrain_out),
        "--log_dir", str(pretrain_out / "logs"),
        "--epochs", str(PRETRAIN_CONFIG["epochs"]),
        "--batch_size", str(PRETRAIN_CONFIG["batch_size"]),
        "--lr", str(PRETRAIN_CONFIG["lr"]),
        "--weight_decay", str(PRETRAIN_CONFIG["weight_decay"]),
        "--max_grad_norm", str(PRETRAIN_CONFIG["max_grad_norm"]),
        "--mlm_probability", str(PRETRAIN_CONFIG["mlm_probability"]),
        "--save_interval", str(PRETRAIN_CONFIG["save_interval"]),
        "--test_steps", str(PRETRAIN_CONFIG["test_steps"]),
        "--gradient_acc_step", str(PRETRAIN_CONFIG["gradient_acc_step"]),
        "--small_test", str(PRETRAIN_CONFIG["small_test"]),
        "--disable_wandb", "True",
    ]

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"  [Pretrain] {name} -> {pretrain_out}")
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env)

    # 3. Regardless of exit code, check result validity first
    val_losses = _parse_val_losses_from_log(log_path)

    if val_losses and _has_valid_pretrain_output(pretrain_out):
        # Even if the process exits with non-zero code (e.g., SIGABRT in group C), accept if results are valid
        if result.returncode != 0:
            print(f"  [Pretrain] {name} -> WARNING: process exited with code {result.returncode}, but valid output detected. Continuing.")
        return val_losses

    # 4. Results invalid and process failed, raise exception
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)

    return val_losses


def main():
    root = os.path.expanduser("~/tmp/downstream_validation")
    Path(root).mkdir(parents=True, exist_ok=True)

    # Save experiment plan
    with open(Path(root) / "plan.json", "w") as f:
        json.dump({
            "experiments": EXPERIMENTS,
            "pretrain_config": PRETRAIN_CONFIG,
        }, f, indent=2)

    results = load_existing_results(root)
    completed_names = {r["name"] for r in results}

    for exp in EXPERIMENTS:
        if exp["name"] in completed_names:
            print(f"\n{'='*70}")
            print(f"Experiment: {exp['name']} -> ALREADY COMPLETED, skipping")
            print(f"{'='*70}")
            continue

        print(f"\n{'='*70}")
        print(f"Experiment: {exp['name']}")
        print(f"{'='*70}")

        print("Step 1: Training tokenizer...")
        tokenizer_path = run_tokenizer_train(exp["name"], exp["dataset"], exp["tokenizer_params"], root)

        print("Step 2: Running MLM pretraining...")
        val_losses = run_pretrain(exp["name"], exp["dataset"], tokenizer_path, root)
        print(f"  Val losses: {val_losses}")

        results.append({
            "name": exp["name"],
            "dataset": exp["dataset"],
            "tokenizer_params": exp["tokenizer_params"],
            "tokenizer_path": tokenizer_path,
            "val_losses": val_losses,
            "final_val_loss": val_losses[-1] if val_losses else None,
            "min_val_loss": min(val_losses) if val_losses else None,
        })

        # Save intermediate summary after each experiment
        with open(Path(root) / "summary.json", "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*70}")
    for r in results:
        fv = f"{r['final_val_loss']:.4f}" if r["final_val_loss"] is not None else "N/A"
        mv = f"{r['min_val_loss']:.4f}" if r["min_val_loss"] is not None else "N/A"
        print(f"  {r['name']}: final_val_loss={fv}  min_val_loss={mv}")

    with open(Path(root) / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed logs and outputs are in: {root}")


if __name__ == "__main__":
    main()
