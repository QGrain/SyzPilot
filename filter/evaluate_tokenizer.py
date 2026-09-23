"""
SyzTokenizer Evaluation Benchmark v2

This script provides a systematic set of tokenizer quality evaluation metrics to help users
tune hyperparameters (e.g., repeat_compress_len, max_token_length, vocab_size) when training SyzTokenizer.

Evaluation dimensions:
1. Reconstruction Efficiency
2. Vocabulary Health
3. Syntax-Aware Segmentation — accepts fine-grained tokenization but rewards correct boundary placement
4. Downstream Proxy — requires an external training script

Core design philosophy:
- Does not require complete syscall names (e.g., ioctl$KVM_CREATE) to be a single token.
- Instead, evaluates whether the tokenizer splits syscalls/arguments at "reasonable semantic boundaries"
  (e.g., $, (, ), = {, 0x should ideally fall on token boundaries).

Usage example:
    python evaluate_tokenizer.py \
        --tokenizer /opt/syzpilot/models/customized_tokenizer_224w \
        --test_dataset /artifact/datasets/dataset_224w/programs \
        --max_length 1024 \
        --output report.json
"""

import os
import re
import json
import argparse
import random
import glob
from pathlib import Path
from typing import List, Dict
from collections import Counter

import numpy as np
from transformers import AutoTokenizer, PreTrainedTokenizerFast


# =============================================================================
# 1. Gold Standard Test Snippets & Semantic Subword Sets
# =============================================================================

GOLD_SNIPPETS = [
    "r0 = openat$auto(0xffffffffffffff9c, &(0x7f0000002000)=\"file.txt\", 0x0, 0x0)",
    "sendmsg$inet(r0, &(0x7f0000002980)={0x0, 0x0, &(0x7f00000028c0)=[{&(0x7f0000002780)={0x0}}]}, 0x0)",
    "socket$nl_route(0x10, 0x3, 0x0)",
    "write(r0, &(0x7f0000002000)=\"data\", 0x200)",
    "mmap(0x0, 0x1000, 0x3, 0x22, r0, 0x0)",
    "ioctl$AUTO(r0, 0x8927, &(0x7f0000002000))",
    "syz_emit_ethernet(0x100, &(0x7f0000002000)=@ethernet={@ipv4={@tcp={}}})",
    "r0 = syz_usb_connect$auto(0x0, 0x0, &(0x7f0000002000), 0x0)",
    "openat$dir(0xffffffffffffff9c, &(0x7f0000001000), 0x42, 0x0)",
]

# These are considered acceptable "semantic subwords" — tokenizing syscalls into these subwords is fine
SEMANTIC_SUBWORDS = {
    "openat", "sendmsg", "socket", "write", "mmap", "ioctl", "syz", "emit",
    "ethernet", "usb", "connect", "open", "read", "close", "stat", "lstat",
    "fstat", "auto", "dir", "inet", "nl", "route", "AUTO", "KVM", "CREATE",
    "VM", "IO", "GET", "SET", "FD", "MEM", "LOCK", "UNLOCK", "RDWR", "CREAT",
}

# Punctuation/structural symbols: have clear boundary significance in syz-program syntax
STRUCTURAL_PUNCTS = {
    "=", "$", "(", ")", "{", "}", "[", "]", ",", "&", "0x", '"', ";", "@",
}


def resolve_tokenizer(tokenizer_path: str):
    try:
        return AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception:
        return PreTrainedTokenizerFast.from_pretrained(tokenizer_path)


# =============================================================================
# 2. Evaluation metric implementations
# =============================================================================

def clean_token(tok: str) -> str:
    """Remove GPT2's Ġ prefix and WordPiece's ## prefix to get the raw substring."""
    return tok.replace("Ġ", "").replace("##", "")


def tokenize_to_spans(tokenizer, text: str) -> List[Dict]:
    """
    Split text into tokens and return each token's raw substring with its [start, end) span in the original string.
    """
    tokens = tokenizer.tokenize(text)
    spans = []
    pos = 0
    for tok in tokens:
        raw = clean_token(tok)
        # Find the first position matching raw starting from pos (handles ByteLevel pre-tokenizer space changes)
        idx = text.find(raw, pos)
        if idx == -1:
            # If not found, spaces may have been merged; try skipping leading whitespace and match again
            stripped = raw.lstrip()
            if stripped:
                idx = text.find(stripped, pos)
                if idx != -1:
                    raw = stripped
        if idx == -1:
            # Still not found, conservatively advance pos
            idx = pos
            raw = ""
        start = idx
        end = idx + len(raw)
        spans.append({"token": tok, "raw": raw, "start": start, "end": end})
        pos = end
    return spans


def evaluate_reconstruction_efficiency(tokenizer, texts: List[str], max_length: int = 1024) -> Dict[str, float]:
    lengths = []
    chars = []
    truncated = 0

    for text in texts:
        encoded = tokenizer(text, add_special_tokens=False)
        token_ids = encoded["input_ids"]
        lengths.append(len(token_ids))
        chars.append(len(text))
        if len(token_ids) > max_length:
            truncated += 1

    n = len(texts)
    total_chars = sum(chars)
    total_tokens = sum(lengths)

    return {
        "chars_per_token": round(total_chars / total_tokens, 2) if total_tokens else 0.0,
        "tokens_per_program_mean": round(np.mean(lengths), 2) if lengths else 0.0,
        "tokens_per_program_std": round(np.std(lengths), 2) if lengths else 0.0,
        "truncation_rate_at_1024": round(truncated / n, 4) if n else 0.0,
    }


def evaluate_vocab_health(tokenizer) -> Dict[str, float]:
    vocab = tokenizer.get_vocab()
    tokens = list(vocab.keys())
    lens = [len(t) for t in tokens]

    repetitive = [t for t in tokens if len(t) >= 10 and len(set(t)) == 1]
    vocab_dir = Path(tokenizer.name_or_path) if hasattr(tokenizer, "name_or_path") else None
    file_size_mb = 0.0
    if vocab_dir and vocab_dir.exists():
        for fname in ("vocab.json", "merges.txt", "tokenizer.json"):
            fpath = vocab_dir / fname
            if fpath.exists():
                file_size_mb += fpath.stat().st_size / (1024 * 1024)

    return {
        "vocab_size": len(tokens),
        "max_token_length": max(lens) if lens else 0,
        "avg_token_length": round(sum(lens) / len(lens), 2) if lens else 0.0,
        "repetitive_token_ratio": round(len(repetitive) / len(tokens), 4) if tokens else 0.0,
        "vocab_file_size_mb": round(file_size_mb, 2),
    }


def evaluate_syntax_awareness(tokenizer, snippets: List[str] = None) -> Dict[str, float]:
    """
    Fine-grained syntax-aware evaluation:
    - subword_syscall_recall: does not require the full syscall name as a single token; instead checks whether all sub-tokens are "acceptable semantic subwords".
    - punct_boundary_accuracy: proportion of punctuation symbols ($, (, etc.) that fall on token boundaries.
    - hex_literal_score: whether the 0x prefix is preserved as a single token or split at a reasonable position.
    """
    if snippets is None:
        snippets = GOLD_SNIPPETS

    syscall_re = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*(?:\$[a-zA-Z0-9_]+)*)\s*\(')
    hex_re = re.compile(r'0x[0-9a-fA-F]+')

    subword_hits = 0
    subword_total = 0

    punct_correct = 0
    punct_total = 0

    hex_0x_preserved = 0
    hex_total = 0

    for snippet in snippets:
        spans = tokenize_to_spans(tokenizer, snippet)
        boundaries = {span["start"] for span in spans} | {span["end"] for span in spans}

        # --- Subword Syscall Recall ---
        for m in syscall_re.finditer(snippet):
            name = m.group(1)
            start = m.start(1)
            end = m.end(1)
            # Find all tokens covering this span
            covered = [span for span in spans if not (span["end"] <= start or span["start"] >= end)]
            # Evaluate: whether each sub-token is a semantic subword or pure punctuation (e.g., $, _)
            all_valid = True
            for span in covered:
                raw = span["raw"]
                if raw in SEMANTIC_SUBWORDS:
                    continue
                if raw and all(c in STRUCTURAL_PUNCTS or c == '_' for c in raw):
                    continue
                all_valid = False
                break
            subword_total += 1
            if all_valid:
                subword_hits += 1

        # --- Punctuation Boundary Accuracy ---
        for p in STRUCTURAL_PUNCTS:
            for m in re.finditer(re.escape(p), snippet):
                pos = m.start()
                end_pos = m.end()
                # We consider it best for punctuation to be an independent token, or at least have its start/end on a token boundary
                # Lenient criterion: at least one of p's start or end falls on a token boundary
                if pos in boundaries or end_pos in boundaries:
                    punct_correct += 1
                punct_total += 1

        # --- Hex Literal Score ---
        for m in hex_re.finditer(snippet):
            hex_total += 1
            start = m.start()
            end = m.end()
            # Lenient criterion: any of the following counts as preserved
            # 1. A token starts at start and begins with "0x"
            # 2. "0x" itself is an independent token
            # 3. "0" and "x" are two consecutive tokens covering start and start+1
            preserved = False
            for i, span in enumerate(spans):
                if span["start"] == start and span["raw"].startswith("0x"):
                    preserved = True
                    break
                elif span["raw"] == "0x" and span["start"] == start:
                    preserved = True
                    break
                elif (span["start"] == start and span["raw"] == "0" and
                      i + 1 < len(spans) and spans[i + 1]["start"] == start + 1 and spans[i + 1]["raw"] == "x"):
                    preserved = True
                    break
            if preserved:
                hex_0x_preserved += 1

    return {
        "subword_syscall_recall": round(subword_hits / subword_total, 4) if subword_total else 0.0,
        "punct_boundary_accuracy": round(punct_correct / punct_total, 4) if punct_total else 0.0,
        "hex_literal_score": round(hex_0x_preserved / hex_total, 4) if hex_total else 0.0,
    }


def sample_program_texts(data_dir: str, n: int = 500, seed: int = 42) -> List[str]:
    files = glob.glob(os.path.join(data_dir, "*"))
    files = [f for f in files if os.path.isfile(f)]
    random.seed(seed)
    if len(files) > n:
        files = random.sample(files, n)
    texts = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                texts.append(fp.read())
        except Exception:
            continue
    return texts


def evaluate_tokenizer(tokenizer_path: str,
                       test_dataset_dir: str,
                       max_length: int = 1024,
                       sample_n: int = 500,
                       seed: int = 42) -> Dict:
    tokenizer = resolve_tokenizer(tokenizer_path)

    vocab_health = evaluate_vocab_health(tokenizer)
    syntax_score = evaluate_syntax_awareness(tokenizer)
    texts = sample_program_texts(test_dataset_dir, n=sample_n, seed=seed)
    if not texts:
        raise ValueError(f"No valid program files found in {test_dataset_dir}")
    recon = evaluate_reconstruction_efficiency(tokenizer, texts, max_length=max_length)

    report = {
        "tokenizer_path": str(tokenizer_path),
        "test_dataset_dir": str(test_dataset_dir),
        "max_length": max_length,
        "vocabulary_health": vocab_health,
        "syntax_awareness": syntax_score,
        "reconstruction_efficiency": recon,
    }
    return report


def compute_composite_score(report: Dict) -> float:
    """Compute a heuristic composite score. Note: this score is subjective and cannot replace downstream task validation."""
    vh = report["vocabulary_health"]
    sa = report["syntax_awareness"]
    re = report["reconstruction_efficiency"]
    cpt = re["chars_per_token"]
    cpt_score = max(0.0, 1.0 - abs(cpt - 8.0) / 8.0)
    trunc_score = 1.0 - min(re["truncation_rate_at_1024"], 1.0)
    syn_score = (
        sa["subword_syscall_recall"] * 0.3 +
        sa["punct_boundary_accuracy"] * 0.4 +
        sa["hex_literal_score"] * 0.3
    )
    rep_score = max(0.0, 1.0 - vh["repetitive_token_ratio"] * 100)
    composite = 0.20 * cpt_score + 0.25 * trunc_score + 0.35 * syn_score + 0.20 * rep_score
    return composite


def print_report(report: Dict):
    print("=" * 70)
    print("SyzTokenizer Evaluation Report")
    print("=" * 70)
    print(f"Tokenizer : {report['tokenizer_path']}")
    print(f"Dataset   : {report['test_dataset_dir']}")
    print()

    vh = report["vocabulary_health"]
    print("[Vocabulary Health]")
    print(f"  Vocab Size            : {vh['vocab_size']}")
    print(f"  Max Token Length      : {vh['max_token_length']}")
    print(f"  Avg Token Length      : {vh['avg_token_length']}")
    print(f"  Repetitive Token Ratio: {vh['repetitive_token_ratio']:.4f}")
    print(f"  Vocab File Size (MB)  : {vh['vocab_file_size_mb']:.2f}")
    print()

    sa = report["syntax_awareness"]
    print("[Syntax Awareness]")
    print(f"  Subword Syscall Recall: {sa['subword_syscall_recall']:.4f}")
    print(f"  Punct Boundary Acc    : {sa['punct_boundary_accuracy']:.4f}")
    print(f"  Hex Literal Score     : {sa['hex_literal_score']:.4f}")
    print()

    re = report["reconstruction_efficiency"]
    print("[Reconstruction Efficiency]")
    print(f"  Chars Per Token       : {re['chars_per_token']:.2f}")
    print(f"  Tokens/Program (mean) : {re['tokens_per_program_mean']:.2f}")
    print(f"  Tokens/Program (std)  : {re['tokens_per_program_std']:.2f}")
    print(f"  Truncation Rate @1024 : {re['truncation_rate_at_1024']:.4f}")
    print("=" * 70)

    composite = compute_composite_score(report)
    print(f"[Heuristic Composite Score] : {composite:.4f}")
    print("  WARNING: This score is a rough, uncalibrated heuristic. It should NOT")
    print("  replace downstream validation (e.g., SyzEncoder MLM val loss or")
    print("  TraceClassifier accuracy) as the final quality criterion.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser("Evaluate SyzTokenizer quality")
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--sample_n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    report = evaluate_tokenizer(
        tokenizer_path=args.tokenizer,
        test_dataset_dir=args.test_dataset,
        max_length=args.max_length,
        sample_n=args.sample_n,
        seed=args.seed,
    )

    print_report(report)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
