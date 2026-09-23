"""
Smoke test for run_tokenizer_search.py
Only runs a tiny subset to verify the end-to-end pipeline.
"""
import os
import sys
import subprocess
from pathlib import Path

out_dir = "/tmp/tokenizer_search_smoke"
os.makedirs(out_dir, exist_ok=True)

cmds = [
    # base route, 2 param combos
    "python train_tokenizer_v2.py --dataset /tmp/tokenizer_search_data/224w/sample_10000 --base_model /opt/syzpilot/models/starencoder --vocab_size 49152 --max_token_length 64 --repeat_compress_len 32 --compress_hex_runs --hex_run_compress_len 16 --batch_size 5000 --out /tmp/tokenizer_search_smoke/tok_base1",
    "python evaluate_tokenizer.py --tokenizer /tmp/tokenizer_search_smoke/tok_base1 --test_dataset /tmp/tokenizer_search_data/224w/sample_10000 --output /tmp/tokenizer_search_smoke/report_base1.json",
    # special route, 2 param combos
    "python train_tokenizer_v2.py --dataset /tmp/tokenizer_search_data/224w/sample_10000 --base_model /opt/syzpilot/models/starencoder --vocab_size 49152 --max_token_length 64 --repeat_compress_len 32 --compress_hex_runs --hex_run_compress_len 16 --batch_size 5000 --special_tokens 'openat\$auto,write,mmap,nl_generic' --out /tmp/tokenizer_search_smoke/tok_sp1",
    "python evaluate_tokenizer.py --tokenizer /tmp/tokenizer_search_smoke/tok_sp1 --test_dataset /tmp/tokenizer_search_data/224w/sample_10000 --output /tmp/tokenizer_search_smoke/report_sp1.json",
]

for cmd in cmds:
    print(f"Running: {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"FAILED: {cmd}")
        sys.exit(1)

print("Smoke test passed!")
