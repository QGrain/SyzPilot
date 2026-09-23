import os
import argparse
import re
import json
import time
from pathlib import Path
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from multiprocessing import Pool, cpu_count


def normalize_prog_text(text: str, repeat_compress_len: int = 32,
                        compress_hex_runs: bool = True,
                        hex_run_compress_len: int = 16) -> str:
    """
    Clean syz-program text by removing dirty data that causes BPE to produce garbage ultra-long tokens.
    """
    text = re.sub(
        r'(.)\1{' + str(repeat_compress_len) + r',}',
        lambda m: m.group(1) * repeat_compress_len,
        text
    )

    if compress_hex_runs:
        text = re.sub(
            r'(?<![0-9a-fA-FxX])([0-9a-fA-F]{' + str(hex_run_compress_len) + r',})(?![0-9a-fA-F])',
            lambda m: m.group(1)[:hex_run_compress_len],
            text
        )

    return text


def _worker_init(repeat_compress_len, compress_hex_runs, hex_run_compress_len):
    global _WORKER_ARGS
    _WORKER_ARGS = (repeat_compress_len, compress_hex_runs, hex_run_compress_len)


def _worker_process(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        args = _WORKER_ARGS
        normalized = normalize_prog_text(
            text,
            repeat_compress_len=args[0],
            compress_hex_runs=args[1],
            hex_run_compress_len=args[2]
        )
        return normalized
    except Exception:
        return None


def batch_iterator_v3(data_dir, batch_size=10000,
                      repeat_compress_len=32,
                      compress_hex_runs=True,
                      hex_run_compress_len=16,
                      workers=32):
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

    start_time = time.time()
    texts = []
    total_processed = 0

    # chunksize=100 reduces IPC overhead for ~3M small files
    with Pool(processes=workers,
              initializer=_worker_init,
              initargs=(repeat_compress_len, compress_hex_runs, hex_run_compress_len)) as pool:
        for content in tqdm(pool.imap_unordered(_worker_process, files, chunksize=100),
                            total=len(files),
                            desc="Processing program files (multiprocess)",
                            unit="file"):
            if content is not None:
                texts.append(content)
                total_processed += 1
            if len(texts) >= batch_size:
                yield texts
                texts = []

    if texts:
        yield texts

    elapsed = time.time() - start_time
    print(f"[v3] File reading + normalization completed: {total_processed} files "
          f"in {elapsed:.2f}s ({total_processed/elapsed:.1f} files/s)")


def resolve_prog_dir(dataset_dir: str) -> str:
    dataset_path = Path(dataset_dir)
    prog_subdir = dataset_path / "programs"
    if prog_subdir.is_dir():
        return str(prog_subdir)
    return str(dataset_path)


def main():
    parser = argparse.ArgumentParser('Train SyzTokenizer v3 with multiprocessing optimization')
    parser.add_argument('--dataset', type=str, default='/artifact/datasets/prog_dataset_300w/')
    parser.add_argument('--base_model', type=str, default='/opt/syzpilot/models/starencoder')
    parser.add_argument('--vocab_size', type=int, default=49152)
    parser.add_argument('--max_token_length', type=int, default=64)
    parser.add_argument('--min_frequency', type=int, default=None)
    parser.add_argument('--repeat_compress_len', type=int, default=32)
    parser.add_argument('--compress_hex_runs', action='store_true', default=False,
                        help='compress long hex-digit runs (must explicitly pass flag to enable)')
    parser.add_argument('--hex_run_compress_len', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=10000)
    parser.add_argument('--out', type=str, default='/opt/syzpilot/models/customized_tokenizer_v3')
    parser.add_argument('--length', type=int, default=None)
    parser.add_argument('--special_tokens', type=str, default='')
    parser.add_argument('--workers', type=int, default=32,
                        help='number of multiprocessing workers for file reading/normalization')
    args = parser.parse_args()

    prog_dir = resolve_prog_dir(args.dataset)
    print(f"Resolved program directory: {prog_dir}")

    try:
        tokenizer = GPT2TokenizerFast.from_pretrained(args.base_model)

        print(f"Training tokenizer with dataset {prog_dir}")
        print(f"  vocab_size={args.vocab_size}, max_token_length={args.max_token_length}")
        print(f"  repeat_compress_len={args.repeat_compress_len}, compress_hex_runs={args.compress_hex_runs}")
        print(f"  workers={args.workers}")

        train_kwargs = {
            "vocab_size": args.vocab_size,
            "max_token_length": args.max_token_length,
        }
        if args.min_frequency is not None:
            train_kwargs["min_frequency"] = args.min_frequency

        new_special_tokens = [t.strip() for t in args.special_tokens.split(",") if t.strip()] if args.special_tokens else None
        if new_special_tokens:
            print(f"  Adding {len(new_special_tokens)} special tokens")

        overall_start = time.time()

        new_tokenizer = tokenizer.train_new_from_iterator(
            text_iterator=batch_iterator_v3(
                prog_dir,
                batch_size=args.batch_size,
                repeat_compress_len=args.repeat_compress_len,
                compress_hex_runs=args.compress_hex_runs,
                hex_run_compress_len=args.hex_run_compress_len,
                workers=args.workers,
            ),
            length=args.length,
            new_special_tokens=new_special_tokens,
            **train_kwargs
        )

        out_path = Path(args.out)
        out_path.mkdir(parents=True, exist_ok=True)
        new_tokenizer.save_pretrained(out_path)

        train_config = {
            "dataset": str(prog_dir),
            "base_model": args.base_model,
            "vocab_size": args.vocab_size,
            "max_token_length": args.max_token_length,
            "min_frequency": args.min_frequency,
            "repeat_compress_len": args.repeat_compress_len,
            "compress_hex_runs": args.compress_hex_runs,
            "hex_run_compress_len": args.hex_run_compress_len,
            "special_tokens": new_special_tokens or [],
            "workers": args.workers,
        }
        with open(out_path / "tokenizer_train_config.json", "w") as f:
            json.dump(train_config, f, indent=2)

        total_elapsed = time.time() - overall_start
        print(f"[v3] Total tokenizer training time: {total_elapsed:.2f}s")
        print(f"New tokenizer saved to {out_path}")

    except KeyboardInterrupt:
        print("Training interrupted by user.")


if __name__ == "__main__":
    main()
