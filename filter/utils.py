import os
import string
from collections import OrderedDict
from random import choice

import torch


def check_dir(path):
    if os.path.isdir(path):
        return True
    try:
        os.makedirs(path)
        return True
    except OSError:
        return False


class CachedTokenizingCollator:
    """Tokenize each distinct program at most once within a trainer process."""

    def __init__(self, tokenizer, max_length: int, cache_entries: int = 0):
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if cache_entries < 0:
            raise ValueError("cache_entries must not be negative")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.cache_entries = int(cache_entries)
        self._cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.in_batch_reuses = 0

    def __call__(self, d_list):
        progs, labels = zip(*d_list)
        resolved = {}
        missing = []
        missing_seen = set()
        if self.cache_entries:
            for prog in progs:
                cached = self._cache.get(prog)
                if cached is not None:
                    self.hits += 1
                    self._cache.move_to_end(prog)
                    resolved[prog] = cached
                elif prog not in missing_seen:
                    missing_seen.add(prog)
                    missing.append(prog)
                else:
                    self.in_batch_reuses += 1
        else:
            missing = list(progs)

        if missing:
            tokenized = self.tokenizer(
                missing,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )
            self.misses += len(missing)
            if self.cache_entries:
                for index, prog in enumerate(missing):
                    cached = (
                        tokenized["input_ids"][index].to(torch.int32).clone(),
                        tokenized["attention_mask"][index].to(torch.bool).clone(),
                    )
                    resolved[prog] = cached
                    self._cache[prog] = cached
                    self._cache.move_to_end(prog)
                    while len(self._cache) > self.cache_entries:
                        self._cache.popitem(last=False)
            else:
                return (
                    tokenized["input_ids"],
                    tokenized["attention_mask"],
                    torch.stack(labels),
                )

        input_ids = torch.stack([resolved[prog][0] for prog in progs]).long()
        attention_mask = torch.stack(
            [resolved[prog][1] for prog in progs]
        ).long()
        return input_ids, attention_mask, torch.stack(labels)

    def cache_info(self):
        return {
            "capacity": self.cache_entries,
            "size": len(self._cache),
            "hits": self.hits,
            "misses": self.misses,
            "in_batch_reuses": self.in_batch_reuses,
        }


def create_collate_fn(tokenizer, max_length: int = 1024,
                      cache_entries: int = 0):
    max_length = tokenizer.model_max_length if max_length is None else max_length
    return CachedTokenizingCollator(
        tokenizer, max_length=max_length, cache_entries=cache_entries
    )


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = (
        attention_mask.unsqueeze(-1)
        .expand(token_embeddings.size())
        .to(dtype=token_embeddings.dtype)
    )
    return torch.sum(
        token_embeddings * input_mask_expanded, 1
    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def rand_str(length):
    return ''.join(choice(string.printable) for _ in range(length))
