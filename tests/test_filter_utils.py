"""Regression tests for exact, bounded training tokenization caching."""

import unittest

import torch

from filter.utils import create_collate_fn, mean_pooling


class FakeTokenizer:
    model_max_length = 10**30

    def __init__(self):
        self.calls = []

    def __call__(self, programs, **kwargs):
        self.calls.append(list(programs))
        max_length = kwargs["max_length"]
        rows = []
        masks = []
        for program in programs:
            values = [ord(char) % 31 + 1 for char in program][:max_length]
            mask = [1] * len(values)
            values.extend([0] * (max_length - len(values)))
            mask.extend([0] * (max_length - len(mask)))
            rows.append(values)
            masks.append(mask)
        return {
            "input_ids": torch.tensor(rows, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


def sample(program, label):
    return program, torch.tensor(label, dtype=torch.long)


class CachedTokenizingCollatorTest(unittest.TestCase):
    def test_overlapping_batches_tokenize_only_new_programs(self):
        tokenizer = FakeTokenizer()
        collator = create_collate_fn(tokenizer, max_length=8, cache_entries=4)

        first = collator([sample("alpha", 0), sample("beta", 1)])
        second = collator([sample("beta", 1), sample("gamma", 2)])

        self.assertEqual(tokenizer.calls, [["alpha", "beta"], ["gamma"]])
        self.assertTrue(torch.equal(first[0][1], second[0][0]))
        self.assertTrue(torch.equal(first[1][1], second[1][0]))
        self.assertEqual(
            collator.cache_info(),
            {
                "capacity": 4, "size": 3, "hits": 1, "misses": 3,
                "in_batch_reuses": 0,
            },
        )

    def test_cached_output_is_identical_to_uncached_output(self):
        batch = [sample("alpha", 0), sample("beta", 1), sample("alpha", 2)]
        cached = create_collate_fn(
            FakeTokenizer(), max_length=8, cache_entries=2
        )(batch)
        uncached = create_collate_fn(
            FakeTokenizer(), max_length=8, cache_entries=0
        )(batch)

        for cached_tensor, uncached_tensor in zip(cached, uncached):
            self.assertTrue(torch.equal(cached_tensor, uncached_tensor))

    def test_capacity_smaller_than_batch_still_resolves_evicted_entries(self):
        collator = create_collate_fn(
            FakeTokenizer(), max_length=8, cache_entries=1
        )
        input_ids, attention_mask, labels = collator([
            sample("alpha", 0), sample("beta", 1), sample("gamma", 2),
        ])

        self.assertEqual(tuple(input_ids.shape), (3, 8))
        self.assertEqual(tuple(attention_mask.shape), (3, 8))
        self.assertEqual(labels.tolist(), [0, 1, 2])
        self.assertEqual(collator.cache_info()["size"], 1)

    def test_same_batch_duplicates_are_counted_as_local_reuses(self):
        tokenizer = FakeTokenizer()
        collator = create_collate_fn(tokenizer, max_length=8, cache_entries=4)

        collator([sample("alpha", 0), sample("alpha", 1)])

        self.assertEqual(tokenizer.calls, [["alpha"]])
        self.assertEqual(collator.cache_info()["misses"], 1)
        self.assertEqual(collator.cache_info()["in_batch_reuses"], 1)

    def test_negative_capacity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be negative"):
            create_collate_fn(FakeTokenizer(), cache_entries=-1)


class MeanPoolingTest(unittest.TestCase):
    def test_preserves_embedding_dtype_and_values(self):
        embeddings = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]],
            dtype=torch.float32,
        )
        mask = torch.tensor([[1, 1, 0]], dtype=torch.long)

        result = mean_pooling(embeddings, mask)

        self.assertEqual(result.dtype, embeddings.dtype)
        self.assertTrue(torch.equal(result, torch.tensor([[2.0, 3.0]])))

    @unittest.skipUnless(
        torch.cuda.device_count() >= 2,
        "requires two CUDA devices to verify cross-device TorchScript loading",
    )
    def test_traced_pooling_is_portable_between_cuda_devices(self):
        class PoolingModule(torch.nn.Module):
            def forward(self, embeddings, mask):
                return mean_pooling(embeddings, mask)

        export_device = torch.device(
            f"cuda:{torch.cuda.device_count() - 1}"
        )
        serving_device = torch.device("cuda:0")
        embeddings = torch.randn(2, 4, 3, device=export_device)
        mask = torch.ones(2, 4, dtype=torch.long, device=export_device)
        traced = torch.jit.trace(
            PoolingModule().to(export_device), (embeddings, mask)
        )

        result = traced.to(serving_device)(
            embeddings.to(serving_device), mask.to(serving_device)
        )

        self.assertEqual(result.device, serving_device)
        self.assertEqual(tuple(result.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
