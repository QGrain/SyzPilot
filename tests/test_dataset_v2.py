"""Canonical online-dataset loading regression tests."""

import hashlib
import pickle
import tempfile
import unittest
from pathlib import Path
from torch.utils.data import DataLoader

from filter.dataset_v2 import ProgramDatasetV2_2, load_canonical_records


def write_batch(root, batch_id, rows):
    programs = {}
    labels = {}
    for program, label in rows:
        signature = hashlib.sha1(program.encode("utf-8")).hexdigest()
        programs[signature] = program
        labels[signature] = label
    with (root / f"progs_batch_{batch_id}.pkl").open("wb") as handle:
        pickle.dump(programs, handle)
    with (root / f"labels_batch_{batch_id}.pkl").open("wb") as handle:
        pickle.dump(labels, handle)


class CanonicalDatasetTest(unittest.TestCase):
    def test_cross_batch_duplicates_keep_deepest_label(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_batch(root, 1, [("repeat()", [False, False, False, True])])
            write_batch(root, 2, [
                ("repeat()", [False, True, False, False]),
                ("unique()", [True, False, False, False]),
            ])

            records = load_canonical_records(root, 4, [1, 2])

            self.assertEqual(len(records), 2)
            repeat_sig = hashlib.sha1(b"repeat()").hexdigest()
            self.assertEqual(records[repeat_sig][1], [False, False, False, True])

    def test_test_dataset_excludes_all_training_signatures(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_batch(root, 0, [
                ("history-overlap()", [False, False, True]),
            ])
            write_batch(root, 1, [
                ("train-only()", [True, False, False]),
                ("overlap()", [False, True, False]),
            ])
            write_batch(root, 2, [
                ("test-only()", [True, False, False]),
                ("overlap()", [False, False, True]),
                ("history-overlap()", [False, True, False]),
            ])

            train = ProgramDatasetV2_2(
                root, 3, [1], canonical_indices=[0, 1, 2]
            )
            seen_records = load_canonical_records(root, 3, [0, 1])
            test = ProgramDatasetV2_2(
                root, 3, [2], exclude_signatures=seen_records,
                canonical_indices=[0, 1, 2],
            )

            self.assertTrue(train.signatures.isdisjoint(test.signatures))
            self.assertEqual(len(test.signatures), 1)
            self.assertIn(hashlib.sha1(b"test-only()").hexdigest(), test.signatures)
            overlap_sig = hashlib.sha1(b"overlap()").hexdigest()
            self.assertEqual(train.records[overlap_sig][1], [False, False, True])

    def test_non_repeating_dataset_yields_each_canonical_record_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_batch(root, 1, [
                ("one()", [True, False]),
                ("two()", [False, True]),
            ])
            dataset = ProgramDatasetV2_2(root, 2, [1], repeat=False)

            rows = list(dataset)

            self.assertEqual(len(rows), 2)
            self.assertEqual({program for program, _ in rows}, {"one()", "two()"})
            batches = list(DataLoader(dataset, batch_size=4))
            self.assertEqual(len(batches), 1)
            self.assertEqual(len(batches[0][0]), 2)

    def test_class_aware_replay_upsamples_rare_positive_classes_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = []
            rows.extend(
                (f"negative${index}()", [True, False, False, False])
                for index in range(4)
            )
            rows.append(("rare$one()", [False, True, False, False]))
            rows.extend(
                (f"medium${index}()", [False, False, True, False])
                for index in range(4)
            )
            rows.extend(
                (f"common${index}()", [False, False, False, True])
                for index in range(9)
            )
            write_batch(root, 1, rows)

            dataset = ProgramDatasetV2_2(
                root, 4, [1], class_aware_replay=True
            )

            self.assertEqual(dataset.replay_weights[0], 1.0)
            self.assertEqual(dataset.replay_weights[1], 3.0)
            self.assertEqual(dataset.replay_weights[2], 1.5)
            self.assertEqual(dataset.replay_weights[3], 1.0)
            self.assertEqual(
                dataset.effective_class_counts,
                {0: 4, 1: 3, 2: 6, 3: 9},
            )

            dataset._refresh_files()
            actual_counts = {class_index: 0 for class_index in range(4)}
            for _, label in dataset.tracking_examples:
                actual_counts[int(label.argmax().item())] += 1
            self.assertEqual(actual_counts, dataset.effective_class_counts)

    def test_replay_cap_and_near_equal_classes_do_not_overshoot(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rows = [("rare()", [False, True, False, False])]
            rows.extend(
                (f"near${index}()", [False, False, True, False])
                for index in range(99)
            )
            rows.extend(
                (f"common${index}()", [False, False, False, True])
                for index in range(100)
            )
            write_batch(root, 1, rows)

            dataset = ProgramDatasetV2_2(
                root, 4, [1], class_aware_replay=True
            )

            self.assertEqual(dataset.replay_weights[1], 8.0)
            self.assertLess(dataset.replay_weights[2], 1.01)
            self.assertEqual(dataset.effective_class_counts[1], 8)
            self.assertEqual(dataset.effective_class_counts[2], 99)
            self.assertEqual(dataset.effective_class_counts[3], 100)

            disabled = ProgramDatasetV2_2(
                root, 4, [1], class_aware_replay=False
            )
            self.assertEqual(
                disabled.effective_class_counts,
                dict(disabled.raw_class_counts),
            )


if __name__ == "__main__":
    unittest.main()
