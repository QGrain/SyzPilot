
from pathlib import Path
from collections import Counter
import math
import pickle
import random
import re
from typing import List, Union

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader


def load_canonical_records(data_root, num_classes, indices, exclude_signatures=None):
    """Load unique programs and keep their deepest label across batch files."""
    data_dir = Path(data_root)
    excluded = set(exclude_signatures or ())
    records = {}
    for batch_id in sorted(set(indices)):
        progs_path = data_dir / f"progs_batch_{batch_id}.pkl"
        labels_path = data_dir / f"labels_batch_{batch_id}.pkl"
        if not progs_path.is_file() or not labels_path.is_file():
            continue
        with progs_path.open("rb") as file_handle:
            programs = pickle.load(file_handle)
        with labels_path.open("rb") as file_handle:
            labels = pickle.load(file_handle)
        if not isinstance(programs, dict) or not isinstance(labels, dict):
            raise ValueError(f"batch {batch_id} payloads must be dictionaries")
        if programs.keys() != labels.keys():
            raise ValueError(f"batch {batch_id} program and label keys differ")
        for signature, program in programs.items():
            if signature in excluded:
                continue
            label = list(labels[signature])
            if len(label) != num_classes:
                raise ValueError(
                    f"batch {batch_id} label width {len(label)} != {num_classes}"
                )
            selected = [index for index, value in enumerate(label) if bool(value)]
            if len(selected) != 1:
                raise ValueError(
                    f"batch {batch_id} label must be exactly one-hot: {label}"
                )
            previous = records.get(signature)
            if previous is None:
                records[signature] = (program, label, selected[0])
                continue
            if previous[0] != program:
                raise ValueError(f"program hash collision for signature {signature}")
            if selected[0] > previous[2]:
                records[signature] = (program, label, selected[0])
    return records


class ProgramDatasetV2(IterableDataset):
    def __init__(self, data_root, num_classes):
        self.data_dir = Path(data_root)
        self.num_classes = num_classes

        self.tracking_examples = []
        self.num_examples = 10000

    def _refresh_files(self):
        id_pattern = re.compile(".*batch_(\\d+).pkl")

        progs = list(self.data_dir.glob("*progs_batch_*.pkl"))
        labels = list(self.data_dir.glob("*labels_batch_*.pkl"))

        progs = {eval(id_pattern.findall(str(file))[0]): file for file in progs}
        labels = {eval(id_pattern.findall(str(file))[0]): file for file in labels}
        ids = sorted(list(progs.keys()))

        while ids and len(self.tracking_examples) < self.num_examples:
            max_id = ids.pop()
            if max_id not in labels:
                continue
            with open(progs[max_id], 'rb') as fp:
                prog_obj = pickle.load(fp)
            with open(labels[max_id], 'rb') as fp:
                label_obj = pickle.load(fp)
            for prog_id in prog_obj:
                if prog_id in label_obj:
                    self.tracking_examples.append((
                        prog_obj[prog_id],
                        torch.tensor(label_obj[prog_id], dtype=torch.int)
                    ))

    def generate(self):
        # Keep running forever
        while True:
            if len(self.tracking_examples) == 0:
                self._refresh_files()
                continue
            yield self.tracking_examples.pop()

    def __iter__(self):
        return iter(self.generate())


class ProgramDatasetV2_1(ProgramDatasetV2):
    def __init__(
            self,
            data_root,
            num_classes,
            prog_filename,
            label_filename,
            split="train",
            train_ratio=0.8,
    ):
        super().__init__(data_root, num_classes)

        with open(self.data_dir / prog_filename, 'rb') as fp:
            self.prog_obj = pickle.load(fp)
        with open(self.data_dir / label_filename, 'rb') as fp:
            self.label_obj = pickle.load(fp)

        self.split = split
        self.keys = list(self.prog_obj.keys())
        split_index = int(len(self.keys) * train_ratio)
        if split == "train":
            self.keys = self.keys[:split_index]
        else:
            self.keys = self.keys[split_index:]

    def _refresh_files(self):
        for prog_id in self.keys:
            if prog_id in self.label_obj:
                self.tracking_examples.append((
                    self.prog_obj[prog_id],
                    torch.tensor(self.label_obj[prog_id], dtype=torch.int),
                ))
        # if self.split == "train":
        random.shuffle(self.tracking_examples)


class ProgramDatasetV2_2(ProgramDatasetV2):
    def __init__(self, data_root, num_classes, indicies: List[int],
                 exclude_signatures=None, canonical_indices=None, repeat=True,
                 class_aware_replay=False, max_positive_replay=8):
        super().__init__(data_root, num_classes)
        self.indicies = indicies
        self.repeat = repeat
        assert len(self.indicies) > 0
        self.num_examples = 1e10
        member_records = load_canonical_records(
            self.data_dir,
            self.num_classes,
            self.indicies,
        )
        canonical_records = member_records
        if canonical_indices is not None:
            canonical_records = load_canonical_records(
                self.data_dir, self.num_classes, canonical_indices
            )
        excluded = set(exclude_signatures or ())
        self.records = {
            signature: canonical_records[signature]
            for signature in member_records
            if signature not in excluded
        }
        if not self.records:
            raise ValueError("No canonical examples remain after deduplication")
        self.signatures = frozenset(self.records)
        self.class_aware_replay = bool(class_aware_replay and repeat)
        self.max_positive_replay = max(1, int(max_positive_replay))
        self.raw_class_counts = Counter(
            exact_class for _, _, exact_class in self.records.values()
        )
        self.replay_weights = {
            class_index: 1.0 for class_index in self.raw_class_counts
        }
        positive_counts = [
            count for class_index, count in self.raw_class_counts.items()
            if class_index > 0 and count > 0
        ]
        if self.class_aware_replay and positive_counts:
            largest_positive_class = max(positive_counts)
            for class_index, count in self.raw_class_counts.items():
                if class_index == 0:
                    continue
                self.replay_weights[class_index] = min(
                    float(self.max_positive_replay),
                    max(1.0, math.sqrt(largest_positive_class / count)),
                )
        self.effective_class_counts = {
            class_index: round(count * self.replay_weights[class_index])
            for class_index, count in self.raw_class_counts.items()
        }

    def __iter__(self):
        if self.repeat:
            return super().__iter__()
        return iter(
            (program, torch.tensor(label, dtype=torch.int))
            for program, label, _ in self.records.values()
        )

    def _refresh_files(self):
        records_by_class = {
            class_index: [] for class_index in self.raw_class_counts
        }
        for program, label, exact_class in self.records.values():
            records_by_class[exact_class].append((program, label))
        for class_index, class_records in records_by_class.items():
            random.shuffle(class_records)
            target_count = self.effective_class_counts[class_index]
            base_repeats, extra_records = divmod(
                target_count, len(class_records)
            )
            for record_index, (program, label) in enumerate(class_records):
                repeats = base_repeats + int(record_index < extra_records)
                self.tracking_examples.extend(
                    (program, torch.tensor(label, dtype=torch.int))
                    for _ in range(repeats)
                )

        random.shuffle(self.tracking_examples)
