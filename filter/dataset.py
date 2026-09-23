from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os

from utils import create_collate_fn, load_pkl

class ProgramDataset(Dataset):
    def __init__(self, prog_path, label_path):
        self.progs_str = load_pkl(prog_path)
        self.progs_sigs = list(self.progs_str.keys())
        self.labels = load_pkl(label_path)
        print('[pid=%d][ProgramDataset] Loaded programs: %d'%(os.getpid(), len(self.progs_str)))

        # get self.label_dim
        if len(self.labels) > 0:
            sig0 = list(self.labels.keys())[0]
            self.label_dim = len(self.labels[sig0])
        else:
            raise ValueError("no labels available")

        assert(len(self.progs_str) == len(self.labels))

    def __len__(self):
        return len(self.progs_str)

    def __getitem__(self, idx):
        sig = self.progs_sigs[idx]
        return self.progs_str[sig], torch.tensor(self.labels[sig])

if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    ds = ProgramDataset("../programs", "../labels.pt")
    dl = DataLoader(ds, batch_size=2, collate_fn=create_collate_fn(tokenizer))

    print(next(iter(dl)))
