import os
import gzip
import shutil
import pickle
import json
import torch
import string
import numpy as np
from collections import OrderedDict
from time import time
from random import choice
from math import ceil


def check_dir(d):
    if os.path.isdir(d):
        return True
    else:
        try:
            os.makedirs(d)
            return True
        except:
            return False


def read_cover(fn):
    cover = []
    with open(fn, 'r') as f:
        cover = [line.strip() for line in f.readlines()]
    return cover


def read_prog(fn):
    syscall_list = []
    with open(fn, 'r') as f:
        syscall_list = [line.strip() for line in f.readlines()]
    return syscall_list


def read_prog_str(fn):
    prog_str = ''
    with open(fn, 'r') as f:
        prog_str = f.read()
    return prog_str


def merge_list(merged, cover):
    merged_set = set(merged)
    cover_set = set(cover)
    return list(merged_set | cover_set)


def merge_covers(covers):
    merged_cover_set = set()
    for cover in covers:
        cover_set = set(cover)
        merged_cover_set |= cover_set
    return list(merged_cover_set)


def copy_file(src_path, dst_path):
    if src_path != dst_path:
        shutil.copy(src_path, dst_path)


def print_t(s, t0):
    print('[%.2fs] %s'%(time()-t0, s))


def str_pc2ia(pc_str):
    pc_hex = int(pc_str, 16)
    ia_hex = pc_hex - 5
    return hex(ia_hex)


def restore_pc(short_pc):
    if type(short_pc) == str:
        return '0xffffffff' + short_pc[2:]
    elif type(short_pc) == list:
        long_pc = []
        for pc in short_pc:
            long_pc.append('0xffffffff' + pc[2:])
        return long_pc
    else:
        raise ValueError('Invalid type of short_pc:', short_pc, type(short_pc))


def save_json(d, fpath):
    with open(fpath, 'w') as f:
        json.dump(d, f)


def load_json(fpath):
    if os.path.isfile(fpath):
        with open(fpath, 'r') as f:
            return json.load(f)
    else:
        return {}


def save_json_gz(d, fpath):
    if fpath.endswith('.json'):
        fpath = f'{fpath}.gz'
    with gzip.open(fpath, 'wt', encoding='utf-8') as f:
        json.dump(d, f)


def load_json_gz(fpath):
    if fpath.endswith('.json'):
        fpath = f'{fpath}.gz'
    if os.path.isfile(fpath):
        with gzip.open(fpath, 'rt', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {}


def save_pkl(d, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(d, f)


def load_pkl(fpath):
    try:
        with open(fpath, 'rb') as f:
            return pickle.load(f)
    except:
        return {}


def get_func_id(fileid, func_name, func_file):
    for fn in fileid:
        if func_file in fn:
            prefix_id = fileid[fn]
            return '%s@%s'%(prefix_id, func_name)
    return None


class CachedTokenizingCollator:
    """Tokenize each distinct program at most once within a trainer process.

    Cached tensors preserve the exact fixed-length tokenizer output.  The cache
    is process-local, so a new tokenizer, model, or max length naturally starts
    with an empty cache and cannot reuse stale encodings.
    """

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
    result = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
    return result

def calc_pos_weight(labels, thres=10, normalize=True, custom_pos=None):
    pos_cnt = labels.sum(dim=0)
    neg_cnt = (labels == 0).sum(dim=0)
    pos_w = neg_cnt/pos_cnt
    if custom_pos != None:
        try:
            pos_w *= custom_pos
        except:
            pass
    if normalize == True:
        pos_w = pos_w / pos_w.min()
    pos_w = torch.clamp(torch.sqrt(pos_w), max=thres)
    return pos_w


def save_pos_weight(pos_w, path):
    pos_w_list = pos_w.tolist()
    with open(path, 'w') as f:
        json.dump(pos_w_list, f)


def load_pos_weight(path):
    with open(path, 'r') as f:
        return json.load(f)


def rand_str(n):
    return ''.join(choice(string.printable) for _ in range(n))


def get_fpaths_and_fns(dirs):
    fpaths = []
    fns = {}
    if dirs is None:
        return fpaths, fns
    for d in dirs:
        files = os.listdir(d)
        files_sorted = sorted(files)
        for fn in files_sorted:
            if fn not in fns:
                fpath = os.path.join(d, fn)
                fpaths.append(fpath)
                fns[fn] = fpath
    return fpaths, fns


def get_fn(fpath):
    return os.path.basename(fpath)


def get_dirn(dirpath):
    if dirpath[-1] == '/':
        dirpath = dirpath[:-1]
    return os.path.basename(dirpath)


# self defined json_dumps for dataset_cache
def my_json_dumps(obj, depth=1, indent=4):
    if isinstance(obj, list):
        # if it is list, then do not write it in multi-line
        return json.dumps(obj)
    elif isinstance(obj, dict):
        # recursively process the sub-dicts
        indent_str = ' ' * indent * depth
        indent_close_str = ' ' * indent * (depth-1)
        return '{\n' + ',\n'.join(f'{indent_str}"{k}": {my_json_dumps(v, depth+1, indent)}' for k, v in obj.items()) + '\n%s}'%indent_close_str
    else:
        return json.dumps(obj)


def sample_numbers_by_rate(numbers, rate):
    ratios = [numbers[i] / rate[i] for i in range(len(numbers))]
    k = ratios.index(min(ratios))
    # make sure that sampled_numbers[j] won't be larger than numbers[j]
    sampled_numbers = [min(int(numbers[k] * rate[j] / rate[k]), numbers[j]) for j in range(len(numbers))]
    return sampled_numbers


# there are {orig_num} programs, calculate the capable {aug_op_num} for generating {aug_num} new programs
def get_aug_op_num_llx(orig_num, aug_num):
    aug_op_num = 0
    c = ceil(aug_num/orig_num)
    if aug_num <= 0.5 * orig_num:
        aug_op_num = c + 0 # so when aug_num is 0, aug_op_num is 0
    elif aug_num <= 1 * orig_num:
        aug_op_num = c + 1
    elif aug_num <= 2 * orig_num:
        aug_op_num = c + 2
    elif aug_num <= 4 * orig_num:
        aug_op_num = c + 5
    else:
        aug_op_num = c + 10
    return aug_op_num


# there are {orig_num} programs, calculate the capable {aug_op_num} for generating {aug_num} new programs
def get_aug_op_num(orig_num, aug_num):
    aug_op_num = 0
    c = ceil(aug_num/orig_num)
    i = -1
    while 1:
        if (aug_num/orig_num) <= 2**i:
            aug_op_num = c + ceil(2**i)+i # 0, 1, 3, 5, 9, 17, 33...
            break
        i = i + 1
    return aug_op_num
