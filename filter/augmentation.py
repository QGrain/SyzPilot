import os
import argparse
import hashlib
import binascii
from random import shuffle
from time import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# Suppose the directory structure is:
# |-reach-labelA
#   |-programs
#   |-coverages
# |-reach-labelB
#   |-programs
#   |-coverages
# ...
# Therefore, this script is used for augmentation of the directory with less samples.
# Usage: python augmentation.py --in_dirs reach-labelA reach-labelB ... [--out_dirs reach-labelA-aug reach-labelB-aug ...] [--local]
# It will augment the programs in the in_dirs and copy the coverages correspondingly [to out_dirs by default] or [locally in the in_dirs].


def read_file(path):
    with open(path, 'r') as f:
        return f.read().strip()


def write_file(s, path):
    with open(path, 'w') as f:
        f.write(s)


def get_out_dir(in_dir, out_dir, local=False):
    if in_dir[-1] == '/':
        in_dir = in_dir[:-1]
    if local == True:
        out_dir = in_dir
    elif out_dir == None or out_dir == '':
        dir_name = os.path.dirname(in_dir)
        base_name = os.path.basename(in_dir)
        out_dir = os.path.join(dir_name, '%s_aug'%base_name)
    return in_dir, out_dir


def calc_hash(prog_str):
        prog_bytes = bytes(prog_str, encoding='utf-8')
        h = hashlib.sha1(prog_bytes)
        return binascii.hexlify(h.digest()).decode('utf-8')


# to be migrated from preprocess_dataset.py
class ProgAugmentor:
    def __init__(self, prog_fpaths_dict, aug_op_num, aug_dir):
        pass


class DataAugmentor:
    def __init__(self, in_dir, out_dir=None, local=False, aug_num_each_op=5):
        self.in_dir, self.out_dir = get_out_dir(in_dir, out_dir, local)
        self.in_prog_dir = os.path.join(self.in_dir, 'programs')
        self.in_cover_dir = os.path.join(self.in_dir, 'coverages')
        self.out_prog_dir = os.path.join(self.out_dir, 'programs')
        self.out_cover_dir = os.path.join(self.out_dir, 'coverages')
        os.makedirs(self.out_prog_dir, exist_ok=True)
        os.makedirs(self.out_cover_dir, exist_ok=True)
        assert(aug_num_each_op >= 1)
        self.aug_num_each_op = aug_num_each_op
        print('[INFO] DataAugmentor initialized')

    def aug(self):
        print('[INFO] Start augmentation')
        prog_files = os.listdir(self.in_prog_dir)
        for fn in tqdm(prog_files, desc='Augmenting files', unit='file'):
            prog_str = read_file(os.path.join(self.in_prog_dir, fn))
            cover = read_file(os.path.join(self.in_cover_dir, fn))
            write_file(prog_str, os.path.join(self.out_prog_dir, fn))
            write_file(cover, os.path.join(self.out_cover_dir, fn))
            if prog_str == '':
                print('[WARN] prog_str of %s is blank, skip the augmentation' % os.path.join(self.in_prog_dir, fn))
                continue
            self.duplication(prog_str, cover)
            # self.substitution(prog_str, cover)
            # self.minimization(prog_str, cover)
            # others are not important
        print('[INFO] Augmentation done')

    def substitution(self, prog_str, cover):
        thesaurus = {
            "accept": "accept4"
        }
        pass

    def minimization(self, prog_str, cover):
        # syz-minimize todo
        pass

    def mutation(self, prog_str, cover):
        # tree-sitter todo
        pass

    def model_generation(self, prog_str, cover):
        # LLM todo
        pass

    def duplication(self, prog_str, cover):
        syscalls = prog_str.split('\n')
        if len(syscalls) == 1:
            aug_num = min(2, self.aug_num_each_op)
            for i in range(aug_num):
                new_syscalls = syscalls.copy()
                if i == 0:
                    new_syscalls.append(new_syscalls[0])
                elif i == 1:
                    new_syscalls.append(new_syscalls[0])
                    new_syscalls.append(new_syscalls[0])
                new_prog_str = '\n'.join(new_syscalls)
                new_sig = calc_hash(new_prog_str)
                write_file(new_prog_str, os.path.join(self.out_prog_dir, new_sig))
                write_file(cover, os.path.join(self.out_cover_dir, new_sig))
        else:
            append_pos = [i for i in range(len(syscalls))]
            insert_pos = [i for i in range(len(syscalls)-1)]
            shuffle(append_pos)
            shuffle(insert_pos)
            aug_num = min(len(append_pos)+len(insert_pos), self.aug_num_each_op)
            for i in range(aug_num):
                new_syscalls = syscalls.copy()
                if i < len(syscalls):
                    new_syscalls.append(new_syscalls[append_pos[i]])
                else:
                    j = i - len(append_pos)
                    new_syscalls.insert(insert_pos[j], new_syscalls[insert_pos[j]])

                new_prog_str = '\n'.join(new_syscalls)
                new_sig = calc_hash(new_prog_str)
                write_file(new_prog_str, os.path.join(self.out_prog_dir, new_sig))
                write_file(cover, os.path.join(self.out_cover_dir, new_sig))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('data augmentation for unbalanced dataset')
    parser.add_argument('--in_dir', type=str, help='in dir to be augmented')
    parser.add_argument('--out_dir', type=str, help='out dir to store')
    parser.add_argument('--local', action='store_true', help='store in local in dir')
    parser.add_argument('--aug_num', type=int, default=5, help='aug_num for each operation')
    args = parser.parse_args()

    data_augmentor = DataAugmentor(args.in_dir, args.out_dir, args.local, args.aug_num)
    data_augmentor.aug()