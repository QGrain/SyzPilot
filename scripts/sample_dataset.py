import os
import argparse
import shutil
import random
from time import time
from tqdm import tqdm


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


def stat_prog(prog_dir, cover_dir):
    prog_fns = os.listdir(prog_dir)
    cover_fns = os.listdir(cover_dir)
    sig_coverfn_hash = {} # sig -> [cover_fns]
    prog_fns_hash = {sig:1 for sig in prog_fns}
    # assert(len(prog_fns), len(prog_fns_hash))
    for fn in tqdm(cover_fns, desc='[stat_prog] build sig_coverfn_hash', unit='file'):
        if '-' in fn:
            sig = fn.split('-')[0]
        else:
            sig = fn
        if sig not in prog_fns_hash:
            continue
        if sig not in sig_coverfn_hash:
            sig_coverfn_hash[sig] = [fn]
        else:
            sig_coverfn_hash[sig].append(fn)
    print('[stat_prog] %d/%d programs are matched in coverages'%(len(sig_coverfn_hash), len(prog_fns)))
    return sig_coverfn_hash


def sample_data_from_corpus(corpus_dir, out_dir, sample_num=None, debug=False):
    prog_out_dir, cover_out_dir = os.path.join(out_dir, 'programs'), os.path.join(out_dir, 'coverages')
    check_dir(prog_out_dir)
    check_dir(cover_out_dir)
    prog_dir, cover_dir = os.path.join(corpus_dir, 'programs'), os.path.join(corpus_dir, 'coverages')
    prog_fns, cover_fns = os.listdir(prog_dir), os.listdir(cover_dir)
    max_num = len(prog_fns)

    success_cnt = 0
    random.shuffle(prog_fns)
    for fn in prog_fns:
        if fn not in cover_fns:
            continue
        new_prog_fn = os.path.join(prog_out_dir, fn)
        new_cover_fn = os.path.join(cover_out_dir, fn)
        shutil.copy(os.path.join(prog_dir, fn), new_prog_fn)
        shutil.copy(os.path.join(cover_dir, fn), new_cover_fn)
        success_cnt += 1
        if sample_num and success_cnt >= sample_num:
            break
    print('Successfully sample %d program from %d for corpus dataset'%(success_cnt, max_num))


def sample_data_from_testcase(testcase_dir, out_dir, sample_num=None, debug=False):
    prog_out_dir, cover_out_dir = os.path.join(out_dir, 'programs'), os.path.join(out_dir, 'coverages')
    check_dir(prog_out_dir)
    check_dir(cover_out_dir)
    prog_dir, cover_dir = os.path.join(testcase_dir, 'programs'), os.path.join(testcase_dir, 'coverages')

    sig_coverfn_hash = stat_prog(prog_dir, cover_dir)
    valid_prog_fns = list(sig_coverfn_hash.keys())
    max_num = len(valid_prog_fns)

    success_cnt = 0
    avg_increase = 0
    random.shuffle(valid_prog_fns)
    for sig in tqdm(valid_prog_fns, desc=f'sample dataset from {len(valid_prog_fns)} testcases', unit='file'):
        merged_cover = set()
        avg_ncover = 0
        non_zero_cnt = 0
        for cover_fn in sig_coverfn_hash[sig]:
            cover = read_cover(os.path.join(cover_dir, cover_fn))
            n_cover = len(cover)
            if n_cover > 0:
                avg_ncover += n_cover
                non_zero_cnt += 1
                merged_cover |= set(cover)
        if len(merged_cover) > 0:
            avg_increase += len(merged_cover) - avg_ncover/non_zero_cnt
            new_prog_fn = os.path.join(prog_out_dir, sig)
            new_cover_fn = os.path.join(cover_out_dir, sig)
            shutil.copy(os.path.join(prog_dir, sig), new_prog_fn)
            with open(new_cover_fn, 'w') as f:
                for pc in merged_cover:
                    f.write('%s\n'%pc)
            success_cnt += 1
        if sample_num and success_cnt >= sample_num:
            break
    avg_increase /= success_cnt
    print('[sample] successfully sample %d from %d valid programs for testcase dataset'%(success_cnt, max_num))
    print('[sample] average cover increase after merging is %.1f'%avg_increase)


def get_args():
    parser = argparse.ArgumentParser(description='Raw Dataset Preprocessor: sample N data from rawset')
    parser.add_argument('-d', '--data_dir', type=str, help='data dir of corpus or testcase')
    parser.add_argument('-o', '--out_dir', type=str, help='out dir of corpus or testcase')
    parser.add_argument('-s', '--sample_num', type=int, help='sample number')
    parser.add_argument('-t', '--testcase', action='store_true', help='process testcase dataset')
    parser.add_argument('-D', '--DEBUG', action='store_true', help='DEBUG mode')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    t0 = time()
    args = get_args()
    if args.testcase:
        sample_data_from_testcase(args.data_dir, args.out_dir, args.sample_num, args.DEBUG)
    else:
        sample_data_from_corpus(args.data_dir, args.out_dir, args.sample_num, args.DEBUG)
    print('cost %.2fs'%(time()-t0))
