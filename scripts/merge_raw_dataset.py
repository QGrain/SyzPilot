import os
import shutil
import argparse
from tqdm import tqdm

def copy_dir_files(src_dir, dst_dir):
    src_fns = os.listdir(src_dir)
    dst_fns = os.listdir(dst_dir)
    dst_fns_dict = {fn:1 for fn in dst_fns}
    print('[INFO] build dst_fns_dict')
    copied_cnt = 0
    for fn in tqdm(src_fns, desc=f'copy files from {src_dir} to {dst_dir}', unit='file'):
        if fn not in dst_fns_dict:
            shutil.copyfile(os.path.join(src_dir, fn), os.path.join(dst_dir, fn))
            copied_cnt += 1
    print(f'[INFO] copied {copied_cnt} files')


parser = argparse.ArgumentParser('merge raw dataset')
parser.add_argument('new_dataset', type=str, help='path to the new raw dataset')
parser.add_argument('mixed_dataset', type=str, help='path to the mixed raw dataset')
args = parser.parse_args()

copy_dir_files(os.path.join(args.new_dataset, 'programs'), os.path.join(args.mixed_dataset, 'programs'))
copy_dir_files(os.path.join(args.new_dataset, 'coverages'), os.path.join(args.mixed_dataset, 'coverages'))