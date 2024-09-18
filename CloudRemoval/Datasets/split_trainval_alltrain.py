"""
@Project : crgame
@File    : data_preprocess.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/8/1 下午6:51
@e-mail  : 1183862787@qq.com
"""
import os.path

from glob import glob
import random

from tqdm import tqdm


# last 0% are val
percent = 0.01
train_root = '/home/liuwang/liuwang_data/documents/datasets/restoration/cloudremove/train'

imgs_gt = glob(f'{train_root}/opt_clear/*.png')
imgs_gt.sort()
print('train+val len=', len(imgs_gt))

f_train = open(f'{train_root}/train_filenames_alltrain.txt', 'w', encoding='utf-8')
f_val = open(f'{train_root}/val_filenames_alltrain.txt', 'w', encoding='utf-8')

for img_path in tqdm(imgs_gt):
    filename = os.path.basename(img_path)
    f_train.write(f'{filename}\n')
    if random.random() < percent:
        f_val.write(f'{filename}\n')
