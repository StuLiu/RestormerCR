"""
@Project : crgame
@File    : data_preprocess.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/8/1 下午6:51
@e-mail  : 1183862787@qq.com
"""
import os
import os.path

from glob import glob
import random

from tqdm import tqdm


# last 0% are val
percent = 0.01
train_root = '/data/train'
val_root = '/work/val'
os.makedirs(val_root, exist_ok=True)

imgs_gt = glob(f'{train_root}/opt_clear/*.png')
imgs_gt.sort()
print('train+val len=', len(imgs_gt))
val_size = int(len(imgs_gt) * percent)

f_train = open(f'{val_root}/train_filenames_alltrain.txt', 'w', encoding='utf-8')
f_val = open(f'{val_root}/val_filenames_alltrain.txt', 'w', encoding='utf-8')

i = 0
for img_path in tqdm(imgs_gt):
    filename = os.path.basename(img_path)
    f_train.write(f'{filename}\n')
    if i < val_size:
        f_val.write(f'{filename}\n')
    i += 1
