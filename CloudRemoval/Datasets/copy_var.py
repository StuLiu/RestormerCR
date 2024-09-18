"""
@Project : crgame
@File    : data_preprocess.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/8/1 下午6:51
@e-mail  : 1183862787@qq.com
"""
import os.path

import skimage
import shutil
from tqdm import tqdm


train_root = '/home/liuwang/liuwang_data/documents/datasets/restoration/cloudremove/train'
val_root = '/home/liuwang/liuwang_data/documents/datasets/restoration/cloudremove/val'

train_root_gt = f'{train_root}/opt_clear'
train_root_cloudy = f'{train_root}/opt_cloudy'
train_root_sarvh = f'{train_root}/SAR/VH'
train_root_sarvv = f'{train_root}/SAR/VV'

val_root_gt = f'{val_root}/opt_clear'
val_root_cloudy =f'{val_root}/opt_cloudy'
val_root_sarvh = f'{val_root}/SAR/VH'
val_root_sarvv = f'{val_root}/SAR/VV'

os.makedirs(val_root_gt, exist_ok=True)
os.makedirs(val_root_cloudy, exist_ok=True)
os.makedirs(val_root_sarvh, exist_ok=True)
os.makedirs(val_root_sarvv, exist_ok=True)

with open(f'{train_root}/val_filenames_alltrain.txt', 'r') as f:
    lines = f.readlines()
    for line in tqdm(lines):
        file_name = line.strip()
        shutil.copy(f'{train_root_gt}/{file_name}', f'{val_root_gt}/{file_name}')
        shutil.copy(f'{train_root_cloudy}/{file_name}', f'{val_root_cloudy}/{file_name}')
        shutil.copy(f'{train_root_sarvh}/{file_name.replace("_p_", "_VH_p_")}',
                    f'{val_root_sarvh}/{file_name.replace("_p_", "_VH_p_")}')
        shutil.copy(f'{train_root_sarvv}/{file_name.replace("_p_", "_VV_p_")}',
                    f'{val_root_sarvv}/{file_name.replace("_p_", "_VV_p_")}')
