"""
@Project : Restormer
@File    : vis_res.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/9/4 下午11:18
@e-mail  : 1183862787@qq.com
"""

import torch
import cv2
import numpy as np
import os
import math
from tqdm import tqdm
from basicsr.metrics.cr_metrics import calculate_score_crgame
import json


test_dir = "/home/liuwang/liuwang_data/documents/datasets/restoration/cloudremove/test/opt_cloudy"
gt_root = "/home/liuwang/liuwang_data/documents/datasets/restoration/cloudremove/train/opt_cloudy"
names = os.listdir(test_dir)
train_names = os.listdir(gt_root)
print(len(names))
print(len(train_names))

neighbors = []
for name in names:
    idx = int(name[:-4].split('_')[-1])
    nei1 = f'{name[:10]}{idx + 1:04d}.png'
    nei2 = f'{name[:10]}{idx - 1:04d}.png'
    if nei1 in train_names:
        neighbors.append(nei1)
    if nei2 in train_names:
        neighbors.append(nei2)
    print(name, nei1, nei2)
print(len(neighbors))

with open(f'{gt_root}/../neighbors.json', 'w') as f:
    json.dump({'neighbors': neighbors}, f)
