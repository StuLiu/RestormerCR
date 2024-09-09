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
import io
import cv2
from glob import glob
import numpy as np


# print(cv2.IMREAD_GRAYSCALE)
train_root = '/home/liuwang/liuwang_data/documents/datasets/restoration/cloudremove/train'
train_merge = '/home/liuwang/liuwang_data/documents/datasets/restoration/cloudremove/train_merge'
train_merge_input = os.path.join(train_merge, 'input')
train_merge_target = os.path.join(train_merge, 'target')
os.makedirs(train_merge, exist_ok=True)
os.makedirs(train_merge_input, exist_ok=True)
os.makedirs(train_merge_target, exist_ok=True)


imgs_gt = glob(f'{train_root}/opt_clear/*.png')
imgs_cloudy = glob(f'{train_root}/opt_cloudy/*.png')
imgs_sarvh = glob(f'{train_root}/SAR/VH/*.png')
imgs_sarvv = glob(f'{train_root}/SAR/VV/*.png')
imgs_gt.sort()
imgs_cloudy.sort()
imgs_sarvh.sort()
imgs_sarvv.sort()

for i in range(0, len(imgs_gt), 100):
    gt = cv2.imread(imgs_gt[i])
    cloudy = cv2.imread(imgs_cloudy[i])
    sarvh = cv2.imread(imgs_sarvh[i], flags=cv2.IMREAD_UNCHANGED)
    sarvv = cv2.imread(imgs_sarvv[i], flags=cv2.IMREAD_UNCHANGED)
    sarvh2 = np.stack([sarvh] * 3, axis=2)
    sarvv2 = np.stack([sarvv] * 3, axis=2)
    imall = np.concatenate([cloudy, sarvh2, sarvv2, gt], axis=1)
    cv2.imshow(f'head', imall.astype(np.uint8))

    j = len(imgs_gt) - i - 1
    gt = cv2.imread(imgs_gt[j])
    cloudy = cv2.imread(imgs_cloudy[j])
    sarvh = cv2.imread(imgs_sarvh[j], flags=cv2.IMREAD_UNCHANGED)
    sarvv = cv2.imread(imgs_sarvv[j], flags=cv2.IMREAD_UNCHANGED)
    sarvh2 = np.stack([sarvh] * 3, axis=2)
    sarvv2 = np.stack([sarvv] * 3, axis=2)
    imall = np.concatenate([cloudy, sarvh2, sarvv2, gt], axis=1)
    cv2.imshow(f'tail', imall.astype(np.uint8))

    print(i, os.path.basename(imgs_gt[i]))
    print(j, os.path.basename(imgs_gt[j]))
    key = cv2.waitKey(0)
    if key == ord('q'):
        exit(0)