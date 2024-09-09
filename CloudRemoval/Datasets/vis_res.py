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


out_dir = ('/home/liuwang/liuwang_data/documents/projects/challenge/Restormer/submits/'
           '0905_cr_restormer-s_128x128_1xb16_160k_alld_hybirdv1_rcs/results')
gt_dir = "/home/liuwang/liuwang_data/documents/datasets/restoration/cloudremove/train/opt_clear"
cloudy_dir = "/home/liuwang/liuwang_data/documents/datasets/restoration/cloudremove/train/opt_cloudy"
gt_root = "/home/liuwang/liuwang_data/documents/datasets/restoration/cloudremove/train"
names = os.listdir(out_dir)
names_gt = sorted(os.listdir(gt_dir))

def is_almost_black(image, threshold=8):

    # 计算平均亮度
    mean_value = np.mean(image)

    # 判断平均值是否低于阈值
    return mean_value < threshold


class Item:
    def __init__(self, name, value, diff, score_neighbor, score_cloudy):
        self.name = name
        self.value = value
        self.diff = diff
        self.score_neighbor = score_neighbor
        self.score_cloudy = score_cloudy

    def __repr__(self):
        return (f"name={self.name}, score={self.value}, is_ocean={self.diff}, "
                f"score_neighbor={self.score_neighbor}, score_cloudy={self.score_cloudy}")


scores = []
for name in tqdm(names):
    img_out = cv2.imread(f'{out_dir}/{name}')
    img_gt = cv2.imread(f'{gt_dir}/{name}')
    sarvh = cv2.imread(f'{gt_root}/SAR/VH/{name}'.replace("_p_", "_VH_p_"), flags=cv2.IMREAD_UNCHANGED)
    score = calculate_score_crgame(img_out, img_gt)
    diff = is_almost_black(sarvh)

    idx = names_gt.index(name)
    img_out = cv2.imread(f'{gt_dir}/{names_gt[idx + 2]}')
    score_neighbor = calculate_score_crgame(img_out, img_gt)

    img_out = cv2.imread(f'{cloudy_dir}/{name}')
    score_cloudy = calculate_score_crgame(img_out, img_gt)
    item = Item(name, score, diff, score_neighbor, score_cloudy)
    scores.append(item)

sorted_items = sorted(scores, key=lambda x: x.value, reverse=False)

with open('./hard_list.txt', 'w') as f:
    for score in sorted_items:
        print(score)
        f.write(f'{score}\n')

        cloudy = cv2.imread(f'{gt_root}/opt_cloudy/{score.name}')
        sarvh = cv2.imread(f'{gt_root}/SAR/VH/{score.name}'.replace("_p_", "_VH_p_"), flags=cv2.IMREAD_UNCHANGED)
        sarvv = cv2.imread(f'{gt_root}/SAR/VV/{score.name}'.replace("_p_", "_VV_p_"), flags=cv2.IMREAD_UNCHANGED)
        sarvh2 = np.stack([sarvh] * 3, axis=2)
        sarvv2 = np.stack([sarvv] * 3, axis=2)

        img_out = cv2.imread(f'{out_dir}/{score.name}')
        img_gt = cv2.imread(f'{gt_dir}/{score.name}')
        imall = np.concatenate([cloudy, sarvh2, sarvv2, img_out, img_gt], axis=1)
        # imgs_in = np.concatenate([cloudy, sarvh, sarvv, img_gt], axis=2)

        cv2.imshow('', imall)

        if cv2.waitKey(0) == ord('q'):
            exit(0)

scores = [item.value for item in scores]
print('avg', np.array(scores).mean())
