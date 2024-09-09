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


cloudy_dir = "/home/liuwang/liuwang_data/documents/datasets/restoration/cloudremove/train/opt_cloudy"
gt_dir = "/home/liuwang/liuwang_data/documents/datasets/restoration/cloudremove/train/opt_clear"
gt_root = "/home/liuwang/liuwang_data/documents/datasets/restoration/cloudremove/train"
names = os.listdir(gt_dir)


def is_almost_black(image, threshold=8):

    # 计算平均亮度
    mean_value = np.mean(image)

    # 判断平均值是否低于阈值
    return bool(mean_value < threshold)


def detect_clouds(image_bgr, train=True):
    img = image_bgr.copy()
    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义检测云的HSV范围
    # 调整范围来更好地识别暗云和轻云
    lower_cloud = np.array([0, 0, 120])
    upper_cloud = np.array([180, 60, 255])

    # 创建云的掩码
    mask = cv2.inRange(hsv, lower_cloud, upper_cloud)

    # 形态学操作：去除小噪点和填补孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 选取较大的连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):  # Index 0 is the background
        if stats[i, cv2.CC_STAT_AREA] < 500:  # 用面积过滤小噪点
            mask[labels == i] = 0

    # 计算云的百分比
    cloud_percentage = 100 * np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])

    # # 可视化结果
    # result = cv2.bitwise_and(img, img, mask=mask)
    # # cv2.imshow('Original', img)
    # # cv2.imshow('Clouds', result)
    # cv2.imshow(f'Mask_{train}', mask)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()

    return cloud_percentage


gtname2isocean_cloudyrate = {}
for name in tqdm(names):
    img_cloudy = cv2.imread(f'{cloudy_dir}/{name}')
    img_gt = cv2.imread(f'{gt_dir}/{name}')
    sarvh = cv2.imread(f'{gt_root}/SAR/VH/{name}'.replace("_p_", "_VH_p_"), flags=cv2.IMREAD_UNCHANGED)
    sarvh2 = np.stack([sarvh] * 3, axis=2)
    isOcean = is_almost_black(sarvh)
    cloudy_rate = detect_clouds(img_cloudy)
    gtname2isocean_cloudyrate[str(name)] = [isOcean, cloudy_rate]
    # print(str(name), isOcean, cloudy_rate)
    # imgall = np.concatenate([img_cloudy, sarvh2, img_gt], axis=1)
    # cv2.imshow('', imgall)
    # if ord('q') == cv2.waitKey(0):
    #     exit(0)
    # break

with open(f'{gt_root}/gtname2isocean_cloudyrate.json', 'w') as f:
    json.dump(gtname2isocean_cloudyrate, f, indent=4)

# sorted_items = sorted(scores, key=lambda x: x.value)
#
# with open('./hard_list.txt', 'w') as f:
#     for score in sorted_items:
#         print(score)
#         f.write(f'{score}\n')
#
#         cloudy = cv2.imread(f'{gt_root}/opt_cloudy/{score.name}')
#         sarvh = cv2.imread(f'{gt_root}/SAR/VH/{score.name}'.replace("_p_", "_VH_p_"), flags=cv2.IMREAD_UNCHANGED)
#         sarvv = cv2.imread(f'{gt_root}/SAR/VV/{score.name}'.replace("_p_", "_VV_p_"), flags=cv2.IMREAD_UNCHANGED)
#         sarvh2 = np.stack([sarvh] * 3, axis=2)
#         sarvv2 = np.stack([sarvv] * 3, axis=2)
#
#         img_out = cv2.imread(f'{out_dir}/{score.name}')
#         img_gt = cv2.imread(f'{gt_dir}/{score.name}')
#         imall = np.concatenate([cloudy, sarvh2, sarvv2, img_out, img_gt], axis=1)
#         # imgs_in = np.concatenate([cloudy, sarvh, sarvv, img_gt], axis=2)
#
#         cv2.imshow('', imall)
#
#         if cv2.waitKey(0) == ord('q'):
#             exit(0)
#
# scores = [item.value for item in scores]
# print('avg', np.array(scores).mean())
