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
import random


def detect_clouds1(img_bgr, is_train=True):
    # 读入图像
    img = img_bgr.copy()

    # 转换到HSV色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义白色区域的HSV范围
    # 这个范围可能需要基于实际图像进行调整
    sensitivity = 1
    lower_white = np.array([0, 0, 255 - sensitivity])
    upper_white = np.array([180, sensitivity, 255])

    # 创建云的掩码
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # 计算云的百分比
    cloud_percentage = 100 * np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])

    # 可视化结果
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow(f'Mask_{is_train}', mask)
    return cloud_percentage


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

    # 可视化结果
    result = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow('Original', img)
    # cv2.imshow('Clouds', result)
    cv2.imshow(f'Mask_{train}', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return cloud_percentage

# print(cv2.IMREAD_GRAYSCALE)
train_root = f'/home/liuwang/liuwang_data/documents/datasets/restoration/cloudremove/train'
imgs_gt = glob(f'{train_root}/opt_clear/*.png')
imgs_cloudy = glob(f'{train_root}/opt_cloudy/*.png')
imgs_sarvh = glob(f'{train_root}/SAR/VH/*.png')
imgs_sarvv = glob(f'{train_root}/SAR/VV/*.png')
imgs_gt.sort()
imgs_cloudy.sort()
imgs_sarvh.sort()
imgs_sarvv.sort()
idx_train = list(range(len(imgs_cloudy)))

test_root = f'/home/liuwang/liuwang_data/documents/datasets/restoration/cloudremove/test'
imgs_cloudy_test = glob(f'{test_root}/opt_cloudy/*.png')
imgs_sarvh_test = glob(f'{test_root}/SAR/VH/*.png')
imgs_sarvv_test = glob(f'{test_root}/SAR/VV/*.png')
imgs_cloudy_test.sort()
imgs_sarvh_test.sort()
imgs_sarvv_test.sort()
idx_test = list(range(len(imgs_cloudy_test)))

while True:
    i = random.choice(idx_train)
    cloudy = cv2.imread(imgs_cloudy[i])
    sarvh = cv2.imread(imgs_sarvh[i], flags=cv2.IMREAD_UNCHANGED)
    sarvv = cv2.imread(imgs_sarvv[i], flags=cv2.IMREAD_UNCHANGED)
    sarvh2 = np.stack([sarvh] * 3, axis=2)
    sarvv2 = np.stack([sarvv] * 3, axis=2)
    gt = cv2.imread(imgs_gt[i])
    imall = np.concatenate([cloudy, sarvh2, sarvv2, gt], axis=1)
    p = detect_clouds(cloudy)
    cv2.imshow(f'train', imall.astype(np.uint8))
    print('train', os.path.basename(imgs_cloudy[i]), p)

    j = random.choice(idx_test)
    cloudy_test = cv2.imread(imgs_cloudy_test[j])
    sarvh_test = cv2.imread(imgs_sarvh_test[j], flags=cv2.IMREAD_UNCHANGED)
    sarvv_test = cv2.imread(imgs_sarvv_test[j], flags=cv2.IMREAD_UNCHANGED)
    sarvh2_test = np.stack([sarvh_test] * 3, axis=2)
    sarvv2_test = np.stack([sarvv_test] * 3, axis=2)
    imall_test = np.concatenate([cloudy_test, sarvh2_test, sarvv2_test], axis=1)

    p = detect_clouds(cloudy_test, False)

    cv2.imshow(f'test', imall_test.astype(np.uint8))
    print('test', os.path.basename(imgs_cloudy_test[j]), p)

    if cv2.waitKey(0) == ord('q'):
        exit(0)
