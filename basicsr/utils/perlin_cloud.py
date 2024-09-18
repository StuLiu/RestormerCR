"""
@Project : crgame
@File    : berlin_clod.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/8/15 下午4:57
@e-mail  : 1183862787@qq.com
"""

import numpy as np
import pylab as plt
from pythonperlin import perlin
import cv2
import time
import random
from tqdm import tqdm


def cloud_gen_perlin(img):

    # Set density - output shape will be dens * shape = (128,128)
    size = img.shape[:2]
    dens = random.choice([16, 32, 64, 128])
    # Set grid shape for randomly seeded gradients
    shape = (size[0] // dens, size[1] // dens)

    # Generate noise
    x = perlin(shape, dens=dens, seed=int(time.time()% 1000007)) * dens / 32
    x = np.stack([x] * 3, axis=2)
    img_cloudy = np.where(x > 0, img + x * 255, img)
    img_cloudy = np.clip(img_cloudy, 0, 255)

    # # Test that noise tiles seamlessly
    # print(dens, shape)
    # cv2.imshow('', img_cloudy.astype(np.uint8))
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #     exit(0)
    return img_cloudy


if __name__ == '__main__':
    img = cv2.imread('zzz.png')
    for _ in tqdm(range(10000)):
        cloud_gen_perlin(img[:128, :128, :])