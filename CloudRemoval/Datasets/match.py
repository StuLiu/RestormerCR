'''
@Project : Restormer 
@File    : match.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2024/9/7 下午2:13
@e-mail  : 1183862787@qq.com
'''
import cv2
import numpy as np

# 加载图像
image1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)

# 初始化ORB检测器
orb = cv2.ORB_create()

# 检测特征点与描述符
kp1, des1 = orb.detectAndCompute(image1, None)
kp2, des2 = orb.detectAndCompute(image2, None)

# 创建 BFMatcher 对象
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 进行匹配
matches = bf.match(des1, des2)

# 绘制前N个匹配项
matches = sorted(matches, key=lambda x: x.distance)
matched_image = cv2.drawMatches(image1, kp1, image2, kp2, matches[:20], None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 找到单应性矩阵
if len(matches) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is not None:
        print("找到重叠区域")
    else:
        print("未找到重叠区域")
else:
    print("不足以找到单应性矩阵")

# 展示匹配的结果
cv2.imshow('Matches', matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
