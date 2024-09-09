'''
@Project : Restormer 
@File    : cr_metrics.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2024/9/3 下午1:10
@e-mail  : 1183862787@qq.com
'''
import cv2
import numpy as np
import skimage.metrics
import imgvision
import torch
import torch.nn.functional as F

from torch.autograd import Variable
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def calculate_ssim_crgame(img1,
                          img2,
                          crop_border,
                          input_order='HWC',
                          test_y_channel=False):
    if not isinstance(img1, torch.Tensor):
        img1 = torch.from_numpy(img1.astype(np.float32) / 255.0).permute(2, 0 ,1).unsqueeze(dim=0)
    if not isinstance(img2, torch.Tensor):
        img2 = torch.from_numpy(img2.astype(np.float32) / 255.0).permute(2, 0 ,1).unsqueeze(dim=0)
    img1 = img1.cuda()
    img2 = img2.cuda()
    return ssim(img1, img2).cpu()


def correlation_coefficient(imagesA, imagesB, eps=1e-7):
    """
    Get correlation coefficient in image level in a mini-batch.
    Args:
        imagesA: predicted images, torch.Tensor, range(0, 1), shape=(bs, c, h, w)
        imagesB: target images, torch.Tensor, range(0, 1), shape=(bs, c, h, w)
        eps: eps for avoiding zero dividing

    Returns:
        cc: correlation coefficient for the inputted images. range (-1, 1)
    """

    # 确保输入都是四维
    if imagesA.ndim != 4 or imagesB.ndim != 4:
        raise ValueError("Input images must have shape (bs, c, h, w).")

    # 检查输入中是否有 NaN 或 inf 值
    if torch.isnan(imagesA).any() or torch.isinf(imagesA).any():
        print(torch.sum(torch.isnan(imagesA)), torch.sum(torch.isinf(imagesA)))
        raise ValueError("Input images imagesA must not contain NaN or inf values.")
    if torch.isnan(imagesB).any() or torch.isinf(imagesB).any():
        print(torch.sum(torch.isnan(imagesB)), torch.sum(torch.isinf(imagesB)))
        raise ValueError("Input images imagesB must not contain NaN or inf values.")

    bs, c, h, w = imagesA.shape

    # 将图像展平为 (bs, c, h*w)
    imagesA_flat = imagesA.view(bs, c, -1)  # 形状 (bs, c, h*w)
    imagesB_flat = imagesB.view(bs, c, -1)  # 形状 (bs, c, h*w)

    # 计算均值
    meanA = imagesA_flat.mean(dim=2, keepdim=True)  # 形状 (bs, c, 1)
    meanB = imagesB_flat.mean(dim=2, keepdim=True)  # 形状 (bs, c, 1)

    # 计算中心化图像
    centered_A = imagesA_flat - meanA  # 形状 (bs, c, h*w)
    centered_B = imagesB_flat - meanB  # 形状 (bs, c, h*w)

    # 计算CC
    up = torch.sum(centered_A * centered_B, dim=2)  # 形状 (bs, c)

    # 计算分母，确保不为0
    sum_sq_A = torch.sum(centered_A ** 2, dim=2)  # (bs, c)
    sum_sq_B = torch.sum(centered_B ** 2, dim=2)  # (bs, c)

    down = torch.sqrt(sum_sq_A * sum_sq_B)  # 形状 (bs, c)

    # 使用 torch.clamp 确保 down 不为0
    down = torch.clamp(down, min=eps)

    # 计算相关系数
    cc = up / down

    return cc.mean()


def calculate_cc_numpy(img1,
                 img2,
                 crop_border,
                 input_order='HWC',
                 test_y_channel=False):
    if type(img1) == torch.Tensor:
        if len(img1.shape) == 4:
            img1 = img1.squeeze(0)
        c = img1.shape[0]
        img1 = img1.detach().cpu().numpy().view(c, -1)
    else:
        if input_order == 'HWC':
            c = img1.shape[-1]
            img1 = img1.transpose(2, 0, 1).reshape(c, -1)
        else:
            c = img1.shape[0]
            img1 = img1.reshape(c, -1)

    if type(img2) == torch.Tensor:
        if len(img2.shape) == 4:
            img2 = img2.squeeze(0)
        c = img2.shape[0]
        img2 = img2.detach().cpu().numpy().view(c, -1)
    else:
        if input_order == 'HWC':
            c = img2.shape[-1]
            img2 = img2.transpose(2, 0, 1).reshape(c, -1)
        else:
            c = img2.shape[0]
            img2 = img2.reshape(c, -1)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    sm = imgvision.spectra_metric(img1, img2)
    return sm.CC('')


def calculate_cc_crgame(img1,
                    img2,
                    crop_border,
                    input_order='HWC',
                    test_y_channel=False):
    if not isinstance(img1, torch.Tensor):
        img1 = torch.from_numpy(img1.astype(np.float32) / 255.0).permute(2, 0 ,1).unsqueeze(dim=0)
    if not isinstance(img2, torch.Tensor):
        img2 = torch.from_numpy(img2.astype(np.float32) / 255.0).permute(2, 0 ,1).unsqueeze(dim=0)
    if img1.ndim == 3:
        img1 = img1.unsqueeze(dim=0)
    if img2.ndim == 3:
        img2 = img2.unsqueeze(dim=0)
    img1 = img1.cuda()
    img2 = img2.cuda()
    return correlation_coefficient(img1, img2).cpu()


def calculate_score_crgame(
        img1,
        img2,
        crop_border=0,
        input_order='HWC',
        test_y_channel=False
):
    return (calculate_cc_crgame(img1, img2, crop_border, input_order, test_y_channel) +
            calculate_ssim_crgame(img1, img2, crop_border, input_order, test_y_channel)) / 2