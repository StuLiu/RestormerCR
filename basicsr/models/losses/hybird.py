'''
@Project : Restormer 
@File    : hybird.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2024/9/1 下午9:54
@e-mail  : 1183862787@qq.com
'''
import torch
import torch.nn as nn
from basicsr.models.losses.ssim_tensor import SSIM


def pearsonr(x, y, batch_first=True):
    assert x.shape == y.shape

    if batch_first:
        dim = -1
    else:
        dim = 0

    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)

    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)

    corr = bessel_corrected_covariance / ((x_std * y_std) + 1e-5)

    return corr


class CCLoss(nn.Module):

    def __init__(self, eps=1e-7, keep_batch=False):
        super().__init__()
        self.eps = eps
        self.keep_batch = keep_batch

    def forward(self, predicted, target):
        """
        Get pearsonr correlation coefficient in image level in a mini-batch.
        Args:
            imagesA: predicted images, torch.Tensor, range(0, 1), shape=(bs, c, h, w)
            imagesB: target images, torch.Tensor, range(0, 1), shape=(bs, c, h, w)
            eps: eps for avoiding zero dividing
        Returns:
            pcc: correlation coefficient for the inputted images. range (-1, 1)
        """
        bs = predicted.size(0)
        img_a = predicted.mean(dim=1).view(bs, -1)
        img_b = target.mean(dim=1).view(bs, -1)
        pcc = pearsonr(img_a, img_b, batch_first=True)
        if not self.keep_batch:
            pcc = pcc.mean()
        else:
            pcc = pcc.squeeze(dim=-1)
        return ((-1 * pcc) + 1) / 2


class PixelWiseFocalL1Loss(nn.Module):
    def __init__(self, gamma=2.0, keep_batch=False, is_cl1=True):
        super(PixelWiseFocalL1Loss, self).__init__()
        self.gamma = gamma
        self.is_cl1 = is_cl1
        self.keep_batch = keep_batch

    def forward(self, pred, target):
        """
        Args:
            pred: restoration results, Tensor, shape=(b, c, h, w)
            target: target image, Tensor, shape=(b, c, h, w)
        Returns:
            ohem_l1: mean l1-loss for those hardest pixels
        """
        if self.is_cl1:
            loss_l1 = torch.sqrt((pred - target) ** 2 + 1e-6).mean(dim=1, keepdim=True)
        else:
            loss_l1 = torch.abs(pred - target).mean(dim=1, keepdim=True)    # (b, 1, h, w)
        pt = torch.exp(-loss_l1.detach())  # 计算 e^(-l1_loss)
        focal_term = (1 - pt) ** self.gamma  # 计算 focal term
        loss_focal = focal_term * loss_l1  # focal l1 loss

        if self.keep_batch:
            return torch.mean(loss_focal, dim=[1, 2, 3])

        return torch.mean(loss_focal)


class SSIMLoss(nn.Module):

    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.ssim_ = SSIM(window_size, size_average)

    def forward(self, pred, target):
        return 1 - self.ssim_(pred, target)


class HybirdV1(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super().__init__()
        self.focall1_fn = PixelWiseFocalL1Loss()
        self.ssim_loss_fn = SSIMLoss()
        self.ccloss_fn = CCLoss()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss_focall1 = self.focall1_fn(pred, target)
        loss_ssim = self.ssim_loss_fn(pred, target)
        loss_cc = self.ccloss_fn(pred, target)
        return self.loss_weight * (loss_focall1 + loss_ssim + loss_cc)


if __name__ == '__main__':
    # 使用示例
    # input_ = torch.randn(4, 3, 256, 256, requires_grad=True)  # 例子中假设输入为 256x256 的图像，batch size 为 4
    # target_ = torch.ones(4, 3, 256, 256)  # 同样假设目标图像
    # l1 = nn.L1Loss()
    # print(l1(input_, target_))
    # # 计算损失
    #
    # ccloss = CCLoss()
    # cc = ccloss(input_, input_ * 2)
    # print(cc)


    # 示例图像，随机生成
    imageA = torch.randn(1, 3, 256, 256).cuda()  # c=3, h=256, w=256
    imageB = torch.randn(1, 3, 256, 256).cuda()  # c=3, h=256, w=256

    ccloss_fn = CCLoss()
    ssim_loss_fn = SSIMLoss()
    focall1_fn = PixelWiseFocalL1Loss()

    print(ccloss_fn(imageA, imageA))
    print(ssim_loss_fn(imageA, imageA))
    print(focall1_fn(imageA, imageB))
