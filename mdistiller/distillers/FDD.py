import torch
import torch.nn as nn
from torch.nn import Module

from ._base import Distiller
from ._common import *

import math
import numpy as np
import random
from .WKD import adaptive_avg_std_pool2d, wkd_feature_loss
from pytorch_wavelets import DWTForward, DWTInverse

# def Normalize(feat):
#     return (feat - feat.mean(dim=0)) / (feat.std(dim=0, unbiased=False) + 1e-6)
#
#
# def decorrelation(feats_stu, feats_tea, kappa=0.01):
#     B, D = feats_stu.shape[0], feats_stu.shape[1]
#     feats_stu_norm, feats_tea_norm = Normalize(feats_stu), Normalize(feats_tea) # normalise along batch dim; [B,D]
#     rcc = torch.mm(feats_tea_norm.T, feats_stu_norm) / B # representation cross-correlation matrix; [D,D]
#     idt = torch.eye(D, device=rcc.device) # identity matrix [D,D]
#     loss_rsd = (rcc - idt).pow(2) # information maximisation
#     loss_rsd[(1 - idt).bool()] *= kappa # decorrelation
#     return loss_rsd.sum(dim=-1).mean(dim=0)


class InverseBottleBlock(nn.Module):
    def __init__(self, s_shape, t_shape, scale=1.0):
        super(InverseBottleBlock, self).__init__()
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape

        in_channel = s_C
        out_channel = t_C

        hidden_channels = int(scale * out_channel)
        self.conv1 = nn.Conv2d(in_channel, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(hidden_channels, out_channel, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv3(x)
        x = self.relu(x)
        return x



def fast_adaptive_std_pool2d(input_tensor, out_size=(1, 1), eps=1e-5):
    """
    快速版本：利用 E[x^2] - E[x]^2 公式并行计算 std
    """
    # 1. 计算 E[x] (均值的池化)
    mean = F.adaptive_avg_pool2d(input_tensor, out_size)

    # 2. 计算 E[x^2] (平方的均值池化)
    mean_sq = F.adaptive_avg_pool2d(input_tensor ** 2, out_size)

    # 3. 计算方差 Var = E[x^2] - (E[x])^2
    # 使用 relu 确保方差非负（防止浮点数精度误差导致微小的负数）
    var = F.relu(mean_sq - mean ** 2)

    # 4. 计算标准差
    std = torch.sqrt(var + eps)

    return std


class FDD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(FDD, self).__init__(student, teacher)
        self.cfg = cfg

        self.feature_loss_weight = cfg.FDD.LOSS.FEAT_WEIGHT
        self.loss_cosine_decay_epoch = cfg.FDD.LOSS.COSINE_DECAY_EPOCH

        # WKD-F: WD for feature distillation

        self.low_high_ratio = cfg.FDD.LOW_HIGH_RATIO
        self.eps = cfg.FDD.EPS
        self.mask_ratio = cfg.FDD.MASK_RATIO

        feat_s_shapes, feat_t_shapes = get_feat_shapes(
            self.student, self.teacher, cfg.FDD.INPUT_SIZE)
        print('feat_student_shapes', feat_s_shapes[-1])
        print('feat_teacher_shapes', feat_t_shapes[-1])

        self.hint_layer = cfg.FDD.HINT_LAYER
        self.projector = cfg.FDD.PROJECTOR
        self.spatial_grid = 1
        if self.projector == "bottleneck":
            self.conv_reg = ConvRegBottleNeck(
                feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer], c_hidden=256, use_relu=True, use_bn=True
            )
        elif self.projector == "conv1x1":
            self.conv_reg = ConvReg(
                feat_s_shapes[self.hint_layer], feat_t_shapes[self.hint_layer], use_relu=True, use_bn=True
            )
        else:
            raise NotImplementedError(f"Unknown projector type: {self.projector}")

        self.teacher = self.teacher.eval()
        self.student = self.student.eval()
        self.dwt = DWTForward(J=1, wave="haar", mode='zero')
        self.idwt = DWTInverse(wave="haar", mode='zero')

    def get_learnable_parameters(self):
        student_params = [v for k, v in self.student.named_parameters()]
        return student_params + list(self.conv_reg.parameters())

    def get_teacher_decouple_feat(self, x, mask_ratio=[0.05, 0.08, 0.10]):
        B, C, H, W = x.shape
        mask_ratio = mask_ratio if isinstance(mask_ratio, (float, int)) else random.choice(mask_ratio)

        spectrum = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        magnitude = spectrum.abs()  # (B, C, H, W//2 + 1)

        magnitude_mean = torch.mean(magnitude, dim=1, keepdim=True)  #
        magnitude_flat = magnitude_mean.view(B, -1)
        threshold = torch.quantile(magnitude_flat, q=1 - mask_ratio, dim=-1, keepdim=True)  # B * 1

        threshold = threshold.unsqueeze(-1).unsqueeze(-1)
        mask = (magnitude_mean >= threshold)  # (B, 1, H, W_fft)  W_fft = W // 2 + 1

        recon_x = torch.where(mask, spectrum, 0.)
        com_x = torch.fft.irfft2(recon_x, s=(H, W), dim=(-2, -1), norm='ortho')
        r_com = x - com_x
        return mask, com_x, r_com


    def get_student_decouple_feat(self, x, mask):
        B, C, H, W = x.shape

        fft_x = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')
        recon_x = fft_x * mask
        com_x = torch.fft.irfft2(recon_x, s=(H, W), dim=(-2, -1), norm='ortho')
        r_com = x - com_x
        return com_x, r_com

    def get_extra_parameters(self):
        return 0

    def magnitude_loss(self, s_feat, t_feat):
        s_fft = torch.fft.rfft2(s_feat, norm='ortho')
        t_fft = torch.fft.rfft2(t_feat, norm='ortho')
        s_mag = s_fft.abs()
        t_mag = t_fft.abs()
        loss_map = F.mse_loss(s_mag, t_mag, reduction='none')
        loss_channel = loss_map.sum(dim=1)
        loss = loss_channel.mean()
        return loss

    def std_loss(self, x, y, reduction='mean'):
        x_std = torch.sqrt(x.var(dim=(-1, -2), unbiased=False) + 1e-6)
        y_std = torch.sqrt(y.var(dim=(-1, -2), unbiased=False) + 1e-6)
        diff = (x_std - y_std).pow(2)  # [B,C]
        if reduction == 'mean':
            return diff.mean()
        elif reduction == 'sum_channel':
            return diff.sum(dim=1).mean()
        elif reduction == 'none':
            return diff
        else:
            raise ValueError(reduction)

    def forward_train(self, image, target, **kwargs):
        with torch.cuda.amp.autocast():
            logits_student, feats_student = self.student(image)
            with torch.no_grad():
                logits_teacher, feats_teacher = self.teacher(image)

        logits_student = logits_student.to(torch.float32)
        loss_ce = F.cross_entropy(logits_student, target)

        decay_start_epoch = self.loss_cosine_decay_epoch
        if kwargs['epoch'] > decay_start_epoch:
            # cosine decay
            self.feature_loss_weight_1 = 0.5 * self.feature_loss_weight * (1 + math.cos(
                (kwargs['epoch'] - decay_start_epoch) / (self.cfg.SOLVER.EPOCHS - decay_start_epoch) * math.pi))
        else:
            self.feature_loss_weight_1 = self.feature_loss_weight

        f_t = feats_teacher["feats"][self.hint_layer].to(torch.float32)
        f_s = feats_student["feats"][self.hint_layer].to(torch.float32)

        f_s = self.conv_reg(f_s)

        mask, t_low, t_high = self.get_teacher_decouple_feat(f_t, mask_ratio=self.mask_ratio)
        # mask, t_low, t_high, high_ratio = self.get_teacher_decouple_feat(f_t, mask_ratio=[0.05, 0.1, 0.15])
        s_low, s_high = self.get_student_decouple_feat(f_s, mask)

        low_loss = F.mse_loss(s_low, t_low, reduction='sum')
        low_loss = low_loss / (s_low.size(0) * s_low.size(2) * s_low.size(3))

        # high_std = self.std_loss(f_t, f_s, reduction='none')
        # high_loss = high_std.sum(dim=-1).mean()

        s_rms = torch.sqrt(s_high.pow(2).mean(dim=(-1, -2)) + 1e-6)
        t_rms = torch.sqrt(t_high.pow(2).mean(dim=(-1, -2)) + 1e-6)
        rms_diff = F.mse_loss(s_rms, t_rms, reduction='none')
        rms_loss = rms_diff.sum(dim=-1).mean()

        feat_loss = self.low_high_ratio * low_loss + rms_loss
        loss_kd = self.feature_loss_weight_1 * feat_loss

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
