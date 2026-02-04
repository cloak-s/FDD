import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from itertools import chain
from pytorch_wavelets import DWTForward, DWTInverse

from mdistiller import normalize, kd_loss
from .base import Distiller
from ._common import DCT, dct_2d, idct_2d, ConvReg, get_feat_shapes

# 2023--ICCV A Simple and Generic Framework for Feature Distillation via Channel-wise Transformation


class Projector(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_shape, t_shape):
        super(Projector, self).__init__()
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(t_C, t_C, kernel_size=1, stride=1, padding=0)


    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class CWT(Distiller):
    def __init__(self, student, teacher, cfg, **kwargs):
        super(CWT, self).__init__(student,teacher)
        s_feat_shape, t_feat_shape = get_feat_shapes(student, teacher, input_size=(32, 32))
        self.proj = Projector(s_feat_shape[-1], t_feat_shape[-1])


    def get_learnable_parameters(self):
        return list(super().get_learnable_parameters()) + list(self.proj.parameters())

    def get_extra_parameters(self):
        num_params = sum(p.numel() for p in self.proj.parameters())
        print(f"Extra parameters amount: {num_params} (approx {num_params / 1e6:.2f}M)")

    def forward_train(self, image, target, **kwargs):
        logit_student, features_student = self.student(image)

        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)


        loss_ce = F.cross_entropy(logit_student, target)

        f_s = features_student['feats'][-1]
        f_t = features_teacher['feats'][-1]

        f_s_trans = self.proj(f_s)

        diff = (f_s_trans - f_t).pow(2)

        feat_loss = diff.sum()/ f_s.size(0)

        loss_dict = {
            'loss_ce': loss_ce,
            'feat_loss': 5e-3 * feat_loss,
        }
        return logit_student, loss_dict

#  7e-4  3e-4 5e-3

