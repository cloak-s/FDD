import numpy as np
import torch
import math
from torch import nn

from mdistiller import pearson_correlation
from .base import Distiller
import torch.nn.functional as F
from .Loss import kd_loss, normalize, center_kernel
from pytorch_wavelets import DWTForward, DWTInverse, DWT1D, IDWT1D
import math
from .New import mutual_info_loss


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2, dim = 1):
        super(Normalize, self).__init__()
        self.power = power
        self.dim = dim
    def forward(self, x):
        norm = x.pow(self.power).sum(self.dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Embed(nn.Module):
    def __init__(self, dim_in=256, dim_out=128):
        super(Embed, self).__init__()
        self.conv2d = nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.l2norm = nn.BatchNorm2d(dim_out)#Normalize(2)
    def forward(self, x):
        x = self.conv2d(x)
        x = self.l2norm(x)
        return x


# 2021 ICCV

class ICKD(Distiller):
    def __init__(self, student, teacher, cfg, **kwargs):
        super(ICKD, self).__init__(student,teacher)
        self.temp = cfg.KD.TEMPERATURE
        s_channels = self.student.get_stage_channels()
        t_channels = self.teacher.get_stage_channels()

        self.proj = Embed(dim_in=s_channels[-1], dim_out=t_channels[-1])

        self.warmup_epochs = cfg.REVIEWKD.WARMUP_EPOCHS

        self.teacher_cls = self.teacher.get_classifier()

    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.proj.parameters())


    def forward_train(self, image, target, **kwargs):
        logit_student, features_student = self.student(image)

        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)
            # feats["feats"] = [f0, f1, f2, f3]
            # feats["preact_feats"] = [f0, f1_pre, f2_pre, f3_pre]
            # feats["pooled_feat"] = avg

        loss_ce = F.cross_entropy(logit_student, target)
        # loss_kd = kd_loss(logit_student, logits_teacher.detach(), self.temp)

        feat_student = features_student['feats'][-1]
        feat_teacher = features_teacher['feats'][-1]

        f_s = self.proj(feat_student)
        B, C, H, W = f_s.shape

        f_s = f_s.view(B, C, -1)
        f_t = feat_teacher.view(B, C, -1)

        emd_s = torch.bmm(f_s, f_s.permute(0, 2, 1))
        emd_s = torch.nn.functional.normalize(emd_s, dim=2)

        emd_t = torch.bmm(f_t, f_t.permute(0, 2, 1))
        emd_t = torch.nn.functional.normalize(emd_t, dim=2)

        G_diff = emd_s - emd_t

        # feat_loss =  min(kwargs["epoch"] / self.warmup_epochs, 1.0) * (G_diff * G_diff).view(B, -1).sum() / (B*C)
        feat_loss = (G_diff * G_diff).view(B, -1).sum() / (B * C)

        logit_loss = kd_loss(logit_student, logits_teacher, temperature=self.temp)

        loss_dict = {
            'loss_ce': loss_ce,
            'logit_loss': logit_loss,
            'feat_loss': 2.5 * feat_loss,
        }
        return logit_student, loss_dict

