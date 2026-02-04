import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from itertools import chain
from pytorch_wavelets import DWTForward, DWTInverse

from ._base import Distiller

# 2024 AAAI-- Understanding the role of projector in KD

class OurDistillationLoss(nn.Module):
    def __init__(self, mode='mse', **kwargs):
        super().__init__()
        self.mode = mode

        # r34
        t_dim = 512
        s_dim = 512

        # projector
        self.embed = nn.Linear(s_dim, t_dim).cuda()
        self.bn_s = torch.nn.BatchNorm1d(t_dim, eps=0.0001, affine=False).cuda()
        self.bn_t = torch.nn.BatchNorm1d(t_dim, eps=0.0001, affine=False).cuda()
        # self.gn_s = torch.nn.GroupNorm(16, t_dim, eps=0.0, affine=False).cuda()
        # self.gn_t = torch.nn.GroupNorm(16, t_dim, eps=0.0, affine=False).cuda()

        # not being used at the moment
        if mode not in ('mse', 'bn_mse', 'bn_corr', 'bn_corr_4', 'log_bn_corr_4'):
            raise ValueError('mode `{}` is not expected'.format(mode))

    def forward_loss(self, z_s, z_t):
        f_t = z_t

        f_s = self.embed(z_s)
        n, d = f_s.shape

        f_s_norm = self.bn_s(f_s)
        f_t_norm = self.bn_t(f_t)

        c_diff = f_s_norm - f_t_norm
        c_diff = torch.abs(c_diff)
        c_diff = c_diff.pow(4.0)

        loss = torch.log(c_diff.sum())
        return loss


def get_feat_channel(student, teacher, input_size):
    # # 保存原来的模式
    s_mode = student.training

    # 切到 eval，避免 BN 更新、Dropout 随机
    student.eval()
    teacher.eval()

    # 用确定性输入，避免消耗 RNG
    device = next(student.parameters()).device
    data = torch.zeros(1, 3, *input_size, device=device)

    with torch.no_grad():
        _, feat_s = student(data)
        _, feat_t = teacher(data)

    feat_s_channel = feat_s['pooled_feat']
    feat_t_channel = feat_t['pooled_feat']

    if s_mode: student.train()

    return feat_s_channel, feat_t_channel


class LBL(Distiller):
    def __init__(self, student, teacher, cfg, **kwargs):
        super(LBL, self).__init__(student,teacher)

        feat_s_shape, feat_t_shape = get_feat_channel(student, teacher, cfg.LBL.INPUT_SIZE)

        self.trans = nn.Linear(feat_s_shape.shape[1], feat_t_shape.shape[1])
        self.bn_s = torch.nn.BatchNorm1d(feat_t_shape.shape[1], eps=0.0001, affine=False)
        self.bn_t = torch.nn.BatchNorm1d(feat_t_shape.shape[1], eps=0.0001, affine=False)

        self.trans.apply(self.init_weights)
        self.bn_s.apply(self.init_weights)
        self.bn_t.apply(self.init_weights)

    def get_learnable_parameters(self):
        # reg_params = chain(*(reg.parameters() for reg in self.regs))
        return list(super().get_learnable_parameters()) + list(self.trans.parameters())

    def forward_train(self, image, target, **kwargs):
        logit_student, features_student = self.student(image)

        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)
            # feats["feats"] = [f0, f1, f2, f3]
            # feats["preact_feats"] = [f0, f1_pre, f2_pre, f3_pre]
            # feats["pooled_feat"] = avg

        loss_ce = F.cross_entropy(logit_student, target)

        feat_student = features_student['pooled_feat']
        feat_teacher = features_teacher['pooled_feat']

        trans_student = self.trans(feat_student)
        f_s = self.bn_s(trans_student)
        f_t = self.bn_t(feat_teacher)

        c_diff = f_s - f_t
        c_diff = torch.abs(c_diff)
        c_diff = c_diff.pow(4.0)

        feat_loss = torch.log(c_diff.sum())

        loss_dict = {
            'loss_ce': loss_ce,
            'loss_kd': feat_loss,
        }
        return logit_student, loss_dict


        # dot = (f_s * f_t).sum(dim=-1, keepdim=True)
        # ft_norm2 = (f_t * f_t).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        # alpha = dot / ft_norm2
        # proj = alpha * f_t
        # feat_loss = F.smooth_l1_loss(proj, f_t) + F.smooth_l1_loss(f_s, proj)