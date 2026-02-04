import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from itertools import chain
from pytorch_wavelets import DWTForward, DWTInverse

from mdistiller import normalize, kd_loss
from .base import Distiller
from .FreKD import closest_square_factors
from .DIST import pearson_correlation
from .New import mutual_info_loss
from .CKA import CenterKernelAlignment
from .FreKD import Fre_Correlation


def orthogonal_loss(features):
    """
    features: List of [B, D], 每个来自一个映射器
    """
    loss = 0.0
    num_pairs = 0
    for i in range(len(features)):
        feature1 = features[i] - features[i].mean(dim=1, keepdim=True)
        for j in range(i + 1, len(features)):
            feature2 = features[j] - features[j].mean(dim=1, keepdim=True)
            sim = F.cosine_similarity(feature1, feature2, dim=1)  # [B]
            loss += sim.pow(2).mean()
            num_pairs += 1
    return loss / num_pairs


# def orthogonal_logit(logits, temp=0.2):
#     loss = 0.0
#     num_pairs = 0
#     for i in range(len(logits)):
#         for j in range(i + 1, len(logits)):
#             # a = logits[i] - logits[i].mean(dim=1, keepdim=True)
#             # b = logits[j] - logits[j].mean(dim=1, keepdim=True)
#             # a = F.normalize(a, dim=1)
#             # b = F.normalize(b, dim=1)
#             # sim = torch.matmul(a.T, b) / temp
#             # labels = torch.arange(logits[i].size(1), device=a.device)
#             # loss += F.cross_entropy(sim, labels)
#             a = logits[i]
#             b = logits[j]
#             a = F.normalize(a, dim=1)
#             b = F.normalize(b, dim=1)
#             gram = torch.matmul(a.T, b) / a.size(0)  # (C, C)
#             I = torch.eye(gram.size(0), device=gram.device)
#             loss += torch.norm(gram - I, p='fro') ** 2
#             num_pairs += 1
#     return loss / num_pairs

def orthogonal_logit(logits, temp=0.2):
    loss = 0.0
    num_pairs = 0
    for i in range(len(logits)):
        for j in range(i + 1, len(logits)):
            a = logits[i]  # B * C
            b = logits[j]  # B * C
            a = F.normalize(a, dim=0)
            b = F.normalize(b, dim=0)
            # gram = torch.matmul(a.T, b) / a.size(0) # (C, C)
            gram = torch.matmul(a.T, b) # (C, C)
            I = torch.eye(gram.size(0), device=gram.device)
            # loss += (torch.norm(gram - I, p='fro') ** 2 ) / a.size(0)
            # loss += (torch.norm(gram - I, p='fro') ** 2) / a.size(0)
            loss += torch.norm(gram - I, p='fro')
            num_pairs += 1
    return loss / num_pairs


def coherence(feat_student, feat_teacher):
    F_teacher = torch.fft.fft(feat_teacher, dim=-1)  # [B, D], complex
    F_student = torch.fft.fft(feat_student, dim=-1)

    # 功率谱
    P_tt = (F_teacher * F_teacher.conj()).real.mean(dim=0)  # [D]
    P_ss = (F_student * F_student.conj()).real.mean(dim=0)  # [D]

    # 互谱
    P_st = (F_student * F_teacher.conj()).mean(dim=0)  # [D], complex

    coherence = (P_st.abs() ** 2) / (P_ss * P_tt + 1e-6)
    # coherence_no_dc = coherence[1:]

    return coherence.mean()



class Reg(nn.Module):
    """Linear regressor"""
    def __init__(self, dim_in=1024, dim_out=1024):
        super(Reg, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x

from ._common import ConvReg, get_feat_shapes

class PEFD(Distiller):
    def __init__(self, student, teacher, cfg, **kwargs):
        super(PEFD, self).__init__(student,teacher)
        self.temp = cfg.PEFD.TEMPERATURE
        self.ce_weight = cfg.PEFD.LOSS.CE_WEIGHT
        self.ot_weight = cfg.PEFD.LOSS.OT_WEIGHT
        self.iters = cfg.PEFD.ITERS
        s_feat_shape, t_feat_shape = get_feat_shapes(student, teacher, input_size=(32, 32), feat_type='pooled_feat')
        _, s_channel = s_feat_shape
        _, t_channel = t_feat_shape
        print('s_channels', s_channel)
        print('t_channels', t_channel)
        self.regs = nn.ModuleList([
            Reg(dim_in=s_channel, dim_out=t_channel)
            for _ in range(self.iters)
        ])
        self.regs.apply(self.init_weights)

    def get_learnable_parameters(self):
        reg_params = chain(*(reg.parameters() for reg in self.regs))
        return list(super().get_learnable_parameters()) + list(reg_params)

    def get_extra_parameters(self):
        num_params = sum(p.numel() for p in self.regs.parameters())
        print(f"Extra parameters amount: {num_params} (approx {num_params / 1e6:.2f}M)")

    def forward_train(self, image, target, **kwargs):
        logit_student, features_student = self.student(image)

        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)

        loss_ce = F.cross_entropy(logit_student, target)

        feat_student = features_student['pooled_feat']
        f_t = features_teacher['pooled_feat']

        f_s_all = [reg(feat_student) for reg in self.regs]  # 原始映射

        f_s = sum(f_s_all) / len(f_s_all)

        feat_loss = (1 - F.cosine_similarity(f_s, f_t)).mean()

        loss_dict = {
                'loss_ce': self.ce_weight * loss_ce,
                'feat_loss': 25 * feat_loss,
            }
        return logit_student, loss_dict

