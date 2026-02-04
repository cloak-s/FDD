import numpy as np
import torch
import math
from torch import nn
from .base import Distiller
import torch.nn.functional as F
from .Loss import kd_loss, normalize, CenterKernelAlignment
from pytorch_wavelets import DWTForward, DWTInverse
import math
from .DIST import pearson_correlation

def logit_decorrelation_loss(logits, eps=1e-6):
    """
    logits: (B, C) 学生模型输出 (未softmax的logit)
    返回一个正则项，鼓励logit在类别维度上去相关，提升类间可分性
    """
    B, C = logits.shape
    # 标准化（零均值 + 单位方差）
    z = logits - logits.mean(dim=0, keepdim=True)  # (B, C)
    z = F.normalize(z, dim=0, eps=eps)

    # 相关性矩阵 (C x C)
    corr = (z.T @ z)  # 协方差近似
    off_diag = corr - torch.diag(torch.diag(corr))

    loss = (off_diag ** 2).sum() / (C * (C - 1))
    return loss


def CenterKernelAlignment(X, Y, with_l2_norm):
    """Compute the CKA similarity betweem samples"""
    # Compute Gram matrix
    gram_X = torch.matmul(X, X.t())
    gram_Y = torch.matmul(Y, Y.t())

    # l2 norm or not
    if with_l2_norm:
        gram_X = gram_X / torch.sqrt(torch.diag(gram_X)[:, None])
        gram_Y = gram_Y / torch.sqrt(torch.diag(gram_Y)[:, None])

    # compute cka
    cka = torch.trace(torch.matmul(gram_X, gram_Y.t())) / torch.sqrt(
        torch.trace(torch.matmul(gram_X, gram_X.t())) * torch.trace(torch.matmul(gram_Y, gram_Y.t()))
    )

    return cka

#
# def cka_loss(teacher_logits, student_logits, with_l2_norm):
#     """Compute the CKA similarity between samples
#     CKA computes similarity between batches
#     input: (N, P) ----> output: (N, N) similarity matrix
#     """
#     N_t = teacher_logits.shape[0]
#     N_s = student_logits.shape[0]
#     assert N_s == N_t  # when use cka, you need to make sure N the same
#
#     # get a similarity score between teacher and student
#     similarity_martix = CenterKernelAlignment(teacher_logits, student_logits, with_l2_norm)
#
#     # maximize the likelihood of it
#     return -similarity_martix

def Center_Kernel_Alignment(student, teacher, eps=1e-6):
    """
    优化后的线性 CKA 计算 (Linear CKA)
    复杂度: O(N^2) instead of O(N^3)
    """
    # 1. 计算 Gram 矩阵 (B, B)
    K = torch.matmul(student, student.t())
    L = torch.matmul(teacher, teacher.t())

    # 2. 高效中心化 (Centering without Matrix Multiplication)
    # 公式: K' = K - row_mean - col_mean + global_mean
    # 因为 Gram 矩阵是对称的，所以 row_mean == col_mean
    def center_gram(G):
        mean_row = G.mean(dim=1, keepdim=True)
        mean_all = G.mean()
        # 利用广播机制进行中心化，避免了构建巨大的 H 矩阵和矩阵乘法
        return G - mean_row - mean_row.t() + mean_all

    K_centered = center_gram(K)
    L_centered = center_gram(L)

    # 3. 计算 HSIC (分子)
    # 优化点: trace(A @ B) 等价于 sum(A * B.T)
    # 因为 K, L 都是对称矩阵，所以直接 sum(K_centered * L_centered)
    hsic = torch.sum(K_centered * L_centered)

    norm_k = torch.sqrt(torch.sum(K_centered * K_centered) + eps)
    norm_l = torch.sqrt(torch.sum(L_centered * L_centered) + eps)

    # 5. 计算 CKA
    cka = hsic / (norm_k * norm_l + eps)

    return cka


def centered_kernel_matrix(x):
    n = x.size()[0]
    H = torch.eye(n, device=x.device, dtype=x.dtype) - torch.ones((n, n), device=x.device, dtype=x.dtype) / n
    return H @ x @ H


def HSIC(K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """
    计算两个核矩阵的 Hilbert-Schmidt Independence Criterion (HSIC)
    Args:
        K, L: 两个核矩阵，形状均为 (n, n)
    Returns:
        HSIC 值（标量）
    """
    n = K.size(0)
    # 计算中心化核矩阵
    K_centered = centered_kernel_matrix(K)
    L_centered = centered_kernel_matrix(L)
    return torch.trace(K_centered @ L_centered) / (n - 1) ** 2

def CKA_align(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算两个核矩阵的 Centered Kernel Alignment (CKA)
    Args:
        K, L: 两个核矩阵，形状均为 (n, n)
    Returns:
        CKA 值（标量），范围 [0, 1]
    """
    # 线性核函数
    K = x @ x.T
    L = y @ y.T

    hsic_kl = HSIC(K, L)
    hsic_kk = HSIC(K, K)
    hsic_ll = HSIC(L, L)

    # return hsic_kl / torch.sqrt(torch.maximum(hsic_kk * hsic_ll, torch.tensor(1e-6, device=x.device)))
    return hsic_kl / torch.sqrt(hsic_kk * hsic_ll)




from ._common import ConvReg, get_feat_shapes
import random

class CKA(Distiller):
    def __init__(self, student, teacher, cfg, **kwargs):
        super(CKA, self).__init__(student,teacher)
        s_channels = self.student.get_stage_channels()
        t_channels = self.teacher.get_stage_channels()

        s_feat_shape, t_feat_shape = get_feat_shapes(student, teacher, input_size=(32, 32))
        self.proj = ConvReg(s_feat_shape[-1], t_feat_shape[-1])
        print('feat_student_channel_shape', s_feat_shape[-1])
        print('feat_teacher__channel_shape', t_feat_shape[-1])

    def get_learnable_parameters(self):
        return (super().get_learnable_parameters()
                + list(self.proj.parameters())
                )


    def forward_train(self, image, target, **kwargs):
        logit_student, features_student = self.student(image)

        with torch.no_grad():
            logits_teacher, features_teacher = self.teacher(image)
            # feats["feats"] = [f0, f1, f2, f3]
            # feats["preact_feats"] = [f0, f1_pre, f2_pre, f3_pre]
            # feats["pooled_feat"] = avg

        loss_ce = F.cross_entropy(logit_student, target)

        # feat_student = features_student['feats'][-1]
        # feat_teacher = features_teacher['feats'][-1]

        # feat_student = self.proj(feat_student)

        cka_loss = 0.
        for f_s, f_t in zip(features_student['feats'], features_teacher['feats']):
            f_s_flat = torch.flatten(f_s, start_dim=1) # B * N
            f_t_flat = torch.flatten(f_t, start_dim=1)
            cka_loss += (1 - CenterKernelAlignment(f_s_flat, f_t_flat, with_l2_norm=True))

        # feat_student_flat = torch.flatten(feat_student, start_dim=1) # B * N
        # feat_teacher_flat = torch.flatten(feat_teacher, start_dim=1)
        # # feat_loss = CenterKernelAlignment(feat_student_flat, feat_teacher_flat, with_l2_norm=True)
        # feat_loss = CKA_align(feat_student_flat, feat_teacher_flat)

        feat_loss = cka_loss

        loss_dict = {
            'loss_ce': loss_ce,
            'feat_loss': 25 * feat_loss,
            # 'dis_loss': 1 * dis_loss,
        }
        return logit_student, loss_dict

