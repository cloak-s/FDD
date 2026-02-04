import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance_matrix
from sympy import resultant
from .base import Distiller
from .Loss import kd_loss


class OrthogonalLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.projector = torch.nn.utils.parametrizations.orthogonal(nn.Linear(in_dim, out_dim, bias=False))
        self.weight = nn.Parameter(torch.ones(out_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

    def forward(self, x):
        x = self.projector(x)
        # scaling_factor = F.relu(self.weight)
        x = self.weight * x
        if self.bias is not None:
            x = x + self.bias
        return x


class VKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super(VKD, self).__init__(student, teacher)

        self.temperature = cfg.VKD.TEMPERATURE
        self.KL_loss_weight = cfg.VKD.LOSS.KD_WEIGHT
        self.CE_loss_weight = cfg.VKD.LOSS.CE_WEIGHT
        self.OT_loss_weight = cfg.VKD.LOSS.OT_WEIGHT

        s_channels = self.student.get_stage_channels()
        t_channels = self.teacher.get_stage_channels()
        self.projector = torch.nn.utils.parametrizations.orthogonal(torch.nn.Linear(s_channels[-1], t_channels[-1], bias=False))
        self.projector.apply(self.init_weights)


    def get_learnable_parameters(self):
        return super().get_learnable_parameters() + list(self.projector.parameters())

    def get_extra_parameters(self):
        num_p = 0
        for p in self.projector.parameters():
            num_p += p.numel()
        return num_p

    def forward_train(self, image, target, **kwargs):
        logits_student, feature_student = self.student(image)
        with torch.no_grad():
            logits_teacher, feature_teacher = self.teacher(image)

        f_s = feature_student['pooled_feat']
        # f_s = f_s - f_s.mean(dim=1, keepdim=True)
        # f_s = F.layer_norm(f_s, (f_s.shape[1],))
        s_ft = self.projector(f_s)

        # t_norm_ft = F.layer_norm(feature_teacher['pooled_feat'], (feature_teacher['pooled_feat'].shape[1], ))
        # f_t = feature_teacher['pooled_feat']
        t_norm_ft = F.layer_norm(feature_teacher['pooled_feat'], (feature_teacher['pooled_feat'].shape[1], ))

        loss_ft = 25 * F.smooth_l1_loss(s_ft, t_norm_ft.detach())
        loss_ce = F.cross_entropy(logits_student, target)
        # loss_kl = 0. * kd_loss(logits_student, logits_teacher.detach(), self.temperature)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_ft,
            # "loss_kl": loss_kl,
        }
        return logits_student, losses_dict

