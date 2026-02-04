import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Distiller


def dist_loss(logits_student, logits_teacher, beta=1., gamma=1., temperature=1.):
    y_s = (logits_student / temperature).softmax(dim=1)
    y_t = (logits_teacher / temperature).softmax(dim=1)
    inter_loss = temperature ** 2 * inter_class_relation(y_s, y_t)
    intra_loss = temperature ** 2 * intra_class_relation(y_s, y_t)
    return beta * inter_loss + gamma * intra_loss


def cosine_similarity(a, b, eps=1e-8):
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


def inter_class_relation(y_s, y_t):
    return 1 - pearson_correlation(y_s, y_t).mean()


def intra_class_relation(y_s, y_t):
    return inter_class_relation(y_s.transpose(0, 1), y_t.transpose(0, 1))


class DIST(Distiller):
    def __init__(self, student, teacher, cfg):
        super(DIST, self).__init__(student, teacher)

        self.cfg = cfg
        self.gama = cfg.DIST.ALPHA
        self.beta = cfg.DIST.BETA
        self.temperature = cfg.DIST.T
        self.warmup = cfg.DIST.WARMUP

    def forward_train(self, image, target, **kwargs):
        logit_student, _ = self.student(image)
        with torch.no_grad():
            logit_teacher, _ = self.teacher(image)

        loss_gt = F.cross_entropy(logit_student, target)
        # loss_kd = (min(kwargs["epoch"] / self.warmup, 1.0) *
        #            dist_loss(logit_student, logit_teacher, beta=self.beta, gamma=self.gama , temperature=self.temperature))
        loss_kd = dist_loss(logit_student, logit_teacher, beta=self.beta, gamma=self.gama, temperature=self.temperature)

        losses_dict = {
            "loss_ce": loss_gt,
            "loss_kd": loss_kd,
        }
        return logit_student, losses_dict

