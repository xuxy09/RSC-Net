import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(dim=-1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class NTXent(nn.Module):
    """
    Normalized Temperature-scaled cross entropy loss
    """
    def __init__(self, tau, kernel='dot'):
        super(NTXent, self).__init__()
        self.tau = tau
        self.kernel = kernel
        self.l2_norm = Normalize()

    def dot_pos(self, z_1, z_2):
        z_1 = self.l2_norm(z_1)
        z_2 = self.l2_norm(z_2)
        batch_size = z_1.size(0)
        feat_dim = z_1.size(1)
        result = torch.bmm(z_1.view(batch_size, 1, feat_dim), z_2.view(batch_size, feat_dim, 1)).squeeze(-1).squeeze(-1)

        # result = result / math.sqrt(feat_dim)
        return result

    def dot_neg(self, z_1, z_2, z_neg):
        z_1 = self.l2_norm(z_1)
        # concatenate z_2 to z_neg
        z_neg = torch.cat([z_2.unsqueeze(1), z_neg], dim=1)
        z_neg = self.l2_norm(z_neg)
        batch_size = z_1.size(0)
        feat_dim = z_1.size(1)
        num_neg = z_neg.size(1)

        z_1 = z_1.unsqueeze(1) # (B, 1, D)
        z_1 = z_1.repeat(1, z_neg.size(1), 1)  # (B, N, D)
        result = torch.bmm(z_1.view(batch_size * num_neg, 1, feat_dim), z_neg.view(batch_size * num_neg, feat_dim, 1))   # (B*N, 1, 1)
        result = result.view(batch_size, num_neg)

        # result = result / math.sqrt(feat_dim)
        return torch.exp(result / self.tau)

    def mse_pos(self, z_1, z_2):
        # z_1 = self.l2_norm(z_1)
        # z_2 = self.l2_norm(z_2)
        result = - F.mse_loss(z_1, z_2, reduction='none')
        result = result.sum(dim=-1)
        return result

    def mse_neg(self, z_1, z_2, z_neg):
        # concatenate z_2 to z_neg
        z_neg = torch.cat([z_2.unsqueeze(1), z_neg], dim=1)

        z_1 = z_1.unsqueeze(1) # (B, 1, D)
        z_1 = z_1.repeat(1, z_neg.size(1), 1)  # (B, N, D)
        result = - F.mse_loss(z_1, z_neg, reduction='none')
        result = result.sum(dim=-1)

        return torch.exp(result / self.tau)

    def forward(self, z_1, z_2, z_neg):
        """
        Parameters:
            z_1: (B, D)
            z_2: (B, D)
            z_neg: (B, N, D)
        """
        if self.kernel == 'dot':
            pos = self.dot_pos
            neg = self.dot_neg
        elif self.kernel == 'mse':
            pos = self.mse_pos
            neg = self.mse_neg
        pos = pos(z_1, z_2) / self.tau   # (B, 1)
        neg = torch.sum(neg(z_1, z_2, z_neg), dim=-1)  # (B, 1)
        loss = - pos + torch.log(neg + 1e-16)
        return loss.mean()


if __name__ == '__main__':
    z_1 = torch.ones((4, 10)) * 0.2
    z_2 = torch.ones((4, 10)) * 0.5
    z_neg = torch.ones(4, 16, 10) * 0.3

    loss = NTXent(tau=1.0, kernel='mse')

    out = loss(z_1, z_2, z_neg)
