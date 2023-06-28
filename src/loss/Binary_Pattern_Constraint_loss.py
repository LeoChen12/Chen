import torch


import torch.nn as nn
import torch.nn.functional as F


class Binary_Pattern_Constraint_loss(nn.Module):
    def __init__(self):
        super(Binary_Pattern_Constraint_loss, self).__init__()
    def weight(self,x):
        w = (4*torch.exp_(-0.05*(x - 128)))/((1+torch.exp_(-0.05*(x-128)))**2)
        return w

    def forward(self, sr, hr):
        weight_sr = self.weight(sr)
        weight_hr = self.weight(hr)
        # loss = F.mse_loss(weight_sr*sr, weight_hr*hr)
        loss = F.mse_loss(weight_sr * sr, weight_hr * hr)
        return loss