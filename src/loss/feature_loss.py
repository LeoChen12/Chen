
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.vgg import Vgg19

class feature_loss(nn.Module):
    def __init__(self):
        super(feature_loss, self).__init__()
        self.vgg = Vgg19()


    def forward(self, sr, hr):
        vgg_sr = self.vgg(sr.repeat(1,3,1,1))
        with torch.no_grad():
            vgg_hr = self.vgg(hr.detach().repeat(1,3,1,1))
        feature_loss = F.l1_loss(vgg_sr[2], vgg_hr[2])
        return feature_loss
