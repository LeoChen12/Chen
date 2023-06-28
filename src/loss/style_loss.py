def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.vgg import Vgg19

class style_loss(nn.Module):
    def __init__(self):
        super(style_loss, self).__init__()
        self.vgg = Vgg19()
        self.gram = gram

    def forward(self, sr, hr):
        vgg_sr = self.vgg(sr.repeat(1,3,1,1))
        style_gram_sr = [self.gram(fmap) for fmap in vgg_sr]

        with torch.no_grad():
            vgg_hr = self.vgg(hr.detach().repeat(1,3,1,1))
            style_gram_hr = [self.gram(fmap) for fmap in vgg_hr]
        style_loss = 0.0
        for j in range(4):

            style_loss += F.l1_loss(style_gram_sr[j], style_gram_hr[j])
        return style_loss
