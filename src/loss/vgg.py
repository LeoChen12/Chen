from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self, conv_index, rgb_range=1):
        super(VGG, self).__init__()
        vgg_features = models.vgg19(pretrained=True).features
        modules = [m for m in vgg_features]
        if conv_index.find('22') >= 0:
            self.vgg = nn.Sequential(*modules[:8])
        elif conv_index.find('54') >= 0:
            self.vgg = nn.Sequential(*modules[:35])

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = common.MeanShift(rgb_range, vgg_mean, vgg_std)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, sr, hr):
        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x
            
        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr.detach())

        loss = F.mse_loss(vgg_sr, vgg_hr)

        return loss
class Vgg19(nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        features = models.vgg19(pretrained=True).features

        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        # self.to_relu_1_2.add_module(str(0), nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))) # modified
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 25):
            self.to_relu_4_3.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h

        h = self.to_relu_2_2(h)
        h_relu_2_2 = h

        h = self.to_relu_3_3(h)
        h_relu_3_3 = h

        h = self.to_relu_4_3(h)
        h_relu_4_3 = h

        out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return out