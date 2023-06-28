from model import common
import torch.nn.functional as F
import torch.nn as nn
import torch
import cv2


def make_model(args, parent=False):
    print('训练的模型为DGSR_NOE')
    return DGSR_NOE(args)


class DGB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(DGB, self).__init__()
        # import pdb
        # pdb.set_trace

        self.da_conv1 = conv(n_feat, n_feat, kernel_size)
        self.da_conv2 = conv(n_feat, n_feat, kernel_size)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        :param x[2]: degradation feature map: B * C * H * W
        '''

        out = self.relu(self.da_conv1(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2(out))
        out = self.conv2(out) + x[0]

        return out


class DGG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(DGG, self).__init__()
        modules_body = []
        modules_body = [
            DGB(
                conv, n_feat, kernel_size, reduction) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)
        self.n_blocks = n_resblocks

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        for i in range(self.n_blocks):
            res = self.body[i](x)
        res = self.body[-1](res)
        res = res + x[0]
        return res

## Residual Channel Attention Network (RCAN)
class DGSR_NOE(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DGSR_NOE, self).__init__()

        self.n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)

        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)

        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        modules_body = [
            DGG(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
            for _ in range(self.n_resgroups)]
        self.compress = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.LeakyReLU(0.1, True)
        )

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        x = self.head(x)
        res = x
        for i in range(self.n_resgroups):
            res = self.body[i](res)
        res = self.body[-1](res)
        res += x

        x = self.tail(res)

        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
