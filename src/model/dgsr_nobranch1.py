from model import common
import torch.nn.functional as F
import torch.nn as nn
import torch
import cv2


def make_model(args, parent=False):
    print('训练的模型为DGSR_no_branch1')
    return DGSR(args)


class DPB(nn.Module):
    def __init__(
            self, conv, n_feat, kernel_size, reduction,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(DPB, self).__init__()

        self.kernel_size = kernel_size
        self.conv_gamma = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat * 2, n_feat, 1, 1, 0, bias=False),
        )
        self.conv_beta = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat * 2, n_feat, 1, 1, 0, bias=False),
        )
        self.ca = CA_layer(n_feat, n_feat, reduction)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        :param x[2]: degradation feature map: B * C * H * W
        '''
        b, c, h, w = x[0].size()


        out = x[0] + self.ca(x)


        return out


class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in // reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        :param x[2]: degradation feature map: B * C * H * W
        '''
        att = self.conv_du(x[1][:, :, None, None])

        return x[0] * att


class DGB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(DGB, self).__init__()
        # import pdb
        # pdb.set_trace

        self.da_conv1 = DPB(conv, n_feat, kernel_size, reduction)
        self.da_conv2 = DPB(conv, n_feat, kernel_size, reduction)
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
        out = self.relu(self.da_conv2([out, x[1], x[2]]))
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
        res = x[0]
        for i in range(self.n_blocks):
            res = self.body[i]([res, x[1], x[2]])
        res = self.body[-1](res)
        res = res + x[0]
        return res


### QR code degradation information encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )
        # self.mlp = nn.Sequential(
        #     nn.Linear(256, 256),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(256, 256),
        # )

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        degra_map = self.E[:4](x)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        # b, c, h, w = degra_map.shape
        # for i in range(c):
        #     fm = degra_map[0][i]
        #     fm = fm.cpu().detach().numpy().astype('uint8')
        #     fm  = fm.reshape(h, w, 1)
        #     cv2.imwrite('/home/wangchen/xindalu_code/EDSR-PyTorch-master/feature/{}-{}.png'.format(h, i), fm)

        # out = self.mlp(fea)

        return fea, degra_map


## Residual Channel Attention Network (RCAN)
class DGSR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(DGSR, self).__init__()

        self.n_resgroups = args.n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction
        scale = args.scale[0]
        act = nn.ReLU(True)

        ### QR code degradation encoder
        self.E = Encoder()

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
        # x = self.sub_mean(x)
        fea, degra_map = self.E(x)
        fea = self.compress(fea)
        x = self.head(x)
        res = x
        for i in range(self.n_resgroups):
            res = self.body[i]([res, fea, degra_map])
        res = self.body[-1](res)
        res += x

        x = self.tail(res)
        # x = self.add_mean(x)

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
