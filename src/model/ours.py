from model import common
import torch.nn.functional as F
import torch.nn as nn
import torch

def make_model(args, parent=False):
    return Ours(args)



class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        self.kernel_size = kernel_size

        self.kernel = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 64 * self.kernel_size * self.kernel_size, bias=False)
        )
        
        
        self.conv = conv(n_feat, n_feat, 1)
        self.ca = CA_layer(n_feat, n_feat, reduction)
  

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        b, c, h, w = x[0].size()
        # import pdb
        # pdb.set_trace()

        # branch 1
        # import pdb
        # pdb.set_trace()
        kernel = self.kernel(x[1]).view(-1, 1, self.kernel_size, self.kernel_size)
        import pdb
        out = self.relu(F.conv2d(x[0].view(1, -1, h, w), kernel, groups=b*c, padding=(self.kernel_size-1)//2))
        out = self.conv(out.view(b, -1, h, w))
        # branch 2
        out = out+self.ca(x)

        return out

class CA_layer(nn.Module):
    def __init__(self, channels_in, channels_out, reduction):
        super(CA_layer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//reduction, 1, 1, 0, bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(channels_in // reduction, channels_out, 1, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''
        att = self.conv_du(x[1][:, :, None, None])

        return x[0] * att

## Residual Group (RG)
# class ResidualGroup(nn.Module):
#     def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
#         super(ResidualGroup, self).__init__()
#         modules_body = []
#         modules_body = [
#             RCAB(
#                 conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
#             for _ in range(n_resblocks)]
#         modules_body.append(conv(n_feat, n_feat, kernel_size))
#         self.body = nn.Sequential(*modules_body)

#     def forward(self, x):
#         res = self.body(x)
#         res += x
#         return res
class DAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction):
        super(DAB, self).__init__()
        # import pdb
        # pdb.set_trace

        self.da_conv1 = RCAB(conv, n_feat, kernel_size, reduction)
        self.da_conv2 = RCAB(conv, n_feat, kernel_size, reduction)
        self.conv1 = conv(n_feat, n_feat, kernel_size)
        self.conv2 = conv(n_feat, n_feat, kernel_size)

        self.relu = nn.LeakyReLU(0.1, True)

    def forward(self, x):
        '''
        :param x[0]: feature map: B * C * H * W
        :param x[1]: degradation representation: B * C
        '''

        out = self.relu(self.da_conv1(x))
        out = self.relu(self.conv1(out))
        out = self.relu(self.da_conv2([out, x[1]]))
        out = self.conv2(out) + x[0]

        return out
class NewResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(NewResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            DAB(
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
            res = self.body[i]([res, x[1]])
        res = self.body[-1](res)
        res = res + x[0]
        
        # res = self.body(x)
        # res += x
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
        import pdb
        pdb.set_trace()
        fea = self.E(x).squeeze(-1).squeeze(-1)
        # out = self.mlp(fea)

        return fea
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()

#         kernel_v = [[0, -1, 0],
#                     [0, 0, 0],
#                     [0, 1, 0]]
#         kernel_h = [[0, 0, 0],
#                     [-1, 0, 1],
#                     [0, 0, 0]]
#         kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0).cuda()
#         kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0).cuda()
#         self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
#         self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)
#         self.E = nn.Sequential(
#             nn.Conv2d(1, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.1, True),

#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.1, True),

#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.1, True),

#             nn.AdaptiveAvgPool2d(1),
#         )

#     def forward(self, x):
#         x_i = x[:, 0]
#         x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
#         x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
#         x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
#         # import pdb
#         # pdb.set_trace()
#         fea = self.E(x_i).squeeze(-1).squeeze(-1)

#         return fea 
## Residual Channel Attention Network (RCAN)
class Ours(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(Ours, self).__init__()
        
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

        # define body module
        # modules_body = [
        #     ResidualGroup(
        #         conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks) \
        #     for _ in range(n_resgroups)]
        modules_body = [
            NewResidualGroup(
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
        fea = self.E(x)
        fea = self.compress(fea)
        x = self.head(x)

        # res = self.body([x, fea])
        res = x
        for i in range(self.n_resgroups):
            res = self.body[i]([res, fea])
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
