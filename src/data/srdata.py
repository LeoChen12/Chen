import os
import glob
import random
import pickle

from data import common
# from data.commom import cutblur

import numpy as np
import imageio
import torch
import torch.utils.data as data
from PIL import Image
import torch.nn.functional as F
import cv2

def cutblur(im1, im2, prob=1.0, alpha=1.0):
    # import pdb
    # pdb.set_trace()
    if im1.size() != im2.size():
        raise ValueError("im1 and im2 have to be the same resolution.")

    if alpha <= 0 or np.random.rand(1) >= prob:
        return im1, im2

    cut_ratio = np.random.randn() * 0.01 + alpha

    h, w = im2.size(2), im2.size(3)
    ch, cw = np.int(h*cut_ratio), np.int(w*cut_ratio)
    cy = np.random.randint(0, h-ch+1)
    cx = np.random.randint(0, w-cw+1)

    # apply CutBlur to inside or outside
    if np.random.random() > 0.5:
        im2[..., cy:cy+ch, cx:cx+cw] = im1[..., cy:cy+ch, cx:cx+cw]
    else:
        im2_aug = im1.clone()
        im2_aug[..., cy:cy+ch, cx:cx+cw] = im2[..., cy:cy+ch, cx:cx+cw]
        im2 = im2_aug

    return im1, im2
def imgBrightness(img1, c, b):
    rows, cols, channels = img1.shape
    # print(rows,cols, channels )
    blank = np.zeros([rows, cols, channels], img1.dtype)
    rst = cv2.addWeighted(img1, c, blank, 1-c, b)
    return rst

class SRData(data.Dataset):
    def __init__(self, args, name='', train=True, benchmark=False):
        self.args = args
        self.name = name
        self.train = train
        self.split = 'train' if train else 'test'
        self.do_eval = True
        self.benchmark = benchmark
        self.input_large = (args.model == 'VDSR')
        self.scale = args.scale
        self.idx_scale = 0
        
        self._set_filesystem(args.dir_data)
        if args.ext.find('img') < 0:
            path_bin = os.path.join(self.apath, 'bin')
            os.makedirs(path_bin, exist_ok=True)

        list_hr, list_lr = self._scan()
        if args.ext.find('img') >= 0 or benchmark:
            self.images_hr, self.images_lr = list_hr, list_lr
        elif args.ext.find('sep') >= 0:
            os.makedirs(
                self.dir_hr.replace(self.apath, path_bin),
                exist_ok=True
            )
            for s in self.scale:
                os.makedirs(
                    os.path.join(
                        self.dir_lr.replace(self.apath, path_bin),
                        'X{}'.format(s)
                    ),
                    exist_ok=True
                )
            
            self.images_hr, self.images_lr = [], [[] for _ in self.scale]
            for h in list_hr:
                b = h.replace(self.apath, path_bin)
                b = b.replace(self.ext[0], '.pt')
                self.images_hr.append(b)
                import pdb
                pdb.set_trace()
                self._check_and_load(args.ext, h, b, verbose=True) 
            for i, ll in enumerate(list_lr):
                for l in ll:
                    b = l.replace(self.apath, path_bin)
                    b = b.replace(self.ext[1], '.pt')
                    self.images_lr[i].append(b)
                    self._check_and_load(args.ext, l, b, verbose=True) 
        if train:
            n_patches = args.batch_size * args.test_every
            n_images = len(args.data_train) * len(self.images_hr)
            if n_images == 0:
                self.repeat = 0
            else:
                # self.repeat = 1
                self.repeat = max(n_patches // n_images, 1)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext[0]))
        )
        names_lr = [[] for _ in self.scale]
        for f in names_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            # filename = filename.replace('3.0', '1.2')
            # import pdb
            # pdb.set_trace()
            for si, s in enumerate(self.scale):
                names_lr[si].append(os.path.join(
                    self.dir_lr, '{}{}'.format(
                         filename,  self.ext[1]
                    )
                ))

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        if self.input_large: self.dir_lr += 'L'
        self.ext = ('.png', '.png')

    def _check_and_load(self, ext, img, f, verbose=True):
        if not os.path.isfile(f) or ext.find('reset') >= 0:
            if verbose:
                print('Making a binary: {}'.format(f))
            with open(f, 'wb') as _f:
                pickle.dump(imageio.imread(img), _f)

    def __getitem__(self, idx):
        lr, hr, filename = self._load_file(idx)
        lr, hr = np.array(lr), np.array(hr)
        # import pdb
        # pdb.set_trace()
        # h, w, c = lr.shape
        # hr = cv2.resize(hr, (w*4, h*4), interpolation=cv2.INTER_CUBIC)
        # import pdb
        # pdb.set_trace()
        # print(filename)
        pair = self.get_patch(lr, hr)

        pair = common.set_channel(*pair, n_channels=self.args.n_colors)
        # print(filename)
        pair_t = common.np2Tensor(*pair, rgb_range=self.args.rgb_range)


        # LR = F.interpolate(pair_t[0].unsqueeze(0),scale_factor=4, mode='nearest')
        # # import pdb
        # # pdb.set_trace()
        # hr_cut, LR = cutblur(pair_t[1].unsqueeze(0), LR, prob=0.8, alpha=0.3)  # cutblur数据增广
        # lr_cut = F.interpolate(LR, scale_factor=1 / 4, mode='nearest').squeeze(0)

        return pair_t[0], pair_t[1], filename

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        f_hr = self.images_hr[idx]
        f_lr = self.images_lr[self.idx_scale][idx]

        filename, _ = os.path.splitext(os.path.basename(f_hr))
        if self.args.ext == 'img' or self.benchmark:
            hr = Image.open(f_hr).convert('RGB')
            lr = Image.open(f_lr).convert('RGB')
            # hr = imageio.imread(f_hr).convert('RGB')
            # lr = imageio.imread(f_lr).convert('RGB')
        elif self.args.ext.find('sep') >= 0:
            with open(f_hr, 'rb') as _f:
                hr = pickle.load(_f)
            with open(f_lr, 'rb') as _f:
                lr = pickle.load(_f)

        return lr, hr, filename

    def get_patch(self, lr, hr):
        scale = self.scale[self.idx_scale]
        if self.train:
            x = random.uniform(0.5, 1.5)
            a = round(x, 3)
            lr = imgBrightness(lr, a, 1)
            hr = imgBrightness(hr, a, 1)
            
            lr, hr = common.get_patch(
                lr,
                hr,
                patch_size=self.args.patch_size,
                scale=scale,
                multi=(len(self.scale) > 1),
                input_large=self.input_large
            )
            if not self.args.no_augment: lr, hr = common.augment(lr, hr)
        else:
            ih, iw = lr.shape[:2]
            hr = hr[0:ih * scale, 0:iw * scale]

        return lr, hr

    def set_scale(self, idx_scale):
        if not self.input_large:
            self.idx_scale = idx_scale
        else:
            self.idx_scale = random.randint(0, len(self.scale) - 1)

