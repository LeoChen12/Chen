import os
from data import srdata

class Rotate_QRdata(srdata.SRData):
    def __init__(self, args, name='Rotate_QRdata', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(Rotate_QRdata, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(Rotate_QRdata, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(Rotate_QRdata, self)._set_filesystem(dir_data)
        # print('*******************')
        # self.dir_hr = os.path.join('/home/wangchen/doublewang/rotate_datasets/rotate_dataset/train', 'hr_4x')
        # self.dir_lr = os.path.join('/home/wangchen/doublewang/rotate_datasets/rotate_dataset/train', 'lr')
        # if self.input_large: self.dir_lr += 'L'
        super(Rotate_QRdata, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, '2x')
        self.dir_lr = os.path.join(self.apath, 'lr')
        if self.input_large: self.dir_lr += 'L'