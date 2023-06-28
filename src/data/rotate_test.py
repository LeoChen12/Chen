import os
from data import srdata

class Rotate_Test(srdata.SRData):
    def __init__(self, args, name='Rotate_Test', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(Rotate_Test, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(Rotate_Test, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(Rotate_Test, self)._set_filesystem(dir_data)
        # self.dir_hr = os.path.join('/home/wangchen/doublewang/rotate_datasets/rotate_dataset/test', 'hr_4x')
        # self.dir_lr = os.path.join('/home/wangchen/doublewang/rotate_datasets/rotate_dataset/test', 'lr')
        # if self.input_large: self.dir_lr += 'L'
        self.dir_hr = os.path.join(self.apath, '2x')
        self.dir_lr = os.path.join(self.apath, 'lr')
        if self.input_large: self.dir_lr += 'L'