import os
from data import srdata

class New_Test(srdata.SRData):
    def __init__(self, args, name='New_Test', train=True, benchmark=False):
        data_range = [r.split('-') for r in args.data_range.split('/')]
        if train:
            data_range = data_range[0]
        else:
            if args.test_only and len(data_range) == 1:
                data_range = data_range[0]
            else:
                data_range = data_range[1]

        self.begin, self.end = list(map(lambda x: int(x), data_range))
        super(New_Test, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_hr, names_lr = super(New_Test, self)._scan()
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = [n[self.begin - 1:self.end] for n in names_lr]
        # names_lr = [[n.replace('3.0', '1.2') for n in names_lr[0]]]
        # import pdb
        # pdb.set_trace()

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(New_Test, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, '2x')
        self.dir_lr = os.path.join(self.apath, 'lr')
        # self.dir_hr = os.path.join('/media/sda/QRcode/various_resolution/3.0')
        # self.dir_lr = os.path.join('/media/sda/QRcode/various_resolution/1.2')
        if self.input_large: self.dir_lr += 'L'