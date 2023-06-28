import model.dgsr as dgsr
import model.lwsr as lwsr
import model.esrgan as esrgan
from option import args
import utility
from thop import profile
import torch



def hum_convert(value):
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = 1024.0
    for i in range(len(units)):
        if (value / size) < 1:
            return "%.2f%s" % (value, units[i])
        value = value / size
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
device = torch.device('cuda')
def main():
    pre_train = args.pre_train
    print('加载的模型来自于：{}'.format(pre_train))


    # model = dgsr.DGSR(args)
    # model = lwsr.LWSR(args)
    model = esrgan.RRDBNet(1,1,64,23)
    kwargs = {}
    load_from = torch.load(pre_train, **kwargs)
    model.load_state_dict(load_from, strict=True)
    model = model.to(device)
    model.eval()


    input = torch.randn(1, 1, 200, 200).to(device)
    macs, params = profile(model, inputs=input.unsqueeze(dim=0))
    print('flops:', hum_convert(macs) ,'\n', 'params:', hum_convert(params))




if __name__ == '__main__':
    main()