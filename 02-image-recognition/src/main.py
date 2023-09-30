#!/usr/bin/env python3

from options import Options
from model import BaseNetwork
from torchsummary import summary

def main():
    opt = Options().parse()
    model = BaseNetwork(opt)
    summary(model, (opt.nc, opt.isize, opt.isize), batch_size = opt.batch_size, device=opt.device)


if __name__ == '__main__':
    main()
