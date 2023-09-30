#!/usr/bin/env python3
import argparse
import os
import torch

# pylint: disable=C0103,C0301,R0903,W0622

class Options():
    """Options class

    Returns:
        [argparse]: argparse containing train and test options
    """

    def __init__(self):
        ##
        #
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        ##
        # Base
        self.parser.add_argument('--device', type=str, default='cpu', help='Device: gpu | cpu')
        self.parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
        self.parser.add_argument('--verbose', action='store_true', help='Print the training and model details.')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--model', type=str, default='BaseNetwork', help='chooses which model to use')
        self.parser.add_argument('--dataset', default='cifar10', help='folder | cifar10 | mnist ')
        self.parser.add_argument('--dataroot', default='', help='path to dataset')

        # Model
        self.parser.add_argument('--isize', type=int, default=64, help='input image size.')
        self.parser.add_argument('--nc', type=int, default=3, help='input image channel count')
        self.parser.add_argument('--ndf', type=int, default=32, help='filter amount')
        self.parser.add_argument('--layers64', type=int, default=1, help='64x64 layer count')
        self.parser.add_argument('--layers32', type=int, default=2, help='32x32 layer count')
        self.parser.add_argument('--layers16', type=int, default=2, help='16x16 layer count')
        self.parser.add_argument('--layers8', type=int, default=1, help='8x8 layer count')
        self.parser.add_argument('--batchnorm', type=bool, default=True, help='use batch normalization')


        ##
        # Train
        self.parser.add_argument('--batch_size', type=int, default=16, help='batch size')
        self.isTrain = True

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        args = vars(self.opt)

        if self.opt.verbose:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        # save to the disk
        if self.opt.name == 'experiment_name':
            self.opt.name = "%s/%s" % (self.opt.model, self.opt.dataset)
        expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
