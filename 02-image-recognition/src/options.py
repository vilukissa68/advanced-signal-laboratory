#!/usr/bin/env python3
import argparse
import os
import time
import pathlib

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
        self.parser.add_argument('--device', type=str, default='cpu', help='Device: cuda | cpu | mps')
        self.parser.add_argument('--outf', default='./output', help='folder to output images and model checkpoints')
        self.parser.add_argument('--verbose', action='store_true', help='Print the training and model details.')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--model', type=str, default='BaseNetwork', help='chooses which model to use')
        self.parser.add_argument('--dataset', default='cifar10', help='folder | cifar10 | mnist ')
        self.parser.add_argument('--dataroot', default='data/GENKI-R2009a/Subsets/GENKI-4K/', help='relative from project root path to dataset')
        self.parser.add_argument('--images_file', default='GENKI-4K_Images.txt', help='Name of image file')
        self.parser.add_argument('--labels_file', default='GENKI-4K_Labels.txt', help='Name of the labels file')
        self.parser.add_argument('--serialization_target_dir', default='serialized', help='where to put serialized data')
        self.parser.add_argument('--serialization_source_dir', default='files', help='source images for serialization')
        self.parser.add_argument('--key_classes', default='label', help='key for class')
        self.parser.add_argument('--key_features', default='features', help='key for features')
        self.parser.add_argument('--load_into_memory', type=bool, default=True, help='load inputs in to memory')
        self.parser.add_argument('--num_workers', type=int, default=1, help='number of of dataloading workers')

        # Model
        self.parser.add_argument('--isize', type=int, default=64, help='input image size.')
        self.parser.add_argument('--nc', type=int, default=3, help='input image channel count')
        self.parser.add_argument('--ndf', type=int, default=32, help='filter amount')
        self.parser.add_argument('--layers64', type=int, default=1, help='64x64 layer count')
        self.parser.add_argument('--layers32', type=int, default=2, help='32x32 layer count')
        self.parser.add_argument('--layers16', type=int, default=2, help='16x16 layer count')
        self.parser.add_argument('--layers8', type=int, default=1, help='8x8 layer count')
        self.parser.add_argument('--batchnorm', type=bool, default=True, help='use batch normalization')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
        self.parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')

        # Train
        self.parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        self.parser.add_argument('--train_split', type=float, default=0.8, help='train-to-test split division')
        self.parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
        self.parser.add_argument('--optimizer', type=str, default='adam', help='optimizer: sgd | adam | adamw')
        self.isTrain = True

    def parse(self):
        """ Parse Arguments.
        """

        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        self.opt.root = pathlib.Path(__file__).parents[1]
        self.opt.datadir = self.opt.root / self.opt.dataroot

        # Concatenate to global paths
        self.opt.images_file = self.opt.datadir / self.opt.images_file
        self.opt.labels_file = self.opt.datadir / self.opt.labels_file
        self.opt.serialization_target_dir = self.opt.datadir / self.opt.serialization_target_dir
        self.opt.serialization_source_dir = self.opt.datadir / self.opt.serialization_source_dir

        args = vars(self.opt)

        # Set tag by start time, experiment name, and model
        self.opt.starttime = time.strftime("%Y%m%d-%H%M%S")
        self.opt.tag = '%s_%s_%s' % (self.opt.starttime, self.opt.name, self.opt.model)

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
