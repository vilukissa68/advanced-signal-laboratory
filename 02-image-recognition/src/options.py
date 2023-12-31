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
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--model', type=str, default='SimpleCNN', help='chooses which model to use')
        self.parser.add_argument('--dataset', default='GENKI-4K', help='GENKI-4K | GENKI-4K-Grayscale | GENKI-4K-Augmented | GENKI-4K-Grayscale-Augmented')
        self.parser.add_argument('--dataroot', default='data/GENKI-R2009a/Subsets/GENKI-4K/', help='relative from project root path to dataset')
        self.parser.add_argument('--images_file', default='GENKI-4K_Images.txt', help='Name of image file')
        self.parser.add_argument('--labels_file', default='GENKI-4K_Labels.txt', help='Name of the labels file')
        self.parser.add_argument('--serialization_target_dir', default='serialized', help='where to put serialized data')
        self.parser.add_argument('--serialization_target_dir_grayscale', default='grayscale', help='where to put serialized data')
        self.parser.add_argument('--serialization_target_dir_augmented', default='augmented', help='where to put serialized data')
        self.parser.add_argument('--serialization_target_dir_grayscale_augmented', default='grayscale_augmented', help='where to put serialized data')
        self.parser.add_argument('--serialization_source_dir', default='files', help='source images for serialization')
        self.parser.add_argument('--key_classes', default='label', help='key for class')
        self.parser.add_argument('--key_features', default='features', help='key for features')
        self.parser.add_argument('--load_into_memory', type=bool, default=True, action=argparse.BooleanOptionalAction, help='load inputs in to memory')
        self.parser.add_argument('--test', type=bool, default=False, action=argparse.BooleanOptionalAction, help='test model')
        self.parser.add_argument('--num_workers', type=int, default=1, help='number of of dataloading workers')
        self.parser.add_argument('--train', type=bool, default=False, action=argparse.BooleanOptionalAction, help='train model')
        self.parser.add_argument('--load_model', type=bool, default=False, action=argparse.BooleanOptionalAction, help='load existing model')
        self.parser.add_argument('--models_dir', type=str, default='models/', help='directory to save models')
        self.parser.add_argument('--weights', type=str, default='best.model', help='name of the weights file')
        self.parser.add_argument('--scaling_factor', type=float, default=1, help='the scaling factor for face ROI')
        self.parser.add_argument('--draw_matrix', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Draw confusion matrix from best epoch')


        # Data
        self.parser.add_argument('--serialize', type=bool, default=False, action=argparse.BooleanOptionalAction, help='serialize data')
        self.parser.add_argument('--view-data', type=bool, default=False, action=argparse.BooleanOptionalAction, help='serialize data')

        # Model
        self.parser.add_argument('--isize', type=int, default=64, help='input image size.')
        self.parser.add_argument('--nc', type=int, default=3, help='input image channel count')
        self.parser.add_argument('--ndf', type=int, default=32, help='filter amount')
        self.parser.add_argument('--layers64', type=int, default=1, help='64x64 layer count')
        self.parser.add_argument('--layers32', type=int, default=2, help='32x32 layer count')
        self.parser.add_argument('--layers16', type=int, default=2, help='16x16 layer count')
        self.parser.add_argument('--layers8', type=int, default=1, help='8x8 layer count')
        self.parser.add_argument('--batchnorm', type=bool, default=True, action=argparse.BooleanOptionalAction, help='use batch normalization')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='beta2')
        self.parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')

        # Train
        self.parser.add_argument('--batch_size', type=int, default=64, help='batch size')
        self.parser.add_argument('--train_split', type=float, default=0.8, help='train-to-test split division')
        self.parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
        self.parser.add_argument('--optimizer', type=str, default='adam', help='optimizer: sgd | adam | adamw')
        self.parser.add_argument('--with_noise', type=bool, default=False, action=argparse.BooleanOptionalAction, help='add noise to input during training')
        self.parser.add_argument('--noise_std', type=float, default=0.1, action=argparse.BooleanOptionalAction, help='noise amplitude')
        self.isTrain = True

        # Debugging
        self.parser.add_argument('--random_input', type=bool, default=False, help='use random input, debugging only')
        self.parser.add_argument('--verbose', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Print the training and model details.')
        self.parser.add_argument('--silent', type=bool, default=False, action=argparse.BooleanOptionalAction, help='silent mode, used in shell scripts, only print output to stdout')
        self.parser.add_argument('--tensorboard', type=bool, default=False, action=argparse.BooleanOptionalAction, help='use tensorboard for logging')
        self.parser.add_argument('--tensorboard_dir', type=str, default='runs', help='tensorboard log directory')

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
        self.opt.serialization_source_dir = self.opt.datadir / self.opt.serialization_source_dir
        self.opt.serialization_target_dir = self.opt.datadir / self.opt.serialization_target_dir
        self.opt.serialization_target_dir_grayscale = self.opt.datadir / self.opt.serialization_target_dir_grayscale
        self.opt.serialization_target_dir_augmented = self.opt.datadir / self.opt.serialization_target_dir_augmented
        self.opt.serialization_target_dir_grayscale_augmented = self.opt.datadir / self.opt.serialization_target_dir_grayscale_augmented

        # Tensorboard dir
        self.opt.tensorboard_dir = self.opt.root / self.opt.tensorboard_dir

        # Path to load weights
        self.opt.weights = self.opt.root / self.opt.models_dir / self.opt.weights

        self.opt.models_dir = self.opt.root / self.opt.models_dir

        # Set input channels based on dataset
        if self.opt.dataset == 'GENKI-4K-Grayscale' or self.opt.dataset == 'GENKI-4K-Grayscale-Augmented':
            self.opt.nc = 1
        else:
            self.opt.nc = 3

        args = vars(self.opt)

        assert not (self.opt.silent == True and self.opt.verbose == True), "Cannot be silent and verbose at the same time."

        ## Model name based on the layers
        self.opt.model = '%s_%sx64_%sx32_%sx16_%sx8' % (self.opt.model, self.opt.layers64, self.opt.layers32, self.opt.layers16, self.opt.layers8)

        # Set tag by start time, experiment name, and model
        self.opt.starttime = time.strftime("%Y%m%d-%H%M%S")
        self.opt.tag = '%s_%s_%s_%s' % (self.opt.starttime, self.opt.name, self.opt.model, self.opt.dataset)

        if self.opt.verbose:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        # save to the disk
        if self.opt.name == 'experiment_name':
            self.opt.name = "%s/%s" % (self.opt.model, self.opt.dataset)

        if not os.path.isdir(self.opt.models_dir):
            os.makedirs(self.opt.models_dir)

        file_name = os.path.join(self.opt.models_dir, (self.opt.tag + '_opt.txt'))
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
