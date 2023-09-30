#!/usr/bin/env python3
#!/usr/bin/env python3

import os
import time
import datetime
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, AdamW
from torchmetrics.classification import AUROC
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score


# Flags

class BaseNetwork(nn.Module):
    """
    Class combined Generator and Discrimantor networks to one GAN model
    """
    def __init__(self, opt):
        super(BaseNetwork, self).__init__()

        assert opt.isize == 64, "only support 64x64 input images"

        layers = []

        # Input layer
        # 64x64x32
        layers.append(nn.Conv2d(opt.nc, opt.ndf, kernel_size=3, stride=1, padding='same', bias=False))

        # Extra 64x64x32 layers (no downsampling)
        for t in range(opt.layers64 - 1):
            layers.append(self.__conv_block(opt))

        # 32x32x32
        layers.append(self.__downsampling_block(opt))

        # Extra 32x32x32 layers (no downsampling)
        for t in range(opt.layers32 - 1):
            layers.append(self.__conv_block(opt))

        # 16x16x32
        layers.append(self.__downsampling_block(opt))

        # Extra 16x16x32 layers (no downsampling)
        for t in range(opt.layers16 - 1):
            layers.append(self.__conv_block(opt))


        # 8x8x32
        layers.append(self.__downsampling_block(opt))

        # Extra 8x8x32 layers (no downsampling)
        for t in range(opt.layers8 - 1):
            layers.append(self.__conv_block(opt))

        # Flatten
        layers.append(nn.Flatten())

        # Vectorize 1x2048
        layers.append(nn.Linear(opt.ndf*8*8, 128, bias=False))

        # Vectorize 1x128
        layers.append(nn.Linear(128, 1, bias=False))

        # Prediction
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)


    def forward(self, input):
        return self.layers(input)

    def __downsampling_block(self, opt):
        if opt.batchnorm:
            return nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(opt.ndf, opt.ndf, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(opt.ndf))
        return nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opt.ndf, opt.ndf, kernel_size=3, stride=2, padding=1, bias=False))

    def __conv_block(self, opt):
        if opt.batchnorm:
            return nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(opt.ndf, opt.ndf, kernel_size=3, stride=1, padding='same', bias=False),
                nn.BatchNorm2d(opt.ndf))
        return nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(opt.ndf, opt.ndf, kernel_size=3, stride=1, padding='same', bias=False))
