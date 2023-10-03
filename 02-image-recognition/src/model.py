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
from torch.optim import Adam, AdamW, SGD
from torchmetrics.classification import AUROC
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from utils import show_image, get_metrics, precision, accuracy, recall, f1_score


# Flags

class BaseNetwork(nn.Module):
    """
    Class combined Generator and Discrimantor networks to one GAN model
    """
    def __init__(self, opt):
        super(BaseNetwork, self).__init__()

        assert opt.isize == 64, "only support 64x64 input images"
        self.opt = opt
        self.epoch = 0
        self.total_steps = 0
        self.loss = 0
        self.best_accuracy = 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        layers = []

        def __downsampling_block(opt):
            if opt.batchnorm:
                return nn.Sequential(
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(opt.ndf, opt.ndf, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(opt.ndf))
            return nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(opt.ndf, opt.ndf, kernel_size=3, stride=1, padding=1, bias=False),
                nn.MaxPool2d(kernel_size=2, stride=2))

        def __conv_block(opt):
            if opt.batchnorm:
                return nn.Sequential(
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(opt.ndf, opt.ndf, kernel_size=3, stride=1, padding='same', bias=False),
                    nn.BatchNorm2d(opt.ndf))
            return nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(opt.ndf, opt.ndf, kernel_size=3, stride=1, padding='same', bias=False))

        # Input layer
        # 64x64x32
        layers.append(nn.Conv2d(opt.nc, opt.ndf, kernel_size=3, stride=1, padding='same', bias=False))

        # Extra 64x64x32 layers (no downsampling)
        for t in range(opt.layers64 - 1):
            layers.append(__conv_block(opt))

        # 32x32x32
        layers.append(__downsampling_block(opt))

        # Extra 32x32x32 layers (no downsampling)
        for t in range(opt.layers32 - 1):
            layers.append(__conv_block(opt))

        # 16x16x32
        layers.append(__downsampling_block(opt))

        # Extra 16x16x32 layers (no downsampling)
        for t in range(opt.layers16 - 1):
            layers.append(__conv_block(opt))


        # 8x8x32
        layers.append(__downsampling_block(opt))

        # Extra 8x8x32 layers (no downsampling)
        for t in range(opt.layers8 - 1):
            layers.append(__conv_block(opt))

        # Flatten
        layers.append(nn.Flatten())

        # Vectorize 1x2048
        layers.append(nn.Linear(opt.ndf*8*8, 128, bias=False))

        # Vectorize 1x128
        layers.append(nn.Linear(128, 1, bias=False))

        # Prediction not needed since we use BCEWithLogitsLoss
        #layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

        # Initialize inputs and ground truth
        self.input_tensor = torch.empty(size=(self.opt.batch_size, self.opt.nc, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.gt = torch.empty(size=(self.opt.batch_size,), dtype=torch.float32, device=self.device)

        # Optimizer
        self.optimizer = Adam(self.parameters(), lr=opt.lr)
        #self.optimizer = SGD(self.parameters(), lr=opt.lr, momentum=0.9)

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()



    def forward(self, input):
        return self.layers(input)

    def set_input(self, data):
        '''Set input per batch'''
        with torch.no_grad():
            self.input_tensor.resize_(data[0].size()).copy_(data[0])
            self.gt.resize_(data[1].size()).copy_(data[1])

    def optimize_parameters(self):
        '''Optimize parameters per batch'''
        self.optimizer.zero_grad()
        prediction = self.forward(self.input_tensor).squeeze(1)
        loss = self.criterion(prediction, self.gt.float())
        loss.backward()
        self.optimizer.step()
        self.loss += loss.item()


    def train_epoch(self, train_loader):
        '''Train model per epoch'''
        for i, data in enumerate(train_loader, 0):
            self.total_steps += self.opt.batch_size
            self.set_input(data)
            self.optimize_parameters()
        print(f'Epoch {self.epoch} loss: {self.loss}')
        self.loss = 0.0



    def train(self, train_loader, val_loader):
        '''Train model for opt.epochs'''
        print(f'Start training at {datetime.datetime.now()}')
        for self.epoch in range(0, self.opt.epochs):
            self.train_epoch(train_loader)
            self.test(val_loader)

    def test(self, val_loader):
        '''Test model'''
        predictions = []
        gts = []
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                self.set_input(data)
                prediction = self.forward(self.input_tensor).squeeze(1)
                prediction_labels = [1 if x > 0.5 else 0 for x in prediction]
                predictions += (prediction_labels)
                gts += ([int(x) for x in self.gt])
            tp, fp, tn, fn = get_metrics(predictions, gts)
            print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
            print(f'Accuracy: {(tp + tn) / (tp + tn + fp + fn)}')

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                print(f'New Best accuracy: {self.best_accuracy}')
