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
from torchsummary import summary
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from utils import show_image, show_batch, get_metrics, precision, accuracy, recall, f1_score


# Flags

class BaseNetwork(nn.Module):
    """
    Class combined Generator and Discrimantor networks to one GAN model
    """
    def __init__(self, opt):
        super(BaseNetwork, self).__init__()

        assert opt.isize == 64, "only support 64x64 input images"
        self.opt = opt
        self.dtype = torch.float
        self.epoch = 0
        self.total_steps = 0
        self.loss = 0
        self.best_accuracy = 0
        self.init_function = nn.init.normal_
        layers = []

        # Select device
        if self.opt.device == 'cuda':
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        elif self.opt.device == 'mps' and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        print(f"Using device: {self.device}")

        def __downsampling_block():
            if self.opt.batchnorm:
                return nn.Sequential(
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(self.opt.ndf, self.opt.ndf, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.BatchNorm2d(self.opt.ndf))
            return nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.opt.ndf, self.opt.ndf, kernel_size=3, stride=1, padding=1, bias=False),
                nn.MaxPool2d(kernel_size=2, stride=2))

        def __conv_block():
            if self.opt.batchnorm:
                return nn.Sequential(
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(self.opt.ndf, self.opt.ndf, kernel_size=3, stride=1, padding='same', bias=False),
                    nn.BatchNorm2d(self.opt.ndf))
            return nn.Sequential(
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(self.opt.ndf, self.opt.ndf, kernel_size=3, stride=1, padding='same', bias=False))

        # Input layer
        # 64x64x32
        layers.append(nn.Conv2d(self.opt.nc, self.opt.ndf, kernel_size=3, stride=1, padding='same', bias=False))

        # Extra 64x64x32 layers (no downsampling)
        for t in range(self.opt.layers64 - 1):
            layers.append(__conv_block())

        # 32x32x32
        layers.append(__downsampling_block())

        # Extra 32x32x32 layers (no downsampling)
        for t in range(self.opt.layers32 - 1):
            layers.append(__conv_block())

        # 16x16x32
        layers.append(__downsampling_block())

        # Extra 16x16x32 layers (no downsampling)
        for t in range(self.opt.layers16 - 1):
            layers.append(__conv_block())


        # 8x8x32
        layers.append(__downsampling_block())

        # Extra 8x8x32 layers (no downsampling)
        for t in range(self.opt.layers8 - 1):
            layers.append(__conv_block())

        # Flatten
        layers.append(nn.Flatten())

        # Vectorize 1x2048
        layers.append(nn.Linear(self.opt.ndf*8*8, 128, bias=False))

        # Vectorize 1x128
        layers.append(nn.Linear(128, 1, bias=False))

        # Prediction not needed since we use BCEWithLogitsLoss
        #layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

        # Initialize weights
        self.initilize_weights()

        # Initialize inputs and ground truth
        self.input_tensor = torch.empty(size=(self.opt.batch_size, self.opt.nc, self.opt.isize, self.opt.isize), dtype=self.dtype, device=self.device)
        self.gt = torch.empty(size=(self.opt.batch_size,), dtype=self.dtype, device=self.device)

        # Optimizer
        if self.opt.optimizer == 'adam':
            self.optimizer = Adam(self.parameters(), lr=self.opt.lr)
        elif self.opt.optimizer == 'adamw':
            self.optimizer = AdamW(self.parameters(), lr=self.opt.lr)
        elif self.opt.optimizer == 'sgd':
            self.optimizer = SGD(self.parameters(), lr=self.opt.lr, momentum=0.9)
        else:
            self.optimizer = Adam(self.parameters(), lr=self.opt.lr)

        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()

    def initilize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                self.init_function(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    self.init_function(m.bias.data, 0)

            elif isinstance(m, nn.BatchNorm2d):
                self.init_function(m.weight.data, 1.0, 0.02)
                self.init_function(m.bias.data, 0)

            elif isinstance(m, nn.Linear):
                self.init_function(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    self.init_function(m.bias.data, 0)
        print('Initialized weights')


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
        prediction = self.forward(self.input_tensor)
        loss = self.criterion(prediction, self.gt.unsqueeze(1))
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
        with torch.no_grad():
            predictions = []
            gts = []
            for i, data in enumerate(val_loader, 0):
                self.set_input(data)
                outputs = self.forward(self.input_tensor).squeeze(1)
                outputs = torch.sigmoid(outputs)
                predicted = (outputs > 0.5)

                predictions.extend(predicted.tolist())
                gts.extend(self.gt.tolist())
                #show_batch(self.input_tensor.to('cpu'), self.gt.to('cpu'), prediction_labels)
            tp, tn, fp, fn = get_metrics(predictions, gts)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            print(f'Epoch {self.epoch}---------------------------------------------------------------')
            print(f'Accuracy: {accuracy} | TP: {tp}, TN: {tn}, fp: {fp}, FN: {fn}')

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                print(f'New Best accuracy: {self.best_accuracy}')

    def print_network(self):
        summary(self, (self.opt.nc, self.opt.isize, self.opt.isize), batch_size = self.opt.batch_size, device='cpu')
