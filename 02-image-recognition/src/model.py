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
import random


# Flags

class BaseNetwork(nn.Module):
    """
    Class combined Generator and Discrimantor networks to one GAN model
    """
    def __init__(self, opt):
        super(BaseNetwork, self).__init__()

        assert opt.isize == 64, "only support 64x64 input images"

        # Make sure that the number of channels is correct
        if opt.dataset == 'GENKI-4K-Grayscale' or opt.dataset == 'GENKI-4K-Grayscale-Augmented':
            assert opt.nc == 1, "only support grayscale input images"
        else:
            assert opt.nc == 3, "only support RGB input images"

        self.opt = opt
        self.opt_runtime = None
        self.dtype = torch.float
        self.epoch = 0
        self.total_steps = 0
        self.loss = 0
        self.recall = 0.0
        self.precision = 0.0
        self.f1 = 0.0
        self.accuracy = 0.0
        self.best_accuracy = 0.0
        self.best_epoch = 0
        self.init_function = nn.init.normal_
        self.best_matrix = None

        if opt.tensorboard:
            self.writer = SummaryWriter(log_dir=opt.tensorboard_dir / self.opt.tag)

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
        self.noise = torch.empty(size=(self.opt.batch_size, self.opt.nc, self.opt.isize, self.opt.isize), dtype=self.dtype, device=self.device)

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
            if self.opt.isTrain and self.opt.with_noise:
                self.noise.data.copy_(torch.normal(0, self.opt.noise_std, size=self.noise.size()))

            self.input_tensor.resize_(data[0].size()).copy_(data[0])
            self.gt.resize_(data[1].size()).copy_(data[1])

    def optimize_parameters(self):
        '''Optimize parameters per batch'''
        self.optimizer.zero_grad()
        prediction = self.forward(self.input_tensor + self.noise)
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

    def trainining_loop(self, train_loader, val_loader):
        '''Train model for opt.epochs'''
        self.train()
        print(f'Start training at {datetime.datetime.now()}')
        for self.epoch in range(0, self.opt.epochs):
            self.train_epoch(train_loader)
            self.test(val_loader)
            print(f'loss: {self.loss}\n')
            if self.opt.tensorboard:
                self.writer.add_scalar('Loss/train', self.loss, self.epoch)
                self.writer.add_scalar('Accuracy/train', self.accuracy, self.epoch)
            self.loss = 0.0

    def test(self, val_loader):
        '''Test model'''
        self.eval()
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
            tp, tn, fp, fn = get_metrics(predictions, gts)
            self.accuracy = (tp + tn) / (tp + tn + fp + fn)
            self.precision = tp / (tp + fp) if tp + fp > 0 else 0
            self.recall = tp / (tp + fn) if tp + fn > 0 else 0
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall) if self.precision + self.recall > 0 else 0
            print(f'Epoch {self.epoch}---------------------------------------------------------------')
            print(f'Accuracy: {self.accuracy} | TP: {tp}, TN: {tn}, fp: {fp}, FN: {fn}')

            if self.accuracy >= self.best_accuracy:
                self.best_accuracy = self.accuracy
                self.best_epoch = self.epoch
                self.best_matrix = (tp, tn, fp, fn)
                print(f'New Best accuracy: {self.best_accuracy}')
                self.save_model()

    def predict(self, input):
        '''Predict image'''
        self.eval()
        with torch.no_grad():
            prediction = self.forward(input)
            prediction = torch.sigmoid(prediction)
            prediction = prediction.squeeze(1)
            return prediction.item()

    
    def print_network(self):
        summary(self, (self.opt.nc, self.opt.isize, self.opt.isize), batch_size = self.opt.batch_size, device='cpu')

    def save_model(self):
        '''Save model'''
        if not os.path.exists(self.opt.models_dir):
            os.makedirs(self.opt.models_dir)
        filename = f'{self.opt.models_dir}/{self.opt.tag}.model'
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "dtype": self.dtype,
            "epoch": self.epoch,
            "total_steps": self.total_steps,
            "best_accuracy": self.best_accuracy,
            "init_function": self.init_function,
            "opt": self.opt,
            "matrix": self.best_matrix
        }, filename)
        if self.opt.verbose:
            print(f'Saved model {filename} with accuracy {self.best_accuracy}')


    def dry_test(self, val_loader):
        self.eval()
        with torch.no_grad():
            predictions = []
            gts = []
            images = [] # Hold images for plotting in the end
            for i, data in enumerate(val_loader, 0):
                self.set_input(data)
                outputs = self.forward(self.input_tensor).squeeze(1)
                images.extend(self.input_tensor.tolist())
                outputs = torch.sigmoid(outputs)
                predicted = (outputs > 0.5)
                predictions.extend(predicted.tolist())
                gts.extend(self.gt.tolist())
            tp, tn, fp, fn = get_metrics(predictions, gts)
            self.accuracy = (tp + tn) / (tp + tn + fp + fn)
            self.precision = tp / (tp + fp) if tp + fp > 0 else 0
            self.recall = tp / (tp + fn) if tp + fn > 0 else 0
            self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall) if self.precision + self.recall > 0 else 0

            # Convert lists to torch tensors
            predictions = torch.tensor(predictions)
            gts = torch.tensor(gts)
            images = torch.tensor(images)

            # Gather tps tns fps fns with images
            tps = images[(predictions == 1) & (predictions == gts)]
            tns = images[(predictions == 0) & (predictions == gts)]
            fps = images[(predictions == 1) & (predictions != gts)]
            fns = images[(predictions == 0) & (predictions != gts)]

            # Take 4 random images from each, if available
            tps = tps[torch.randperm(tps.size(0))][:min(4, len(tps))]
            tns = tns[torch.randperm(tns.size(0))][:min(4, len(tns))]
            fps = fps[torch.randperm(fps.size(0))][:min(4, len(fps))]
            fns = fns[torch.randperm(fns.size(0))][:min(4, len(fns))]

            # Plot images
            show_batch(tps, [1,1,1,1], [1,1,1,1], "True positives", True, "tps.png")
            show_batch(tns, [0,0,0,0], [0,0,0,0], "True negatives", True, "tns.png")
            show_batch(fps, [0,0,0,0], [1,1,1,1], "False positives", True, "fps.png")
            show_batch(fns, [1,1,1,1], [0,0,0,0], "False negatives", True, "fns.png")

            print(f'Accuracy: {self.accuracy} | TP: {tp}, TN: {tn}, fp: {fp}, FN: {fn}')

            return tp, tn, fp, fn


def load_model(path):
    '''Load model'''
    if not os.path.exists(path):
        raise ValueError(f'File {path} does not exist')
    dict = torch.load(path)
    model = BaseNetwork(dict["opt"])
    model.load_state_dict(dict["model_state_dict"])
    model.optimizer.load_state_dict(dict["optimizer_state_dict"])
    model.dtype = dict["dtype"]
    model.epoch = dict["epoch"]
    model.total_steps = dict["total_steps"]
    model.best_accuracy = dict["best_accuracy"]
    model.init_function = dict["init_function"]
    model.opt = dict["opt"]
    model.best_matrix = dict["matrix"]
    print(f'Loaded model {path} with accuracy {model.best_accuracy}')
    return model
