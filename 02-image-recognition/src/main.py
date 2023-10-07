#!/usr/bin/env python3

from options import Options
from model import BaseNetwork
from dataloader import get_dataloaders
from utils import enablePrint, blockPrint

def main():
    opt = Options().parse()

    if opt.silent:
        blockPrint()

    model = BaseNetwork(opt)
    model.print_network()

    train, test = get_dataloaders(opt)

    model.to(opt.device)
    model.trainining_loop(train, test)
    print("Training finished")
    print("Best accuracy: ", model.best_accuracy, " at epoch: ", model.best_epoch)

    if opt.silent:
        enablePrint()
        print("accuracy: ", model.best_accuracy, " epoch: ", model.best_epoch)


if __name__ == '__main__':
    main()
