#!/usr/bin/env python3

from options import Options
from model import BaseNetwork
from dataloader import get_dataloaders

def main():
    opt = Options().parse()
    model = BaseNetwork(opt)
    model.print_network()

    train, test = get_dataloaders(opt)

    model.to(opt.device)
    model.trainining_loop(train, test)
    print("Training finished")
    print("Best accuracy: ", model.best_accuracy)


if __name__ == '__main__':
    main()
