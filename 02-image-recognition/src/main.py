#!/usr/bin/env python3

from options import Options
from model import BaseNetwork
from dataloader import get_dataloaders, serialize_all_in_dir, show_image
from utils import enablePrint, blockPrint, show_batch

def main():
    opt = Options().parse()

    if opt.silent:
        blockPrint()


    if opt.serialize:
        print("Serializing data")
        serialize_all_in_dir(opt)
        print("Finished serializing data")

    if opt.view_data:
        print("Viewing data")
        train, test = get_dataloaders(opt)
        for batch in train:
            show_batch(batch[0], batch[1])
        print("Finished viewing data")




    model = BaseNetwork(opt)
    model.print_network()

    if opt.train:
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
