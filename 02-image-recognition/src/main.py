#!/usr/bin/env python3

from options import Options
from model import BaseNetwork, load_model
from dataloader import get_dataloaders, serialize_all_in_dir, show_image
from utils import enablePrint, blockPrint, show_batch, draw_matrix
from smile_detection import smile_detection

def main():
    opt = Options().parse()
    model = None

    if opt.silent:
        # Disable printing to console when running from script
        blockPrint()

    if opt.serialize:
        # Serialize data for faster loading and data augmentation
        print("Serializing data")
        serialize_all_in_dir(opt)
        print("Finished serializing data")

    if opt.view_data:
        # View data
        print("Viewing data")
        train, test = get_dataloaders(opt)
        for batch in train:
            show_batch(batch[0], batch[1])
        print("Finished viewing data")

    if opt.test and opt.load_model:
        print("Testing model")
        # Do dry test and plot images
        train, test = get_dataloaders(opt)

        model = load_model(opt.weights)
        model.to(opt.device)
        tp, tn, fp, fn = model.dry_test(test)
        draw_matrix(tp, tn, fp, fn)
        return


    if opt.load_model:
        # Load model for smile detection
        model = load_model(opt.weights)
        model.to(opt.device)
        smile_detection(model, opt)


    if opt.train:
        # Train new model
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

    if opt.draw_matrix:
        # Draw confusion matrix for best model
        if not model:
            model = load_model(opt.weights)
            model.to(opt.device)
        lst = model.best_matrix
        print(lst)
        draw_matrix(lst[0], lst[1], lst[2], lst[3])



if __name__ == '__main__':
    main()
