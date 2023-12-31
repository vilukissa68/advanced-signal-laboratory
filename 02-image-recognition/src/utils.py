#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import sys, os

def blockPrint():
    """Disable print"""
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    """Enable print"""
    sys.stdout = sys.__stdout__

def show_image(image, label):
    '''Show a single image with plt show'''
    plt.imshow(image.permute(1, 2, 0))
    plt.title(label)
    plt.show()

def show_batch(images, labels, predictions=None, title=None, save=False, filename=None):
    '''Show a batch of images with plt show'''
    print(images.shape[0])
    print("labels: ", labels)
    print("predictions: ", predictions)
    s = int(np.sqrt(images.shape[0]))
    fig, axs = plt.subplots(s, s, figsize=(s, s))

    for index, ax in enumerate(axs.flat):
        ax.imshow(images[index].permute(1, 2, 0))
        if predictions is None:
            title = "GT:{0}".format(labels[index])
        else:
            # Check tp tn fp fn
            if predictions[index] == labels[index]:
                if predictions[index] == 1:
                    title = "True Positive"
                else:
                    title = "True Negative"
            else:
                if predictions[index] == 1:
                    title = "False Positive"
                else:
                    title = "False Negative"
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    if save:
        if filename is None:
            filename = "batch.png"
        else:
            # Set figure size half of the default
            fig.set_size_inches(plt.rcParams["figure.figsize"][0] / 1.5, plt.rcParams["figure.figsize"][1] / 1.5)
            plt.savefig(filename, format='png', dpi=200)

    plt.tight_layout()
    plt.show()

def get_metrics(predictions, labels):
    # Capture the total number of true positives, true negatives, false positives, and false negatives
    tp = [1 if p == 1 and l == 1 else 0 for p, l in zip(predictions, labels)].count(1)
    tn = [1 if p == 0 and l == 0 else 0 for p, l in zip(predictions, labels)].count(1)
    fp = [1 if p == 1 and l == 0 else 0 for p, l in zip(predictions, labels)].count(1)
    fn = [1 if p == 0 and l == 1 else 0 for p, l in zip(predictions, labels)].count(1)
    return tp, tn, fp ,fn

def draw_matrix(tp, tn, fp, fn):
    # Create the data array
    data = np.array([[tp, fp], [fn, tn]])

    # Plot the confusion matrix
    heatmap = plt.pcolor(data, cmap='Blues')

    # Add colorbar legend
    plt.colorbar(heatmap)

    # Add axis labels and ticks
    plt.xticks([0.5, 1.5], ['Positive', 'Negative'])
    plt.yticks([0.5, 1.5], ['Positive', 'Negative'])
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')

    # Add text annotations for each cell
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j + 0.5, i + 0.5, str(data[i, j]), ha='center', va='center')

    # Save as svg
    plt.savefig('confusion_matrix.svg', format='svg', dpi=1200)

    # Show the plot
    plt.show()


def precision(tp, fp):
    return tp / (tp + fp)

def accuracy(tp, tn, fp, fn):
    return (tp + tn) / (tp + tn + fp + fn)

def recall(tp, fn):
    return tp / (tp + fn)

def f1_score(tp, fp, fn):
    return 2 * (precision(tp, fp) * recall(tp, fn)) / (precision(tp, fp) + recall(tp, fn))
