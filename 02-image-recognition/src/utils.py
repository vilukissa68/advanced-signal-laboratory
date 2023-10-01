#!/usr/bin/env python3
import matplotlib.pyplot as plt

def show_image(image):
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
