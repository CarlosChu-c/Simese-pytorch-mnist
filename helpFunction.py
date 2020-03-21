import matplotlib.pyplot as plt
import numpy as np


def plot_mnist(numpy_all, numpy_labels):
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']

    for i in range(10):
        f = numpy_all[np.where(numpy_labels == i)]
        plt.plot(f[:, 0], f[:, 1], '.', c=c[i])
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc=5)
    plt.savefig('distribution.png')
    plt.show()


def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.savefig('Iteration&Loss.png')
    plt.show()
