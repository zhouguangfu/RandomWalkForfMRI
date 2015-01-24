"""
Demo of the histogram (hist) function with a few features.

In addition to the basic histogram, this demo shows a few optional features:

    * Setting the number of data bins
    * The ``normed`` flag, which normalizes bin heights so that the integral of
      the histogram is 1. The resulting histogram is a probability density.
    * Setting the face color of the bars
    * Setting the opacity (alpha value).

"""
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from pylab import *

TEMP_IMG_DIR = 'G:/workingdir/BAA/seven/result/temp.png'

def show_date_index_formatter(x, y, xlabel, ylabel, title, color, label, clear=False):
    if  clear:
        plt.clf()
    plt.plot(x, y, color + 'o-', alpha=0.5, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc=4)
    plt.savefig(TEMP_IMG_DIR)
