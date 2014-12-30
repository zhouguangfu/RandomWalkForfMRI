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
import numpy as np
import matplotlib.pyplot as plt

RESULT_DATA_DIR = "G:/workingdir/result/asrg/doc/"
ASRG_RESULT_DOC_DATA_DIR = "G:/workingdir/result/asrg/doc/test/"
TEMP_IMG_DIR = 'G:/workingdir/result/asrg/doc/temp.png'

def show_barchart(x, y, std_x, std_y, xlabel, ylabel, title, ROI):

    n_groups = len(x)
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, x, bar_width,
                 alpha=opacity,
                 color='b',
                 yerr=std_x,
                 error_kw=error_config,
                 label='AC')

    rects2 = plt.bar(index + bar_width, y, bar_width,
                 alpha=opacity,
                 color='r',
                 yerr=std_y,
                 error_kw=error_config,
                 label='PC')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(index + bar_width, ROI)
    plt.legend()

    plt.tight_layout()

    # Tweak spacing to prevent clipping of ylabel
    plt.savefig(TEMP_IMG_DIR)
