__author__ = 'zgf'

import numpy as np
import matplotlib.pyplot as plt
from configs import *

def show_barchart(x, std_x, xlabel, ylabel, title, color, legend_labels, ROI):
    plt.clf()
    n_groups = len(x[0])
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.6
    error_config = {'ecolor': '0.3'}

    for i in range(len(x)):
        plt.bar(index + i*bar_width, x[i], bar_width,
                alpha=opacity,
                color=color[i],
                yerr=std_x[i],
                error_kw=error_config,
                label=legend_labels[i])

        # plt.errorbar(index + i*bar_width,
        #             x[i],
        #             alpha=opacity,
        #             color=color[i],
        #             yerr=std_x[i],
        #             fmt='--',
        #             marker='o',
        #             mec='blue',
        #             label=legend_labels[i])

    axes = plt.gca()
    axes.set_ylim(0, 1.2)
    plt.grid()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(index + len(x[0]) * bar_width/4, ROI)
    plt.legend()

    plt.tight_layout()
    # plt.show()

    plt.savefig(TEMP_IMG_DIR)

# ROI = ['rOFA', 'lOFA', 'rFFA', 'lFFA']
# legend_labels = ['Manual', 'GSS', 'AC', 'Mask_WSAC']
# color = ['b', 'r', 'g', 'm']
#
# xlabel = 'X Label'
# ylabel = 'Y Label'
# title = 'The Title'
#
# means = []
# vars = []
# mean1 = [0.32537, 0.18451, 0.47457, 0.30718]
# var1 = [0.015817442, 0.039065719, 0.01348288, 0.036834124]
#
# mean2 = [0.38407, 0.28764, 0.55483, 0.44633]
# var2 = [0.016259347, 0.046853605, 0.013837051, 0.047415325]
#
# mean3 = [0.51629, 0.42523, 0.65469, 0.57537]
# var3 = [0.031672557, 0.038038433, 0.021821297, 0.051407585]
#
# mean4 = [0.59615, 0.49581, 0.70982, 0.66702]
# var4 = [0.025917054, 0.08043669, 0.032131855, 0.073176066]
#
# means.append(mean1)
# means.append(mean2)
# means.append(mean3)
# means.append(mean4)
#
# vars.append(var1)
# vars.append(var2)
# vars.append(var3)
# vars.append(var4)
#
# stds = []
# for i in range(len(color)):
#     std = []
#     for var in vars[i]:
#         std.append(np.sqrt(float(var)))
#     stds.append(std)
#
# show_barchart(means, stds, xlabel, ylabel, title, color, legend_labels, ROI)
