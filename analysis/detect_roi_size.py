__author__ = 'zgf'
import datetime
import numpy as np
import nibabel as nib

from configs import *
from skimage.segmentation import random_walker

#global varibale
#random walk method result
# image = nib.load(ANALYSIS_DIR + 'rw/rw_prob_result_file.nii.gz').get_data()

#manual method result
image = nib.load(ANALYSIS_DIR + 'manual/' + 'manual.nii.gz').get_data()

if __name__ == "__main__":
    starttime = datetime.datetime.now()

    all_means = []
    all_stds = []
    region_sizes = np.zeros((image.shape[3], len(ROI)))
    for roi_index in range(len(ROI)):
        all_subject_session = (image == (roi_index + 1)).astype(np.int32)
        for i in range(image.shape[3]):
            # result = np.sum(all_subject_session.reshape(), axis=3)
            # print 'i: ', i, '   ' + ROI[roi_index] + '_sum:', all_subject_session[..., i].sum()
            region_sizes[i, roi_index] = all_subject_session[..., i].sum()
        all_means.append(region_sizes[:, roi_index].mean())
        all_stds.append(region_sizes[:, roi_index].std())

    # print ROI[0], region_sizes[:, 0].mean(), region_sizes[:, 0].std()
    # print ROI[1], region_sizes[:, 1].mean(), region_sizes[:, 1].std()
    # print ROI[2], region_sizes[:, 2].mean(), region_sizes[:, 2].std()
    # print ROI[3], region_sizes[:, 3].mean(), region_sizes[:, 3].std()
    # print '------------------------------------------------------'

    # import matplotlib.pyplot as plt
    # n_groups = len(all_means)
    # index = np.arange(n_groups)
    # bar_width = 0.2
    # opacity = 0.4
    # error_config = {'ecolor': '0.3'}
    #
    # rects1 = plt.bar(index, all_means, bar_width,
    #                  alpha=opacity,
    #                  color='b',
    #                  yerr=all_stds,
    #                  error_kw=error_config)
    #
    # plt.xlabel('ROI Name')
    # plt.ylabel('Region Size')
    # plt.title('All ROIs Region Size')
    # plt.xticks(index + bar_width / 2, ROI)
    # plt.tight_layout()
    # plt.legend(loc=4)
    # plt.show()

    all_means = []
    all_stds = []
    for roi_index in range(len(ROI)):
        means = []
        stds = []
        for j in range(len(SUBJECT_NAMES)):
            session_region_sizes = region_sizes[j * 7 : (j + 1) * 7, roi_index]
            means.append(session_region_sizes.mean())
            stds.append(session_region_sizes.std())
        all_means.append(means)
        all_stds.append(stds)

    roi_index = 3

    import matplotlib.pyplot as plt
    n_groups = len(all_means[roi_index])
    index = np.arange(n_groups)
    bar_width = 0.2
    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    rects1 = plt.bar(index, all_means[0], bar_width,
                     alpha=opacity,
                     color='b',
                     yerr=all_stds[0],
                     error_kw=error_config,
                     label='r_OFA')
    rects1 = plt.bar(index + bar_width, all_means[1], bar_width,
                     alpha=opacity,
                     color='g',
                     yerr=all_stds[1],
                     error_kw=error_config,
                     label='l_OFA')
    rects1 = plt.bar(index + bar_width * 2, all_means[2], bar_width,
                     alpha=opacity,
                     color='r',
                     yerr=all_stds[2],
                     error_kw=error_config,
                     label='r_pFus')
    rects1 = plt.bar(index + bar_width * 3, all_means[3], bar_width,
                     alpha=opacity,
                     color='c',
                     yerr=all_stds[3],
                     error_kw=error_config,
                     label='l_pFus')

    plt.xlabel('Subject Name')
    plt.ylabel('Region Size')
    plt.title('ROI Region Size')
    plt.xticks(index + bar_width, SUBJECT_NAMES)
    plt.tight_layout()
    plt.legend(loc=2)
    plt.show()

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
































