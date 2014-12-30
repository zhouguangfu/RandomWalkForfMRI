__author__ = 'zgf'

import time
import numpy as np
import nibabel as nib

from skimage.segmentation import random_walker
from configs import *

if __name__ == "__main__":
    starttime = time.clock()
    roi_peak_points = np.load(RSRG_RESULT_AVERAGE_DATA_DIR + RESULT_NPY_FILE)

    image = nib.load(FOUR_D_DATA_DIR + ALL_SESSION_AVEARGE_FILE)
    affine = image.get_affine()
    data = image.get_data()

    labels = np.zeros_like(data)
    # for i in range(len(ROI)):
    for i in range(1):
        mask = nib.load(SUB_2006_ROI_PROB + ROI[0] + '_prob.nii.gz') #0 -- r_OFA
        mask = mask.get_data()

        markers = np.zeros_like(data[..., i])
        # data[mask == 0, i] = 0
        markers[mask < 0.1] = 1
        markers[mask > 0.45] = 2

        print (mask > 0.45).sum()
        print (mask < 0.1).sum() - (mask <= 0).sum()

        label = random_walker(data[..., i], markers, beta=10, mode='bf')
        labels[..., i] = label

        print 'i: ', i

    nib.save(nib.Nifti1Image(labels, affine), RW_RESULT_DATA_DIR +  'rw_test.nii.gz')

    print "Program end..."

































