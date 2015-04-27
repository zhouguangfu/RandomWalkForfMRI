__author__ = 'zhouguangfu'

import datetime
import numpy as np
import nibabel as nib

from configs import *
from skimage.segmentation import random_walker

#global varibale
image = nib.load(ACTIVATION_DATA_DIR)
# image = nib.load(ALL_202_SUBJECTS_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

all_prob_mask = nib.load(ALL_PROB_MASK).get_data().astype(np.bool)

r_OFA_mask = nib.load(PROB_ROI_202_SUB_FILE + 'r_OFA_prob.nii.gz').get_data() > 0
l_OFA_mask = nib.load(PROB_ROI_202_SUB_FILE + 'l_OFA_prob.nii.gz').get_data() > 0
r_pFus_mask = nib.load(PROB_ROI_202_SUB_FILE + 'r_pFus_prob.nii.gz').get_data() > 0
l_pFus_mask = nib.load(PROB_ROI_202_SUB_FILE + 'l_pFus_prob.nii.gz').get_data() > 0

if __name__ == "__main__":
    starttime = datetime.datetime.now()

    thin_background_image = nib.load(ATLAS_TOP_DIR + 'all_sessions_background_skeletonize.nii.gz').get_data()
    thin_foreground_image = nib.load(ATLAS_TOP_DIR + 'all_sessions_skeletonize.nii.gz').get_data()

    markers = np.zeros_like(image[..., 0])
    markers[thin_foreground_image[..., 0] == 1] = 1
    markers[thin_background_image[..., 0] == 1] = 2
    markers[image[..., 0] == 0] = -1

    rw_labels = random_walker(image[..., 0], markers, beta=10, mode='bf')
    rw_labels[rw_labels == -1] = 0
    nib.save(nib.Nifti1Image(rw_labels, affine), ATLAS_TOP_DIR + 'test.nii.gz')




    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
































