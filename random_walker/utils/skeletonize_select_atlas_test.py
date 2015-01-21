__author__ = 'zhouguangfu'

import datetime
import numpy as np
import nibabel as nib
import csv

from configs import *
from skimage import morphology
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

    # image[image <= 0] = 0
    # image[image > 0] = 1

    # image[image >= 0] = 0
    # image[image < 0] = 1
    #
    # thin_image = np.zeros_like(image)
    # for i in range(image.shape[3]):
    #     for j in range(image.shape[2]):
    #         slice_image = morphology.skeletonize(image[..., j, i]).astype(np.uint8)
    #         thin_image[..., j, i] = slice_image
    #     print 'i: ', i

    # nib.save(nib.Nifti1Image(thin_image, affine), ATLAS_TOP_DIR + 'all_sessions_background_skeletonize.nii.gz')
    # nib.save(nib.Nifti1Image(thin_image, affine), ATLAS_TOP_DIR + 'all_202_image_skeletonize.nii.gz')
    # nib.save(nib.Nifti1Image(thin_image, affine), ATLAS_TOP_DIR + 'all_sessions_skeletonize.nii.gz')


    #see the size of the certain ROI
    thin_background_image = nib.load(ATLAS_TOP_DIR + 'all_sessions_background_skeletonize.nii.gz').get_data()
    thin_foreground_image = nib.load(ATLAS_TOP_DIR + 'all_sessions_skeletonize.nii.gz').get_data()

    for i in range(image.shape[3]):
         background = [thin_background_image[r_OFA_mask > 0, i].sum(),
                       thin_background_image[l_OFA_mask > 0, i].sum(),
                       thin_background_image[r_pFus_mask > 0, i].sum(),
                       thin_background_image[l_pFus_mask > 0, i].sum()]

         foreground = [thin_foreground_image[r_OFA_mask > 0, i].sum(),
                       thin_foreground_image[l_OFA_mask > 0, i].sum(),
                       thin_foreground_image[r_pFus_mask > 0, i].sum(),
                       thin_foreground_image[l_pFus_mask > 0, i].sum()]

         print 'i: ', i, '   back: ', background, '   fore: ', foreground


    markers = np.zeros_like(image[..., 0])
    markers[image[..., 0] == 0] = -1
    markers[thin_foreground_image[..., 0] == 1] = 1
    markers[thin_background_image[..., 0] == 1] = 2

    rw_labels = random_walker(image[..., 0], markers, beta=10, mode='bf')
    rw_labels[rw_labels == -1] = 0
    nib.save(nib.Nifti1Image(rw_labels, affine), ATLAS_TOP_DIR + 'all_sessions_skeletonize_test.nii.gz')



    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
































