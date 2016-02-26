__author__ = 'zhouguangfu'

import datetime
import multiprocessing
import os

import nibabel as nib
import numpy as np
from skimage.segmentation import random_walker

from configs import *

SUBJECTS_SESSION_NUMBERS = 70
ATLAS_NUM = 202 #1 - 202

#Global varibale
image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

complete_atlas_data = nib.load(ATLAS_TOP_DIR + 'complete_atlas_label.nii.gz').get_data()
left_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_LEFT_BRAIN_FILE).get_data()
right_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_RIGHT_BRAIN_FILE).get_data()

#Load all subjects session's skeletonize data
thin_background_image = nib.load(ATLAS_TOP_DIR + 'all_sessions_skeletonize_background.nii.gz').get_data()
thin_foreground_image = nib.load(ATLAS_TOP_DIR + 'all_sessions_skeletonize_foreground.nii.gz').get_data()

r_OFA_mask = nib.load(PROB_ROI_202_SUB_FILE + 'r_OFA_prob.nii.gz').get_data() > 0
l_OFA_mask = nib.load(PROB_ROI_202_SUB_FILE + 'l_OFA_prob.nii.gz').get_data() > 0
r_pFus_mask = nib.load(PROB_ROI_202_SUB_FILE + 'r_pFus_prob.nii.gz').get_data() > 0
l_pFus_mask = nib.load(PROB_ROI_202_SUB_FILE + 'l_pFus_prob.nii.gz').get_data() > 0

r_OFA_group_mask = nib.load(LABEL_ROI_202_SUB_FILE + 'r_OFA_label.nii.gz').get_data() > 0
l_OFA_group_mask = nib.load(LABEL_ROI_202_SUB_FILE + 'l_OFA_label.nii.gz').get_data() > 0
r_pFus_group__mask = nib.load(LABEL_ROI_202_SUB_FILE + 'r_pFus_label.nii.gz').get_data() > 0
l_pFus_group_mask = nib.load(LABEL_ROI_202_SUB_FILE + 'l_pFus_label.nii.gz').get_data() > 0


all_labels_rois_data = nib.load( RW_AGGRAGATOR_RESULT_DATA_DIR  +  'a_peak_points_result.nii.gz').get_data()

#Process subject data.
def process_single_subject():
    result_image = np.zeros_like(image)

    for subject_index in range(result_image.shape[3]):
        region_result_RW = np.zeros_like(result_image[..., 0])

        #********************************************* right brain process ********************************************
        #--------------r_OFA---------------
        markers = np.zeros_like(image[..., subject_index])
        markers[all_labels_rois_data[..., subject_index] == 1] = 1
        #--------------r_pFus-------------
        markers[all_labels_rois_data[..., subject_index] == 3] = 2

        markers[thin_background_image[..., subject_index] == 1] = 3
        markers[right_barin_mask == False] = -1
        rw_labels = random_walker(image[..., subject_index], markers, beta=10, tol=0.001, spacing=None, mode='bf')
        rw_labels[rw_labels == -1] = 0
        region_result_RW[rw_labels == 1] = 1
        region_result_RW[rw_labels == 2] = 3

        #********************************************* left brain process *********************************************
        #-------------------l_OFA--------------------
        markers = np.zeros_like(image[..., subject_index])
        markers[all_labels_rois_data[..., subject_index] == 2] = 1

        #-------------------l_pFus-------------------
        markers[all_labels_rois_data[..., subject_index] == 4] = 2

        markers[thin_background_image[..., subject_index] == 1] = 3
        markers[left_barin_mask == False] = -1

        rw_labels = random_walker(image[..., subject_index], markers, beta=10, tol=0.001, spacing=None, mode='bf')
        rw_labels[rw_labels == -1] = 0
        region_result_RW[rw_labels == 1] = 2
        region_result_RW[rw_labels == 2] = 4

        result_image[..., subject_index] = region_result_RW

        print 'subject_index: ', subject_index

    # #Save the result
    nib.save(nib.Nifti1Image(result_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + '_peak_points_rw_result.nii.gz')

if __name__ == "__main__":
    starttime = datetime.datetime.now()
    process_single_subject()

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."

































