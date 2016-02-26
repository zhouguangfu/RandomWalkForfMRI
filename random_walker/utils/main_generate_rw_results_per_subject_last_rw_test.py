__author__ = 'zhouguangfu'

import datetime
import multiprocessing
import os

import nibabel as nib
import numpy as np
from random_walker import random_walker

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


all_labels_rois_data = nib.load( RW_AGGRAGATOR_RESULT_DATA_DIR  +  '_peak_points_result.nii.gz').get_data()

#Process subject data.
def process_single_subject(subject_index):
    region_result_RW = np.zeros_like(image[..., 0])

    #********************************************* right brain process ********************************************
    #--------------r_OFA---------------
    markers = np.zeros_like(image[..., subject_index])
    markers[all_labels_rois_data[..., subject_index] == 1] = 1
    #--------------r_pFus-------------
    markers[all_labels_rois_data[..., subject_index] == 3] = 2

    markers[thin_background_image[..., subject_index] == 1] = 3
    markers[right_barin_mask == False] = -1
    rw_labels = random_walker(image[..., subject_index], markers, beta=130, tol=0.001, spacing=None, mode='bf')
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

    rw_labels = random_walker(image[..., subject_index], markers, beta=130, tol=0.001, spacing=None, mode='bf')
    rw_labels[rw_labels == -1] = 0
    region_result_RW[rw_labels == 1] = 2
    region_result_RW[rw_labels == 2] = 4

    print 'subject_index: ', subject_index

    return region_result_RW



if __name__ == "__main__":
    starttime = datetime.datetime.now()
    result_image = np.zeros_like(image)

    # for subject_index in range(image.shape[3]):
    #     result_image[..., subject_index] = process_single_subject(subject_index)

    process_num = 14
    for cycle_index in range(image.shape[3] / process_num):
        pool = multiprocessing.Pool(processes=process_num)
        pool_outputs = pool.map(process_single_subject, range(cycle_index * process_num,
                                                              (cycle_index + 1) * process_num))
        pool.close()
        pool.join()

        for i in range(len(pool_outputs)):
            result_image[..., cycle_index * process_num + i] = pool_outputs[i]

        print 'Cycle index: ', cycle_index, 'Time cost: ', (datetime.datetime.now() - starttime)
        starttime = datetime.datetime.now()

    #Save the result
    nib.save(nib.Nifti1Image(result_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + 'cube_1_peak_points_rw_result.nii.gz')

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."

































