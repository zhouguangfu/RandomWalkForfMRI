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


#Process subject data.
def process_single_subject(subject_index):
    global image, complete_atlas_data, left_barin_mask, right_barin_mask

    result_image = np.zeros_like(complete_atlas_data[..., 0 : ATLAS_NUM])
    result_skeletonize_image = np.zeros_like(complete_atlas_data[..., 0 : ATLAS_NUM])

    for atlas_index in range(ATLAS_NUM):
        atlas_data = np.zeros_like(complete_atlas_data[..., 0])
        region_result_RW = np.zeros_like(atlas_data)
        skeletonize_markers_RW = np.zeros_like(atlas_data)

        #Use all 202 atalses.
        atlas_data[complete_atlas_data[..., atlas_index] == 1] = 1
        atlas_data[complete_atlas_data[..., atlas_index] == 2] = 2
        atlas_data[complete_atlas_data[..., atlas_index] == 3] = 3
        atlas_data[complete_atlas_data[..., atlas_index] == 4] = 4

        #********************************************* right brain process ********************************************
        #--------------r_OFA---------------
        markers = np.zeros_like(image[..., subject_index])
        atlas_roi_mask = np.logical_and(atlas_data == 1, thin_foreground_image[..., subject_index] == 1)
        if atlas_roi_mask.sum() <= 0:
            atlas_roi_mask = np.logical_and(r_OFA_group_mask, thin_foreground_image[..., subject_index] == 1)
        markers[atlas_roi_mask] = 1
        if (markers == 1).sum() == 0:
            print '**********r_OFA*******', ' subject_index: ', subject_index, '   atlas_index: ', atlas_index
        #--------------r_pFus-------------
        atlas_roi_mask = np.logical_and(atlas_data == 3, thin_foreground_image[..., subject_index] == 1)
        if atlas_roi_mask.sum() <= 0:
            atlas_roi_mask = np.logical_and(r_pFus_group__mask, thin_foreground_image[..., subject_index] == 1)
        markers[atlas_roi_mask] = 2
        if (markers == 2).sum() == 0:
            print '*******r_pFus************', ' subject_index: ', subject_index, '   atlas_index: ', atlas_index

        back_image = image[np.logical_and(right_barin_mask, thin_background_image[..., subject_index] == 1), subject_index]
        # back_threshold = np.sort(back_image)[DEFAULT_BACKGROUND_THR]

        # markers[np.logical_and(image[..., subject_index] < back_threshold, thin_background_image[..., subject_index] == 1)] = 3
        markers[thin_background_image[..., subject_index] == 1] = 3
        markers[right_barin_mask == False] = -1
        skeletonize_markers_RW[markers == 1] = 1
        skeletonize_markers_RW[markers == 2] = 3
        skeletonize_markers_RW[markers == 3] = 5

        rw_labels = random_walker(image[..., subject_index], markers, beta=130, mode='cg_mg')
        rw_labels[rw_labels == -1] = 0
        region_result_RW[rw_labels == 1] = 1
        region_result_RW[rw_labels == 2] = 3
        region_result_RW[rw_labels == 3] = 5

        #********************************************* left brain process *********************************************
        #-------------------l_OFA--------------------
        markers = np.zeros_like(image[..., subject_index])

        atlas_roi_mask = np.logical_and(atlas_data == 2, thin_foreground_image[..., subject_index] == 1)
        if atlas_roi_mask.sum() <= 0:
            atlas_roi_mask = np.logical_and(l_OFA_group_mask, thin_foreground_image[..., subject_index] == 1)
            # fore_image = image[np.logical_and(atlas_data == 2, thin_foreground_image[..., subject_index] == 1), subject_index]
        # fore_threshold = -np.sort(-fore_image)[30]
        # print 'subject index: ', subject_index, 'atlas_index: ', atlas_index, '   l_OFA size: ', atlas_roi_mask.sum()
        markers[atlas_roi_mask] = 1
        if (markers == 1).sum() == 0:
            print '**********l_OFA*********', ' subject_index: ', subject_index, '   atlas_index: ', atlas_index

        #-------------------l_pFus-------------------
        atlas_roi_mask = np.logical_and(atlas_data == 4, thin_foreground_image[..., subject_index] == 1)
        if atlas_roi_mask.sum() <= 0:
            atlas_roi_mask = np.logical_and(l_pFus_group_mask, thin_foreground_image[..., subject_index] == 1)

        # fore_image = image[np.logical_and(atlas_data == 4, thin_foreground_image[..., subject_index] == 1), subject_index]
        # fore_threshold = -np.sort(-fore_image)[30]
        # print 'subject index: ', subject_index, 'atlas_index: ', atlas_index, '   l_pFus size: ', atlas_roi_mask.sum()
        markers[atlas_roi_mask] = 2

        if (markers == 2).sum() == 0:
            print '********l_pFus***********', ' subject_index: ', subject_index, '   atlas_index: ', atlas_index

        back_image = image[np.logical_and(left_barin_mask, thin_background_image[..., subject_index] == 1), subject_index]
        # back_threshold = np.sort(back_image)[DEFAULT_BACKGROUND_THR]

        # markers[np.logical_and(image[..., subject_index] < back_threshold, thin_background_image[..., subject_index] == 1)] = 3
        markers[thin_background_image[..., subject_index] == 1] = 3
        markers[left_barin_mask == False] = -1
        skeletonize_markers_RW[markers == 1] = 2
        skeletonize_markers_RW[markers == 2] = 4
        skeletonize_markers_RW[markers == 3] = 5

        rw_labels = random_walker(image[..., subject_index], markers, beta=130, mode='cg_mg')
        rw_labels[rw_labels == -1] = 0
        region_result_RW[rw_labels == 1] = 2
        region_result_RW[rw_labels == 2] = 4
        region_result_RW[rw_labels == 3] = 5

        result_image[..., atlas_index] = region_result_RW
        result_skeletonize_image[..., atlas_index] = skeletonize_markers_RW

        print 'subject_index: ', subject_index, '   atlas_index: ', atlas_index

    # #Save the result
    nib.save(nib.Nifti1Image(result_skeletonize_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                               '_markers_rw.nii.gz')
    nib.save(nib.Nifti1Image(result_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                              '_regions_rw.nii.gz')

    print 'subject_index:', subject_index, 'atlas-based rw finished...'

if __name__ == "__main__":
    starttime = datetime.datetime.now()

    img = nib.load(ALL_PROB_MASK)
    mask = img.get_data()
    affine = img.get_affine()

    if not os.path.exists(RW_AGGRAGATOR_RESULT_DATA_DIR):
        os.makedirs(RW_AGGRAGATOR_RESULT_DATA_DIR)

    #Multi rocess begin
    print 'Multi process begin...'

    DIR_PREFIX = 'subjects_rw_all_atlas_results/'
    if not os.path.exists(RW_AGGRAGATOR_RESULT_DATA_DIR + DIR_PREFIX):
        os.makedirs(RW_AGGRAGATOR_RESULT_DATA_DIR + DIR_PREFIX)
    RW_AGGRAGATOR_RESULT_DATA_DIR = RW_AGGRAGATOR_RESULT_DATA_DIR + DIR_PREFIX

    # #For single process
    # for subject_index in range(SUBJECTS_SESSION_NUMBERS):
    #     process_single_subject(subject_index)
    # process_single_subject(16)

    #For multi process
    starttime = datetime.datetime.now()
    process_num = 5
    for cycle_index in range(SUBJECTS_SESSION_NUMBERS / process_num):
        pool = multiprocessing.Pool(processes=process_num)
        pool_outputs = pool.map(process_single_subject, range(cycle_index * process_num,
                                                              (cycle_index + 1) * process_num))
        pool.close()
        pool.join()

        print 'Cycle index: ', cycle_index, 'Time cost: ', (datetime.datetime.now() - starttime)
        starttime = datetime.datetime.now()

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."

































