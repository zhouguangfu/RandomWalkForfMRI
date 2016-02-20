__author__ = 'zhouguangfu'

import datetime
import multiprocessing
import os

import nibabel as nib
import numpy as np
from skimage.segmentation import random_walker

from configs import *

SESSION_NUMBERS = 7

DEFAULT_TOP_RANK = 30 #1 - 202
#global varibale
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

#Compute dice coefficient
def dice(volume1, volume2):
    if volume1.shape != volume2.shape:
        raise ValueError("Shape mismatch: volume1 and volume2 must have the same shape.")
    intersection = np.logical_and(volume1, volume2)
    if volume1.sum() + volume2.sum() == 0:
        return 0.0
    else:
        return 2. * intersection.sum() / (volume1.sum() + volume2.sum())

def process_single_subject(subject_index):
    global image, complete_atlas_data, left_barin_mask, right_barin_mask

    result_image = np.zeros((complete_atlas_data.shape[0], complete_atlas_data.shape[1],
                                  complete_atlas_data.shape[2], len(ROI) + 1))

    for atlas_index in range(DEFAULT_TOP_RANK):
        atlas_data = np.zeros_like(complete_atlas_data[..., 0])
        region_result_RW = np.zeros_like(atlas_data)
        skeletonize_markers_RW = np.zeros_like(atlas_data)

        # r_OFA_indexs =  np.load(ATLAS_TOP_DIR + ROI[0] + '_' + str(subject_index) + '_top_sort.npy')
        # l_OFA_indexs =  np.load(ATLAS_TOP_DIR + ROI[1] + '_' + str(subject_index) + '_top_sort.npy')
        # r_pFus_indexs =  np.load(ATLAS_TOP_DIR + ROI[2] + '_' + str(subject_index) + '_top_sort.npy')
        # l_pFus_indexs =  np.load(ATLAS_TOP_DIR + ROI[3] + '_' + str(subject_index) + '_top_sort.npy')

        # atlas_data[complete_atlas_data[..., r_OFA_indexs[atlas_index]] == 1] = 1
        # atlas_data[complete_atlas_data[..., l_OFA_indexs[atlas_index]] == 2] = 2
        # atlas_data[complete_atlas_data[..., r_pFus_indexs[atlas_index]] == 3] = 3
        # atlas_data[complete_atlas_data[..., l_pFus_indexs[atlas_index]] == 4] = 4

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

        #--------------r_pFus-------------
        atlas_roi_mask = np.logical_and(atlas_data == 3, thin_foreground_image[..., subject_index] == 1)
        if atlas_roi_mask.sum() <= 0:
            atlas_roi_mask = np.logical_and(r_pFus_group__mask, thin_foreground_image[..., subject_index] == 1)
        markers[atlas_roi_mask] = 2

        back_image = image[np.logical_and(right_barin_mask, thin_background_image[..., subject_index] == 1), subject_index]
        # back_threshold = np.sort(back_image)[DEFAULT_BACKGROUND_THR]

        # markers[np.logical_and(image[..., subject_index] < back_threshold, thin_background_image[..., subject_index] == 1)] = 3
        markers[thin_background_image[..., subject_index] == 1] = 3
        markers[right_barin_mask == False] = -1
        skeletonize_markers_RW[markers == 1] = 1
        skeletonize_markers_RW[markers == 2] = 3
        skeletonize_markers_RW[markers == 3] = 5

        rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
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

        #-------------------l_pFus-------------------
        atlas_roi_mask = np.logical_and(atlas_data == 4, thin_foreground_image[..., subject_index] == 1)
        if atlas_roi_mask.sum() <= 0:
            atlas_roi_mask = np.logical_and(l_pFus_group_mask, thin_foreground_image[..., subject_index] == 1)

        # fore_image = image[np.logical_and(atlas_data == 4, thin_foreground_image[..., subject_index] == 1), subject_index]
        # fore_threshold = -np.sort(-fore_image)[30]
        # print 'subject index: ', subject_index, 'atlas_index: ', atlas_index, '   l_pFus size: ', atlas_roi_mask.sum()
        markers[atlas_roi_mask] = 2

        back_image = image[np.logical_and(left_barin_mask, thin_background_image[..., subject_index] == 1), subject_index]
        # back_threshold = np.sort(back_image)[DEFAULT_BACKGROUND_THR]

        # markers[np.logical_and(image[..., subject_index] < back_threshold, thin_background_image[..., subject_index] == 1)] = 3
        markers[thin_background_image[..., subject_index] == 1] = 3
        markers[left_barin_mask == False] = -1
        skeletonize_markers_RW[markers == 1] = 2
        skeletonize_markers_RW[markers == 2] = 4
        skeletonize_markers_RW[markers == 3] = 5

        rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
        rw_labels[rw_labels == -1] = 0
        region_result_RW[rw_labels == 1] = 2
        region_result_RW[rw_labels == 2] = 4
        region_result_RW[rw_labels == 3] = 5

        #Aggragator process

        for roi_index in range(len(ROI) + 1):
            temp = np.zeros_like(region_result_RW)
            #The weight value is related to the image intensity.
            if roi_index == len(ROI):
                temp[region_result_RW == (roi_index + 1)] = (-image[region_result_RW == (roi_index + 1), subject_index] +
                                          np.abs(-image[region_result_RW == (roi_index + 1), subject_index].min())) / \
                                         (np.abs(-image[region_result_RW == (roi_index + 1), subject_index].min()) +
                                          np.abs(-image[region_result_RW == (roi_index + 1), subject_index].max()))
            else:
                temp[region_result_RW == (roi_index + 1)] = (image[region_result_RW == (roi_index + 1), subject_index] +
                                         np.abs(image[region_result_RW == (roi_index + 1), subject_index].min())) /\
                                         (np.abs(image[region_result_RW == (roi_index + 1), subject_index].min()) +
                                         np.abs(image[region_result_RW == (roi_index + 1), subject_index].max()))

            #Majority vote.
            # temp = temp * 1. / temp.sum()
            result_image[..., roi_index] = result_image[..., roi_index] + temp

        print 'subject_index: ', subject_index, '   atlas_index: ', atlas_index

    for roi_index in range(len(ROI) + 1):
        if roi_index < len(ROI):
            nib.save(nib.Nifti1Image(result_image[..., roi_index], affine), RW_AGGRAGATOR_RESULT_DATA_DIR +
                                     ROI[roi_index] + '_' + str(subject_index) + '_aggragator.nii.gz')
        else:
            nib.save(nib.Nifti1Image(result_image[..., roi_index], affine), RW_AGGRAGATOR_RESULT_DATA_DIR +
                                     'background_' + str(subject_index) + '_aggragator.nii.gz')

    # #Save the result
    # nib.save(nib.Nifti1Image(skeletonize_markers_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
    #                                                            '_skeletonize_markers_rw.nii.gz')
    # nib.save(nib.Nifti1Image(region_result_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
    #                                                           '_skeletonize_regions_rw.nii.gz')

    # print 'subject_index:', subject_index, '    atlas_index:', atlas_index
    print 'subject_index:', subject_index, 'atlas-based rw finished...'

if __name__ == "__main__":
    starttime = datetime.datetime.now()

    img = nib.load(ALL_PROB_MASK)
    mask = img.get_data()
    affine = img.get_affine()

    if not os.path.exists(RW_AGGRAGATOR_RESULT_DATA_DIR):
        os.makedirs(RW_AGGRAGATOR_RESULT_DATA_DIR)

    all_means = []
    all_stds = []

    #Multi rocess begin
    print 'Multi process begin...'
    for subject_index in range(10):
        DIR_PREFIX = str(DEFAULT_TOP_RANK) + '/'
        if not os.path.exists(RW_AGGRAGATOR_RESULT_DATA_DIR + DIR_PREFIX):
            os.makedirs(RW_AGGRAGATOR_RESULT_DATA_DIR + DIR_PREFIX)

        pool = multiprocessing.Pool(processes=SESSION_NUMBERS)
        pool_outputs = pool.map(process_single_subject, range(subject_index * SESSION_NUMBERS,
                                                              (subject_index + 1) * SESSION_NUMBERS))
        pool.close()
        pool.join()

        break

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."

































