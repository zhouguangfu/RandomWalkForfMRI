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
# thin_foreground_image = nib.load(ATLAS_TOP_DIR + 'all_sessions_skeletonize_foreground.nii.gz').get_data()


def get_peak_point_cord(subject_image, mask):
    temp = np.zeros_like(subject_image)
    temp[mask] = subject_image[mask]
    peak_cord = np.unravel_index(temp.argmax(), temp.shape)

    return peak_cord

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

        r_OFA_flag = False
        l_OFA_flag = False
        r_pFus_flag = False
        l_pFus_flag = False

        #********************************************* right brain process ********************************************
        #--------------r_OFA---------------
        markers = np.zeros_like(image[..., subject_index])
        # atlas_roi_mask = np.logical_and(atlas_data == 1, thin_foreground_image[..., subject_index] == 1)
        # if atlas_roi_mask.sum() == 0:
        #     r_OFA_flag = True
        # else:
        #     markers[atlas_roi_mask] = 1

        atlas_roi_mask = (atlas_data == 1)
        if atlas_roi_mask.sum() == 0:
            r_OFA_flag = True
        else:
            roi_peak_cord = get_peak_point_cord(image[..., subject_index], atlas_roi_mask)
            markers[roi_peak_cord[0], roi_peak_cord[1], roi_peak_cord[2]] = 1

        #--------------r_pFus-------------
        # atlas_roi_mask = np.logical_and(atlas_data == 3, thin_foreground_image[..., subject_index] == 1)
        # if atlas_roi_mask.sum() <= 0:
        #     r_pFus_flag = True
        # else:
        #     markers[atlas_roi_mask] = 2

        atlas_roi_mask = (atlas_data == 3)
        if atlas_roi_mask.sum() <= 0:
            r_pFus_flag = True
        else:
            roi_peak_cord = get_peak_point_cord(image[..., subject_index], atlas_roi_mask)
            markers[roi_peak_cord[0], roi_peak_cord[1], roi_peak_cord[2]] = 2

        if r_OFA_flag and r_pFus_flag:
            region_result_RW[right_barin_mask > 0] = 5
        elif r_OFA_flag:
            markers[markers == 2] = 1
            markers[thin_background_image[..., subject_index] == 1] = 2
            markers[right_barin_mask == False] = -1

            rw_labels = random_walker(image[..., subject_index], markers, beta=130, mode='cg')
            region_result_RW[rw_labels == 1] = 3
            region_result_RW[rw_labels == 2] = 5

            skeletonize_markers_RW[markers == 1] = 3
            skeletonize_markers_RW[markers == 2] = 5
        elif r_pFus_flag:
            markers[thin_background_image[..., subject_index] == 1] = 2
            markers[right_barin_mask == False] = -1

            rw_labels = random_walker(image[..., subject_index], markers, beta=130, mode='cg')
            region_result_RW[rw_labels == 1] = 1
            region_result_RW[rw_labels == 2] = 5

            skeletonize_markers_RW[markers == 1] = 1
            skeletonize_markers_RW[markers == 2] = 5
        else:
            markers[thin_background_image[..., subject_index] == 1] = 3
            markers[right_barin_mask == False] = -1

            rw_labels = random_walker(image[..., subject_index], markers, beta=130, mode='cg')
            rw_labels[rw_labels == -1] = 0
            region_result_RW[rw_labels == 1] = 1
            region_result_RW[rw_labels == 2] = 3
            region_result_RW[rw_labels == 3] = 5

            skeletonize_markers_RW[markers == 1] = 1
            skeletonize_markers_RW[markers == 2] = 3
            skeletonize_markers_RW[markers == 3] = 5



        #********************************************* left brain process *********************************************
        #-------------------l_OFA--------------------
        markers = np.zeros_like(image[..., subject_index])

        # atlas_roi_mask = np.logical_and(atlas_data == 2, thin_foreground_image[..., subject_index] == 1)
        # if atlas_roi_mask.sum() <= 0:
        #     l_OFA_flag = True
        # else:
        #     markers[atlas_roi_mask] = 1

        atlas_roi_mask = (atlas_data == 2)
        if atlas_roi_mask.sum() <= 0:
            l_OFA_flag = True
        else:
            roi_peak_cord = get_peak_point_cord(image[..., subject_index], atlas_roi_mask)
            markers[roi_peak_cord[0], roi_peak_cord[1], roi_peak_cord[2]] = 1

         #-------------------l_pFus-------------------
        # atlas_roi_mask = np.logical_and(atlas_data == 4, thin_foreground_image[..., subject_index] == 1)
        # if atlas_roi_mask.sum() <= 0:
        #     l_pFus_flag = True
        # else:
        #     markers[atlas_roi_mask] = 2
        atlas_roi_mask = (atlas_data == 4)
        if atlas_roi_mask.sum() <= 0:
            l_pFus_flag = True
        else:
            roi_peak_cord = get_peak_point_cord(image[..., subject_index], atlas_roi_mask)
            markers[roi_peak_cord[0], roi_peak_cord[1], roi_peak_cord[2]] = 2


        if l_OFA_flag and l_pFus_flag:
            region_result_RW[left_barin_mask > 0] = 5
        elif l_OFA_flag:
            markers[markers == 2] = 1
            markers[thin_background_image[..., subject_index] == 1] = 2
            markers[left_barin_mask == False] = -1

            rw_labels = random_walker(image[..., subject_index], markers, beta=130, mode='cg')
            region_result_RW[rw_labels == 1] = 4
            region_result_RW[rw_labels == 2] = 5

            skeletonize_markers_RW[markers == 1] = 4
            skeletonize_markers_RW[markers == 2] = 5
        elif l_pFus_flag:
            markers[thin_background_image[..., subject_index] == 1] = 2
            markers[left_barin_mask == False] = -1
            rw_labels = random_walker(image[..., subject_index], markers, beta=130, mode='cg')
            region_result_RW[rw_labels == 1] = 2
            region_result_RW[rw_labels == 2] = 5

            skeletonize_markers_RW[markers == 1] = 2
            skeletonize_markers_RW[markers == 2] = 5
        else:
            markers[thin_background_image[..., subject_index] == 1] = 3
            markers[left_barin_mask == False] = -1

            rw_labels = random_walker(image[..., subject_index], markers, beta=130, mode='cg')
            rw_labels[rw_labels == -1] = 0
            region_result_RW[rw_labels == 1] = 2
            region_result_RW[rw_labels == 2] = 4
            region_result_RW[rw_labels == 3] = 5

            skeletonize_markers_RW[markers == 1] = 2
            skeletonize_markers_RW[markers == 2] = 4
            skeletonize_markers_RW[markers == 3] = 5



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

    #Multi rocess begin
    print 'Multi process begin...'

    DIR_PREFIX = 'subjects_rw_all_atlas_results/'
    RW_AGGRAGATOR_RESULT_DATA_DIR = RW_AGGRAGATOR_RESULT_DATA_DIR + DIR_PREFIX
    if not os.path.exists(RW_AGGRAGATOR_RESULT_DATA_DIR):
        os.makedirs(RW_AGGRAGATOR_RESULT_DATA_DIR)

    # #For single process
    # for subject_index in range(SUBJECTS_SESSION_NUMBERS):
    #     process_single_subject(subject_index)
    # process_single_subject(0)

    #For multi process
    starttime = datetime.datetime.now()
    process_num = 14
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

































