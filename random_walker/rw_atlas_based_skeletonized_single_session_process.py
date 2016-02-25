__author__ = 'zhouguangfu'

import datetime
import multiprocessing
import os

import nibabel as nib
import numpy as np
from skimage.segmentation import random_walker

from configs import *

SUBJECT_SESSION_INDEX = 0 #0, 1, 2, 3, ,4 ,5, 6, 7, 8, 9
SESSION_NUMBERS = 7

BACKGROUND_MAKRERS_THR = [-3, -2, -1, 0, 1, 2, 3] #len 7 default - (-1)
OBJECT_MARKERS_NUM = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] #len 10 default - 30
# ATLAS_SELECTED = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,150] #len 12 default - 30
ATLAS_SELECTED = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] #len 12 default - 30

DEFAULT_TOP_RANK = 30 # 0 - 100, default
DEFAULT_Z_TOP = 60 #default 60
DEFAULT_BACKGROUND_THR = -1 #default -1

#global varibale
image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

complete_atlas_data = nib.load(ATLAS_TOP_DIR + 'complete_atlas_label.nii.gz').get_data()
left_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_LEFT_BRAIN_FILE).get_data()
right_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_RIGHT_BRAIN_FILE).get_data()

thin_background_image = nib.load(ATLAS_TOP_DIR + 'all_sessions_background_skeletonize.nii.gz').get_data()
thin_foreground_image = nib.load(ATLAS_TOP_DIR + 'all_sessions_skeletonize.nii.gz').get_data()

r_OFA_mask = nib.load(PROB_ROI_202_SUB_FILE + 'r_OFA_prob.nii.gz').get_data() > 0
l_OFA_mask = nib.load(PROB_ROI_202_SUB_FILE + 'l_OFA_prob.nii.gz').get_data() > 0
r_pFus_mask = nib.load(PROB_ROI_202_SUB_FILE + 'r_pFus_prob.nii.gz').get_data() > 0
l_pFus_mask = nib.load(PROB_ROI_202_SUB_FILE + 'l_pFus_prob.nii.gz').get_data() > 0

r_OFA_group_mask = nib.load(LABEL_ROI_202_SUB_FILE + 'r_OFA_label.nii.gz').get_data() > 0
l_OFA_group_mask = nib.load(LABEL_ROI_202_SUB_FILE + 'l_OFA_label.nii.gz').get_data() > 0
r_pFus_group__mask = nib.load(LABEL_ROI_202_SUB_FILE + 'r_pFus_label.nii.gz').get_data() > 0
l_pFus_group_mask = nib.load(LABEL_ROI_202_SUB_FILE + 'l_pFus_label.nii.gz').get_data() > 0

DEFAULT_BACKGROUND_THR = 200

def dice(volume1, volume2):
    if volume1.shape != volume2.shape:
        raise ValueError("Shape mismatch: volume1 and volume2 must have the same shape.")
    intersection = np.logical_and(volume1, volume2)
    if volume1.sum() + volume2.sum() == 0:
        return 0.0
    else:
        return 2. * intersection.sum() / (volume1.sum() + volume2.sum())

def atlas_based_aggragator(subject_index):
    region_result_RW = process_single_subject(subject_index)
    weight = np.ones(DEFAULT_TOP_RANK, dtype=float)
    weighted_result = []
    for roi_index in range(len(ROI) + 1):
        temp = np.zeros_like(region_result_RW)
        temp[region_result_RW == (roi_index + 1)] = 1

        if roi_index == len(ROI):
            for i in range(temp.shape[3]):
                temp_data = temp[..., i].copy()
                # temp[temp_data == 1, i] = 1.0 - (image[temp_data == 1, subject_index] + np.abs(image[temp_data == 1, subject_index].min())) /\
                #             (np.abs(image[temp_data == 1, subject_index].min()) + np.abs(image[temp_data == 1, subject_index].max()))
                temp[temp_data == 1, i] = (-image[temp_data == 1, subject_index] + np.abs(-image[temp_data == 1, subject_index].min())) /\
                            (np.abs(-image[temp_data == 1, subject_index].min()) + np.abs(-image[temp_data == 1, subject_index].max()))
        else:
            for i in range(temp.shape[3]):
                temp_data = temp[..., i].copy()
                temp[temp_data == 1, i] = (image[temp_data == 1, subject_index] + np.abs(image[temp_data == 1, subject_index].min())) /\
                            (np.abs(image[temp_data == 1, subject_index].min()) + np.abs(image[temp_data == 1, subject_index].max()))

        # if roi_index == len(ROI):
        #     for i in range(temp.shape[3]):
        #         temp_data = temp[..., i].copy()
        #         temp[temp_data == 1, i] = logistic.cdf(0 - image[temp_data == 1, subject_index])
        # else:
        #     for i in range(temp.shape[3]):
        #         temp_data = temp[..., i].copy()
        #         temp[temp_data == 1, i] = logistic.cdf(image[temp_data == 1, subject_index])

        weighted_result.append(np.average(temp, axis=3, weights=weight))

        if roi_index < len(ROI):
            nib.save(nib.Nifti1Image(weighted_result[roi_index], affine), RW_AGGRAGATOR_RESULT_DATA_DIR +
                                ROI[roi_index] + '_' + str(subject_index) + '_aggragator.nii.gz')
        else:
            nib.save(nib.Nifti1Image(weighted_result[roi_index], affine), RW_AGGRAGATOR_RESULT_DATA_DIR +
                                'background_' + str(subject_index) + '_aggragator.nii.gz')
        print 'subject_index: ', subject_index, '   roi_index: ', roi_index

    return weighted_result

def process_single_subject(subject_index):
    global image, complete_atlas_data, left_barin_mask, right_barin_mask

    region_result_RW = np.zeros((complete_atlas_data.shape[0], complete_atlas_data.shape[1], complete_atlas_data.shape[2], DEFAULT_TOP_RANK))
    skeletonize_markers_RW = np.zeros((complete_atlas_data.shape[0], complete_atlas_data.shape[1], complete_atlas_data.shape[2], DEFAULT_TOP_RANK))

    for atlas_index in range(DEFAULT_TOP_RANK):
        atlas_data = np.zeros_like(complete_atlas_data[..., 0])

        r_OFA_indexs =  np.load(ATLAS_TOP_DIR + ROI[0] + '_' + str(subject_index) + '_top_sort.npy')
        l_OFA_indexs =  np.load(ATLAS_TOP_DIR + ROI[1] + '_' + str(subject_index) + '_top_sort.npy')
        r_pFus_indexs =  np.load(ATLAS_TOP_DIR + ROI[2] + '_' + str(subject_index) + '_top_sort.npy')
        l_pFus_indexs =  np.load(ATLAS_TOP_DIR + ROI[3] + '_' + str(subject_index) + '_top_sort.npy')

        atlas_data[complete_atlas_data[..., r_OFA_indexs[atlas_index]] == 1] = 1
        atlas_data[complete_atlas_data[..., l_OFA_indexs[atlas_index]] == 2] = 2
        atlas_data[complete_atlas_data[..., r_pFus_indexs[atlas_index]] == 3] = 3
        atlas_data[complete_atlas_data[..., l_pFus_indexs[atlas_index]] == 4] = 4


        # markers[thin_foreground_image[..., i] == 1] = 1
        # markers[thin_background_image[..., i] == 1] = 2
        # markers[r_OFA_mask == False] = -1

        #right brain process
        #--------------r_OFA---------------
        markers = np.zeros_like(image[..., subject_index])
        atlas_roi_mask = np.logical_and(atlas_data == 1, thin_foreground_image[..., subject_index] == 1)
        if atlas_roi_mask.sum() <= 0:
            atlas_roi_mask = np.logical_and(r_OFA_group_mask, thin_foreground_image[..., subject_index] == 1)
        # fore_image = image[atlas_roi_mask, subject_index]
        # fore_threshold = -np.sort(-fore_image)[30]
        # print 'subject index: ', subject_index, 'atlas_index: ', atlas_index, '   r_OFA size: ', atlas_roi_mask.sum()
        markers[atlas_roi_mask] = 1

        #--------------r_pFus-------------
        atlas_roi_mask = np.logical_and(atlas_data == 3, thin_foreground_image[..., subject_index] == 1)
        if atlas_roi_mask.sum() <= 0:
            atlas_roi_mask = np.logical_and(r_pFus_group__mask, thin_foreground_image[..., subject_index] == 1)
        # fore_image = image[np.logical_and(atlas_data == 3, thin_foreground_image[..., subject_index] == 1), subject_index]
        # fore_threshold = -np.sort(-fore_image)[30]
        # print 'subject index: ', subject_index, 'atlas_index: ', atlas_index, '   r_pFus size: ', atlas_roi_mask.sum()
        markers[atlas_roi_mask] = 2

        back_image = image[np.logical_and(right_barin_mask, thin_background_image[..., subject_index] == 1), subject_index]
        # back_threshold = np.sort(back_image)[DEFAULT_BACKGROUND_THR]

        # markers[np.logical_and(image[..., subject_index] < back_threshold, thin_background_image[..., subject_index] == 1)] = 3
        markers[thin_background_image[..., subject_index] == 1] = 3
        markers[right_barin_mask == False] = -1
        skeletonize_markers_RW[markers == 1, atlas_index] = 1
        skeletonize_markers_RW[markers == 2, atlas_index] = 3
        skeletonize_markers_RW[markers == 3, atlas_index] = 5

        rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
        rw_labels[rw_labels == -1] = 0
        region_result_RW[rw_labels == 1, atlas_index] = 1
        region_result_RW[rw_labels == 2, atlas_index] = 3
        region_result_RW[rw_labels == 3, atlas_index] = 5

        #left brain process
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
        skeletonize_markers_RW[markers == 1, atlas_index] = 2
        skeletonize_markers_RW[markers == 2, atlas_index] = 4
        skeletonize_markers_RW[markers == 3, atlas_index] = 5

        rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
        rw_labels[rw_labels == -1] = 0
        region_result_RW[rw_labels == 1, atlas_index] = 2
        region_result_RW[rw_labels == 2, atlas_index] = 4
        region_result_RW[rw_labels == 3, atlas_index] = 5


    nib.save(nib.Nifti1Image(skeletonize_markers_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                               '_skeletonize_markers_rw.nii.gz')
    nib.save(nib.Nifti1Image(region_result_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                              '_skeletonize_regions_rw.nii.gz')

    # print 'subject_index:', subject_index, '    atlas_index:', atlas_index
    print 'subject_index:', subject_index, 'atlas-based rw finished...'

    return region_result_RW

def generate_rw_prob_result(rw_atlas_based_aggrator_result):
    #generate the prob result
    temp_image = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], SESSION_NUMBERS))
    for subject_index in range(SESSION_NUMBERS):
        for roi_index in range(len(ROI) + 1):
            temp_image[rw_atlas_based_aggrator_result[..., subject_index, roi_index] > 0 , subject_index] = 1

        coords = np.array(np.nonzero(temp_image[..., subject_index] == 1), dtype=np.int32).T
        for i in range(coords.shape[0]):
            temp_image[coords[i, 0], coords[i, 1], coords[i, 2], subject_index] = \
                np.array([rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 0],
                          rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 1],
                          rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 2],
                          rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 3],
                          rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 4]]).argmax() + 1
        print 'subject_index: ', subject_index
    temp_image[temp_image == 5] = 0
    nib.save(nib.Nifti1Image(temp_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + RW_PROB_RESULT_FILE)

    return temp_image

def compute(single_subject_sessions):
    dices = np.zeros((SESSION_NUMBERS, SESSION_NUMBERS)).astype(np.float) - 1

    for j in range(SESSION_NUMBERS - 1):
        for k in range(j + 1, SESSION_NUMBERS):
            dices[j, k] = dice(single_subject_sessions[..., j] > 0, single_subject_sessions[..., k] > 0)

    values = dices[dices >= 0]
    return values.mean(), values.std()

if __name__ == "__main__":
    starttime = datetime.datetime.now()

    img = nib.load(ALL_PROB_MASK)
    mask = img.get_data()
    affine = img.get_affine()

    if not os.path.exists(RW_AGGRAGATOR_RESULT_DATA_DIR):
        os.makedirs(RW_AGGRAGATOR_RESULT_DATA_DIR)

    all_means = []
    all_stds = []

    ORIGIN_RW_AGGRAGATOR_RESULT_DATA_DIR = RW_AGGRAGATOR_RESULT_DATA_DIR

    # for thr in BACKGROUND_MAKRERS_THR:
    # for thr in OBJECT_MARKERS_NUM:
    for session_index in range(10):
        SUBJECT_SESSION_INDEX = session_index

        for thr in ATLAS_SELECTED:
            thr = 30
            RW_AGGRAGATOR_RESULT_DATA_DIR = ORIGIN_RW_AGGRAGATOR_RESULT_DATA_DIR
            DIR_PREFIX = str(thr) + '/'
            if not os.path.exists(RW_AGGRAGATOR_RESULT_DATA_DIR + DIR_PREFIX):
                os.makedirs(RW_AGGRAGATOR_RESULT_DATA_DIR + DIR_PREFIX)
            RW_AGGRAGATOR_RESULT_DATA_DIR = RW_AGGRAGATOR_RESULT_DATA_DIR + DIR_PREFIX

            # DEFAULT_BACKGROUND_THR = thr
            # DEFAULT_Z_TOP = thr
            DEFAULT_TOP_RANK = thr
            print '---------------------------------', thr, '---------------------------------------'

            rw_atlas_based_aggrator_result = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], SESSION_NUMBERS, len(ROI) + 1), dtype=np.float)

            pool = multiprocessing.Pool(processes=SESSION_NUMBERS)
            pool_outputs = pool.map(atlas_based_aggragator, range(SUBJECT_SESSION_INDEX * SESSION_NUMBERS, (SUBJECT_SESSION_INDEX + 1) * SESSION_NUMBERS))
            pool.close()
            pool.join()

            for i in range(SESSION_NUMBERS):
                for roi_index in range(len(ROI) + 1):
                    rw_atlas_based_aggrator_result[..., i, roi_index] = pool_outputs[i][roi_index]

            # for roi_index in range(len(ROI) + 1):
            #     if roi_index == len(ROI):
            #         nib.save(nib.Nifti1Image(rw_atlas_based_aggrator_result[..., roi_index], affine),
            #                  RW_AGGRAGATOR_RESULT_DATA_DIR + RW_PROB_BACKGROUND_RESULT_FILE)
            #     else:
            #         nib.save(nib.Nifti1Image(rw_atlas_based_aggrator_result[..., roi_index], affine),
            #                  RW_AGGRAGATOR_RESULT_DATA_DIR + ROI[roi_index] + '_' +RW_ATLAS_BASED_AGGRATOR_RESULT_FILE)

            #generate rw prob result
            result_labels_file = generate_rw_prob_result(rw_atlas_based_aggrator_result)

            means = []
            stds = []
            for roi_index in range(0, len(ROI)):
            # for roi_index in range(0, 1):
                single_subject_sessions = (result_labels_file == (roi_index + 1)).astype(np.int32)
                mean, std = compute(single_subject_sessions)
                means.append(mean)
                stds.append(std)

            all_means.append(means)
            all_stds.append(stds)

            break

        print 'all_means: ', all_means
        print 'all_stds: ', all_stds
        print '--------------------------------- ', session_index, ' ----------------------------------------------------'

    # all_means = np.array(all_means)
    # all_stds = np.array(all_stds)
    #
    # import matplotlib.pyplot as plt
    #
    # plt.plot(OBJECT_MARKERS_NUM, all_means[:, 0].tolist(), '--ro')
    # plt.plot(OBJECT_MARKERS_NUM, all_means[:, 1].tolist(), '--go')
    # plt.plot(OBJECT_MARKERS_NUM, all_means[:, 2].tolist(), '--bo')
    # plt.plot(OBJECT_MARKERS_NUM, all_means[:, 3].tolist(), '--yo')
    # plt.show()

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
































