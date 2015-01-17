__author__ = 'zhouguangfu'

import datetime
import numpy as np
import nibabel as nib
import multiprocessing
import os

from docx import Document
from docx.shared import Inches
from configs import *
from analysis.inter_subject_bar import show_barchart
from skimage.segmentation import random_walker

SUBJECT_SESSION_INDEX = 0 #0, 1, 2, 3, ,4 ,5, 6, 7, 8, 9
SESSION_NUMBERS = 7

BACKGROUND_MAKRERS_THR = [-3, -2, -1, 0, 1, 2, 3] #len 7 default - (-1)
OBJECT_MARKERS_NUM = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] #len 10 default - 30
# ATLAS_SELECTED = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,150] #len 12 default - 30
ATLAS_SELECTED = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] #len 12 default - 30

DEFAULT_TOP_RANK = 10 # 0 - 100, default
DEFAULT_Z_TOP = 60 #default 60
DEFAULT_BACKGROUND_THR = -1 #default -1

#global varibale
image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

complete_atlas_data = nib.load(ATLAS_TOP_DIR + 'complete_atlas_label.nii.gz').get_data()
left_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_LEFT_BRAIN_FILE).get_data()
right_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_RIGHT_BRAIN_FILE).get_data()

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
        weighted_result.append(np.average(temp, axis=3, weights=weight))
        print 'subject_index: ', subject_index, '   roi_index: ', roi_index

    return weighted_result

def process_single_subject(subject_index):
    global image, complete_atlas_data, left_barin_mask, right_barin_mask

    region_result_RW = np.zeros((complete_atlas_data.shape[0], complete_atlas_data.shape[1], complete_atlas_data.shape[2], DEFAULT_TOP_RANK))

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

        #process right brain
        markers = np.zeros_like(image[..., subject_index])

        markers[right_barin_mask == 0] = -1
        temp_volume = image[..., subject_index].copy()
        temp_volume[right_barin_mask == 0] = 1000
        temp_volume[temp_volume > DEFAULT_BACKGROUND_THR] = 1000
        markers[temp_volume <= DEFAULT_BACKGROUND_THR] = 3 #background

        z_atlas_mask = np.zeros_like(atlas_data)
        if (atlas_data == 1).sum() <= DEFAULT_Z_TOP :
            z_atlas_mask[atlas_data == 1] = 1
        else:
            temp_image = image[atlas_data == 1, subject_index]
            threshold = -np.sort(-temp_image)[DEFAULT_Z_TOP]
            z_atlas_mask[image[..., subject_index] > threshold] = 1
            z_atlas_mask[atlas_data != 1] = 0
        markers[z_atlas_mask == 1] = 1

        z_atlas_mask = np.zeros_like(atlas_data)
        if (atlas_data == 3).sum() <= DEFAULT_Z_TOP :
            z_atlas_mask[atlas_data == 3] = 1
        else:
            temp_image = image[atlas_data == 3, subject_index]
            threshold = -np.sort(-temp_image)[DEFAULT_Z_TOP]
            z_atlas_mask[image[..., subject_index] > threshold] = 1
            z_atlas_mask[atlas_data != 3] = 0
        markers[z_atlas_mask == 1] = 2


        # markers[ atlas_data == 1] = 1
        # markers[ atlas_data == 3] = 2

        rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
        region_result_RW[rw_labels == 1, atlas_index] = 1
        region_result_RW[rw_labels == 2, atlas_index] = 3
        region_result_RW[rw_labels == 3, atlas_index] = 5

        #----------------------------------------------------------------------------------
        #process left brain
        markers = np.zeros_like(image[..., subject_index])

        markers[left_barin_mask == 0] = -1
        temp_volume = image[..., subject_index].copy()
        temp_volume[left_barin_mask == 0] = 1000
        temp_volume[temp_volume > DEFAULT_BACKGROUND_THR] = 1000
        markers[temp_volume <= DEFAULT_BACKGROUND_THR] = 3 #background

        z_atlas_mask = np.zeros_like(atlas_data)
        if (atlas_data == 2).sum() <= DEFAULT_Z_TOP :
            z_atlas_mask[atlas_data == 2] = 1
        else:
            temp_image = image[atlas_data == 2, subject_index]
            threshold = -np.sort(-temp_image)[DEFAULT_Z_TOP]
            z_atlas_mask[image[..., subject_index] > threshold] = 1
            z_atlas_mask[atlas_data != 2] = 0
        markers[z_atlas_mask == 1] = 1

        z_atlas_mask = np.zeros_like(atlas_data)
        if (atlas_data == 4).sum() <= DEFAULT_Z_TOP :
            z_atlas_mask[atlas_data == 4] = 1
        else:
            temp_image = image[atlas_data == 4, subject_index]
            threshold = -np.sort(-temp_image)[DEFAULT_Z_TOP]
            z_atlas_mask[image[..., subject_index] > threshold] = 1
            z_atlas_mask[atlas_data != 4] = 0
        markers[z_atlas_mask == 1] = 2

        # markers[ atlas_data == 2] = 1
        # markers[ atlas_data == 4] = 2

        rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
        region_result_RW[rw_labels == 1, atlas_index] = 2
        region_result_RW[rw_labels == 2, atlas_index] = 4
        region_result_RW[rw_labels == 3, atlas_index] = 5

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
    for thr in ATLAS_SELECTED:

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



    print 'all_means: ', all_means
    print 'all_stds: ', all_stds

    all_means = np.array(all_means)
    all_stds = np.array(all_stds)

    import matplotlib.pyplot as plt

    plt.plot(OBJECT_MARKERS_NUM, all_means[:, 0].tolist(), '--ro')
    plt.plot(OBJECT_MARKERS_NUM, all_means[:, 1].tolist(), '--go')
    plt.plot(OBJECT_MARKERS_NUM, all_means[:, 2].tolist(), '--bo')
    plt.plot(OBJECT_MARKERS_NUM, all_means[:, 3].tolist(), '--yo')
    plt.show()

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."

































