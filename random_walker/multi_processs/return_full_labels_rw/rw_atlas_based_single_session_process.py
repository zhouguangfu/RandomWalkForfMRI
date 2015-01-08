__author__ = 'zhouguangfu'

import datetime
import numpy as np
import nibabel as nib
import multiprocessing
import os

from skimage.segmentation import random_walker
from configs import *


SUBJECT_SESSION_INDEX = 0 #0? 7? 14? 21? 28? 35? 42? 49? 56? 63
SESSION_NUMBERS = 1

#global varibale
image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

complete_atlas_data = nib.load(ATLAS_TOP_DIR + 'complete_atlas_label.nii.gz').get_data()
left_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_LEFT_BRAIN_FILE).get_data()
right_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_RIGHT_BRAIN_FILE).get_data()

def atlas_based_aggragator(subject_index):
    region_result_RW = process_single_subject(subject_index)
    weight = np.ones(TOP_RANK, dtype=float)
    weighted_result = []
    for roi_index in range(len(ROI) + 1):
        temp = np.zeros_like(region_result_RW)
        temp[region_result_RW == (roi_index + 1)] = 1
        weighted_result.append(np.average(temp, axis=3, weights=weight))
        print 'subject_index: ', subject_index, '   roi_index: ', roi_index

    return weighted_result

def process_single_subject(subject_index):
    global image, complete_atlas_data, left_barin_mask, right_barin_mask

    indexs =  np.load(ATLAS_TOP_DIR + str(subject_index) + '_top_sort.npy')
    region_result_RW = np.zeros((complete_atlas_data.shape[0], complete_atlas_data.shape[1], complete_atlas_data.shape[2], TOP_RANK))

    for atlas_index in range(TOP_RANK):
        atlas_data = complete_atlas_data[..., indexs[atlas_index]]

        #process right brain
        markers = np.zeros_like(image[..., subject_index])

        markers[right_barin_mask == 0] = -1
        temp_volume = image[..., subject_index].copy()
        temp_volume[right_barin_mask == 0] = 1
        temp_volume[temp_volume > 0] = 1
        markers[temp_volume <= 0] = 3 #background

        z_atlas_mask = np.zeros_like(atlas_data)
        if (atlas_data == 1).sum() <= Z_TOP :
            z_atlas_mask[atlas_data == 1] = 1
        else:
            temp_image = image[atlas_data == 1, subject_index]
            threshold = -np.sort(-temp_image)[Z_TOP]
            z_atlas_mask[image[..., subject_index] > threshold] = 1
            z_atlas_mask[atlas_data != 1] = 0
        markers[z_atlas_mask == 1] = 1

        z_atlas_mask = np.zeros_like(atlas_data)
        if (atlas_data == 3).sum() <= Z_TOP :
            z_atlas_mask[atlas_data == 3] = 1
        else:
            temp_image = image[atlas_data == 3, subject_index]
            threshold = -np.sort(-temp_image)[Z_TOP]
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
        temp_volume[left_barin_mask == 0] = 1
        temp_volume[temp_volume > 0] = 1
        markers[temp_volume <= 0] = 3 #background

        z_atlas_mask = np.zeros_like(atlas_data)
        if (atlas_data == 2).sum() <= Z_TOP :
            z_atlas_mask[atlas_data == 2] = 1
        else:
            temp_image = image[atlas_data == 2, subject_index]
            threshold = -np.sort(-temp_image)[Z_TOP]
            z_atlas_mask[image[..., subject_index] > threshold] = 1
            z_atlas_mask[atlas_data != 2] = 0
        markers[z_atlas_mask == 1] = 1

        z_atlas_mask = np.zeros_like(atlas_data)
        if (atlas_data == 4).sum() <= Z_TOP :
            z_atlas_mask[atlas_data == 4] = 1
        else:
            temp_image = image[atlas_data == 4, subject_index]
            threshold = -np.sort(-temp_image)[Z_TOP]
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

if __name__ == "__main__":
    starttime = datetime.datetime.now()

    img = nib.load(ALL_PROB_MASK)
    mask = img.get_data()
    affine = img.get_affine()

    if not os.path.exists(RW_AGGRAGATOR_RESULT_DATA_DIR):
        os.makedirs(RW_AGGRAGATOR_RESULT_DATA_DIR)

    rw_atlas_based_aggrator_result = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], SESSION_NUMBERS, len(ROI) + 1), dtype=np.float)

    pool = multiprocessing.Pool(processes=SESSION_NUMBERS)
    pool_outputs = pool.map(atlas_based_aggragator, range(SESSION_NUMBERS))
    pool.close()
    pool.join()

    for i in range(SESSION_NUMBERS):
        for roi_index in range(len(ROI) + 1):
            rw_atlas_based_aggrator_result[..., i, roi_index] = pool_outputs[i][roi_index]

    for roi_index in range(len(ROI) + 1):
        if roi_index == len(ROI):
            nib.save(nib.Nifti1Image(rw_atlas_based_aggrator_result[..., roi_index], affine), RW_AGGRAGATOR_RESULT_DATA_DIR + RW_PROB_BACKGROUND_RESULT_FILE)
        else:
            nib.save(nib.Nifti1Image(rw_atlas_based_aggrator_result[..., roi_index], affine), RW_AGGRAGATOR_RESULT_DATA_DIR + ROI[roi_index] + '_' +RW_ATLAS_BASED_AGGRATOR_RESULT_FILE)


    #generate the prob result
    temp_image = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], SESSION_NUMBERS))
    for subject_index in range(SESSION_NUMBERS):
        for roi_index in range(len(ROI) + 1):
            temp_image[rw_atlas_based_aggrator_result[..., subject_index, roi_index] > 0 , subject_index] = 1

        coords = np.array(np.nonzero(temp_image[..., subject_index] == 1), dtype=np.int32).T
        for i in range(coords.shape[0]):
            temp_image[coords[i, 0], coords[i, 1], coords[i, 2], subject_index] = np.array([rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 0],
                                                                              rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 1],
                                                                              rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 2],
                                                                              rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 3],
                                                                              rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 4]]).argmax() + 1
        print 'subject_index: ', subject_index
    temp_image[temp_image == 5] = 0
    nib.save(nib.Nifti1Image(temp_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + RW_PROB_RESULT_FILE)

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."

































