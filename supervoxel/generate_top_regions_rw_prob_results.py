__author__ = 'zgf'

import datetime
import numpy as np
import nibabel as nib
import os

from configs import *
from skimage.segmentation import random_walker
from skimage.segmentation import slic

DEFAULT_TOP_RANK = 10 # 202
SUBJECT_NUM = 35
SUPERVOXEL_SEGMENTATION = 10000

#global varibale
image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

def atlas_based_aggragator(subject_index):
    # region_results_RW = nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
    #                              '_basic_regions_supervoxel.nii.gz').get_data()
    # region_results_RW = nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
    #                              '_neighbor_regions_supervoxel.nii.gz').get_data()
    region_results_RW = nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                 '_radius_regions_supervoxel.nii.gz').get_data()

    weight = np.ones(DEFAULT_TOP_RANK, dtype=float)
    weighted_result = []

    for roi_index in range(len(ROI) + 1):
        temp = np.zeros_like(region_results_RW[..., 0:DEFAULT_TOP_RANK])
        temp[region_results_RW[..., 0:DEFAULT_TOP_RANK] == roi_index] = 1

        for i in range(temp.shape[3]):
            temp_data = temp[..., i].copy()
            if (temp_data == 1).sum() != 0:
                temp[temp_data == 1, i] = (image[temp_data == 1, subject_index] - image[temp_data == 1, subject_index].min()) / \
                                          (image[temp_data == 1, subject_index].max() - image[temp_data == 1, subject_index].min())

        weighted_result.append(np.average(temp, axis=3, weights=weight))

        if roi_index > 0:
            nib.save(nib.Nifti1Image(weighted_result[roi_index], affine), RW_AGGRAGATOR_RESULT_DATA_DIR +
                                                                          ROI[roi_index-1] + '_' + str(subject_index) + '_aggragator.nii.gz')
        else:
            nib.save(nib.Nifti1Image(weighted_result[roi_index], affine), RW_AGGRAGATOR_RESULT_DATA_DIR +
                                                                          'background_' + str(subject_index) + '_aggragator.nii.gz')
        print 'subject_index: ', subject_index, '   roi_index: ', roi_index

    return weighted_result

def generate_rw_prob_result(rw_atlas_based_aggrator_result, subject_sum):
    #generate the prob result
    temp_image = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], subject_sum))
    for subject_index in range(subject_sum):
        for roi_index in range(len(ROI) + 1):
            temp_image[rw_atlas_based_aggrator_result[..., subject_index, roi_index] > 0 , subject_index] = 1

        coords = np.array(np.nonzero(temp_image[..., subject_index] == 1), dtype=np.int32).T
        for i in range(coords.shape[0]):
            temp_image[coords[i, 0], coords[i, 1], coords[i, 2], subject_index] = \
                np.array([rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 0],
                          rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 1],
                          rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 2],
                          rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 3],
                          rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 4]]).argmax()
        print 'generate subject_index: ', subject_index
        # temp_image[temp_image == 5] = 0
    nib.save(nib.Nifti1Image(temp_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR +
                                                  'top_' + str(DEFAULT_TOP_RANK) + '_'+ RW_PROB_RESULT_FILE)

    return temp_image


if __name__ == "__main__":
    starttime = datetime.datetime.now()

    img = nib.load(ALL_PROB_MASK)
    mask = img.get_data()
    affine = img.get_affine()

    if not os.path.exists(RW_AGGRAGATOR_RESULT_DATA_DIR):
        os.makedirs(RW_AGGRAGATOR_RESULT_DATA_DIR)

    rw_atlas_based_aggrator_result = np.zeros((image.shape[0], image.shape[1], image.shape[2], SUBJECT_NUM, len(ROI) + 1))
    # for subject_index in range(0, 7): #by sessions
    for subject_index in range(0, SUBJECT_NUM): #by sessions
        # select_optimal_parcel_min_distance(subject_index)
        # select_optimal_parcel_max_region_mean(subject_index)
        weighted_result = atlas_based_aggragator(subject_index)
        for i in range(len(ROI) + 1):
            rw_atlas_based_aggrator_result[..., subject_index, i] = weighted_result[i]

    generate_rw_prob_result(rw_atlas_based_aggrator_result, SUBJECT_NUM)

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."































