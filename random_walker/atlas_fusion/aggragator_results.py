__author__ = 'zgf'

import numpy as np
import nibabel as nib
from configs import *
import datetime
import os

image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

DEFAULT_TOP_RANK = 202
SESSION_NUMBERS = 7

#Aggragator the result
def atlas_based_aggragator(subject_index):
    region_result_RW = np.zeros((image.shape[0], image.shape[1], image.shape[2], DEFAULT_TOP_RANK))
    single_subject_rw_regions = nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                         '_regions_rw.nii.gz').get_data()

    # r_OFA_indexs =  np.load(ATLAS_TOP_DIR + ROI[0] + '_' + str(subject_index) + '_top_sort.npy')
    # l_OFA_indexs =  np.load(ATLAS_TOP_DIR + ROI[1] + '_' + str(subject_index) + '_top_sort.npy')
    # r_pFus_indexs =  np.load(ATLAS_TOP_DIR + ROI[2] + '_' + str(subject_index) + '_top_sort.npy')
    # l_pFus_indexs =  np.load(ATLAS_TOP_DIR + ROI[3] + '_' + str(subject_index) + '_top_sort.npy')
    #
    # #Top atlas
    # for atlas_index in range(DEFAULT_TOP_RANK):
    #     region_result_RW[single_subject_rw_regions[..., r_OFA_indexs[atlas_index]] == 1] = 1
    #     region_result_RW[single_subject_rw_regions[..., l_OFA_indexs[atlas_index]] == 2] = 2
    #     region_result_RW[single_subject_rw_regions[..., r_pFus_indexs[atlas_index]] == 3] = 3
    #     region_result_RW[single_subject_rw_regions[..., l_pFus_indexs[atlas_index]] == 4] = 4

    #Use all atlases.
    region_result_RW = single_subject_rw_regions

    weight = np.ones(DEFAULT_TOP_RANK, dtype=float)

    for roi_index in range(len(ROI) + 1):
        temp = np.zeros_like(region_result_RW)
        temp[region_result_RW == (roi_index + 1)] = 1

        for i in range(temp.shape[3]):
            if roi_index == len(ROI):
                temp_data = temp[..., i].copy()
                temp[temp_data == 1, i] = (-image[temp_data == 1, subject_index] + np.abs(-image[temp_data == 1, subject_index].min())) /\
                            (np.abs(-image[temp_data == 1, subject_index].min()) + np.abs(-image[temp_data == 1, subject_index].max()))
            else:
                temp_data = temp[..., i].copy()
                temp[temp_data == 1, i] = (image[temp_data == 1, subject_index] + np.abs(image[temp_data == 1, subject_index].min())) /\
                            (np.abs(image[temp_data == 1, subject_index].min()) + np.abs(image[temp_data == 1, subject_index].max()))

        weighted_result = np.average(temp, axis=3, weights=weight)

        if roi_index < len(ROI):
            nib.save(nib.Nifti1Image(weighted_result, affine), RW_AGGRAGATOR_RESULT_DATA_DIR +
                                ROI[roi_index] + '_' + str(subject_index) + '_aggragator.nii.gz')
        else:
            nib.save(nib.Nifti1Image(weighted_result, affine), RW_AGGRAGATOR_RESULT_DATA_DIR +
                                'background_' + str(subject_index) + '_aggragator.nii.gz')
        print 'subject_index: ', subject_index, '   roi_index: ', roi_index

def connect_results():
    rw_atlas_based_aggrator_results = np.zeros((image.shape[0], image.shape[1], image.shape[2], image.shape[3], len(ROI) + 1))
    for roi_index in range(len(ROI) + 1):
        rw_atlas_based_aggrator_result = np.zeros_like(image)
        if roi_index != len(ROI):
            for i in range(image.shape[3]):
                rw_atlas_based_aggrator_result[..., i] = \
                    nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + str(DEFAULT_TOP_RANK) + '/' +
                             ROI[roi_index] + '_' + str(i)+ '_aggragator.nii.gz').get_data()

            nib.save(nib.Nifti1Image(rw_atlas_based_aggrator_result, affine),
                     RW_AGGRAGATOR_RESULT_DATA_DIR + 'rw/' + ROI[roi_index] + '_aggragator.nii.gz')
        else:
            for i in range(image.shape[3]):
                rw_atlas_based_aggrator_result[..., i] = \
                    nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + str(DEFAULT_TOP_RANK) + '/background_' +
                             str(i)+ '_aggragator.nii.gz').get_data()

            nib.save(nib.Nifti1Image(rw_atlas_based_aggrator_result, affine),
                     RW_AGGRAGATOR_RESULT_DATA_DIR + 'rw/' + 'background_aggragator.nii.gz')

        rw_atlas_based_aggrator_results[..., roi_index] = rw_atlas_based_aggrator_result
        print 'connect_results: roi_index: ', roi_index

    return rw_atlas_based_aggrator_results


def generate_rw_prob_result(rw_atlas_based_aggrator_result):
    #generate the prob result
    temp_image = np.zeros_like(image)
    for subject_index in range(image.shape[3]):
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
    nib.save(nib.Nifti1Image(temp_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + 'rw/' + RW_PROB_RESULT_FILE)




if __name__ == "__main__":
    starttime = datetime.datetime.now()

    if not os.path.exists(RW_AGGRAGATOR_RESULT_DATA_DIR + 'rw/'):
        os.makedirs(RW_AGGRAGATOR_RESULT_DATA_DIR + 'rw/')

    # for subject_index in range(image.shape[3]):
    #     atlas_based_aggragator(subject_index)

    rw_atlas_based_aggrator_result = connect_results()
    generate_rw_prob_result(rw_atlas_based_aggrator_result)

    endtime = datetime.datetime.now()

    print 'Time cost: ', (endtime - starttime)
    print "Program end..."