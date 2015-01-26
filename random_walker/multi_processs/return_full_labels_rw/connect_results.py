__author__ = 'zgf'

import numpy as np
import nibabel as nib
from configs import *
import datetime
import os

image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

DEFAULT_TOP_RANK = 50
SESSION_NUMBERS = 7

# def connect_results():
#     for roi_index in range(len(ROI)):
#         rw_atlas_based_aggrator_result = np.zeros_like(image)
#         for i in range(len(SUBJECT_NAMES)):
#             for j in range(len(SESSION_NUMBERS)):
#                 rw_atlas_based_aggrator_result[..., i * SESSION_NUMBERS + j] = \
#                     nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + str(DEFAULT_TOP_RANK) + '/' + ROI[roi_index] + '_' + str(i)+ '_aggragator.nii.gz')
#
#         nib.save(nib.Nifti1Image(rw_atlas_based_aggrator_result, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + ROI[roi_index] + '_aggragator.nii.gz')


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
    endtime = datetime.datetime.now()

    if not os.path.exists(RW_AGGRAGATOR_RESULT_DATA_DIR + 'rw/'):
        os.makedirs(RW_AGGRAGATOR_RESULT_DATA_DIR + 'rw/')

    rw_atlas_based_aggrator_result = connect_results()
    generate_rw_prob_result(rw_atlas_based_aggrator_result)

    print 'Time cost: ', (endtime - starttime)
    print "Program end..."