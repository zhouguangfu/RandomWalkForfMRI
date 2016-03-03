__author__ = 'zgf'

import numpy as np
import nibabel as nib
from configs import *
import datetime
import os
import multiprocessing

image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

ATLAS_NUM = 10
SESSION_NUMBERS = 7
SUBJECTS_SESSION_NUMBERS = 70

LEFT_RIGHT_BRAIN_NAME = ['left_brain', 'right_brain']

#Aggragator the result
def atlas_based_aggragator(subject_index):
    region_result_RW = np.zeros((image.shape[0], image.shape[1], image.shape[2], ATLAS_NUM))
    single_subject_rw_regions = nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + 'subjects_rw_all_atlas_results/' +
                                         str(subject_index) + '_regions_rw.nii.gz').get_data()


    left_brain_indexs = np.load(ATLAS_TOP_DIR + LEFT_RIGHT_BRAIN_NAME[0] + '_' + str(subject_index) + '_top_sort.npy')
    right_brain_indexs = np.load(ATLAS_TOP_DIR + LEFT_RIGHT_BRAIN_NAME[1] + '_' + str(subject_index) + '_top_sort.npy')

    #Top atlas
    for atlas_index in range(ATLAS_NUM):
        region_result_RW[single_subject_rw_regions[..., right_brain_indexs[atlas_index]] == 1, atlas_index] = 1
        region_result_RW[single_subject_rw_regions[..., left_brain_indexs[atlas_index]] == 2, atlas_index] = 2
        region_result_RW[single_subject_rw_regions[..., right_brain_indexs[atlas_index]] == 3, atlas_index] = 3
        region_result_RW[single_subject_rw_regions[..., left_brain_indexs[atlas_index]] == 4, atlas_index] = 4

        region_result_RW[single_subject_rw_regions[..., right_brain_indexs[atlas_index]] == 5, atlas_index] = 5
        region_result_RW[single_subject_rw_regions[..., left_brain_indexs[atlas_index]] == 5, atlas_index] = 5

    weight = np.ones(ATLAS_NUM, dtype=float)
    weight = weight * 1. / ATLAS_NUM

    for roi_index in range(len(ROI) + 1):
        temp = np.zeros_like(region_result_RW)
        temp[region_result_RW == (roi_index + 1)] = 1

        weighted_result = np.average(temp, axis=3, weights=weight)

        if roi_index < len(ROI):
            nib.save(nib.Nifti1Image(weighted_result, affine), RW_AGGRAGATOR_RESULT_DATA_DIR +
                                ROI[roi_index] + '_' + str(subject_index) + '_non_weight_mv.nii.gz')
        else:
            nib.save(nib.Nifti1Image(weighted_result, affine), RW_AGGRAGATOR_RESULT_DATA_DIR +
                                'background_' + str(subject_index) + '_non_weight_mv.nii.gz')
        print 'subject_index: ', subject_index, '   roi_index: ', roi_index

def connect_results():
    rw_atlas_based_aggrator_results = np.zeros((image.shape[0], image.shape[1], image.shape[2], image.shape[3], len(ROI) + 1))
    for roi_index in range(len(ROI) + 1):
        rw_atlas_based_aggrator_result = np.zeros_like(image)
        if roi_index != len(ROI):
            for i in range(image.shape[3]):
                rw_atlas_based_aggrator_result[..., i] = \
                    nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + ROI[roi_index] + '_' + str(i)+
                             '_non_weight_mv.nii.gz').get_data()

            nib.save(nib.Nifti1Image(rw_atlas_based_aggrator_result, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + 'rw/' +
                     ROI[roi_index] + '_non_weight_mv.nii.gz')
        else:
            for i in range(image.shape[3]):
                rw_atlas_based_aggrator_result[..., i] = \
                    nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + '/background_' +
                             str(i)+ '_non_weight_mv.nii.gz').get_data()

            nib.save(nib.Nifti1Image(rw_atlas_based_aggrator_result, affine),
                     RW_AGGRAGATOR_RESULT_DATA_DIR + 'rw/' + 'background_non_weight_mv.nii.gz')

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
    # nib.save(nib.Nifti1Image(temp_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + 'rw/mv_' + RW_PROB_RESULT_FILE)
    nib.save(nib.Nifti1Image(temp_image, affine),   RW_AGGRAGATOR_RESULT_DATA_DIR + 'staple/rw/top_rank_' + str(ATLAS_NUM)
             + '_' + RW_PROB_RESULT_FILE)



if __name__ == "__main__":
    starttime = datetime.datetime.now()

    if not os.path.exists(RW_AGGRAGATOR_RESULT_DATA_DIR + 'rw/'):
        os.makedirs(RW_AGGRAGATOR_RESULT_DATA_DIR + 'rw/')

    # for subject_index in range(image.shape[3]):
    #     atlas_based_aggragator(subject_index)

    #For multi process
    for i in range(8):
        ATLAS_NUM = (i + 1) * 10
        print '------------------------------- ', ATLAS_NUM, ' ------------------------------------'

        starttime = datetime.datetime.now()
        process_num = 14
        for cycle_index in range(SUBJECTS_SESSION_NUMBERS / process_num):
            pool = multiprocessing.Pool(processes=process_num)
            pool_outputs = pool.map(atlas_based_aggragator, range(cycle_index * process_num,
                                                              (cycle_index + 1) * process_num))
            pool.close()
            pool.join()

            print 'Cycle index: ', cycle_index, 'Time cost: ', (datetime.datetime.now() - starttime)
            starttime = datetime.datetime.now()

        rw_atlas_based_aggrator_result = connect_results()
        generate_rw_prob_result(rw_atlas_based_aggrator_result)

    endtime = datetime.datetime.now()

    print 'Time cost: ', (endtime - starttime)
    print "Program end..."