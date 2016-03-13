__author__ = 'zgf'

import numpy as np
import nibabel as nib
from configs import *
import datetime
import os
import multiprocessing

CV_FEAT_ACTIVATION_DATA_DIR = ANALYSIS_DIR + 'gold/' + 'CV_Feat_gold.nii.gz'
image = nib.load(CV_FEAT_ACTIVATION_DATA_DIR)
# image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

ATLAS_NUM = 202
SESSION_NUMBERS = 7
SUBJECTS_SESSION_NUMBERS = 70

LEFT_RIGHT_BRAIN_NAME = ['left_brain', 'right_brain']
left_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_LEFT_BRAIN_FILE).get_data()
right_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_RIGHT_BRAIN_FILE).get_data()

#Aggragator the result
def atlas_based_aggragator(subject_index):
    global ATLAS_NUM
    single_subject_rw_regions = nib.load(RW_RESULT_DATA_DIR + 'aggragator/' + 'subjects_rw_all_atlas_results/' +
                                            str(subject_index) + '_regions_rw.nii.gz').get_data()

    region_result_RW = np.zeros((image.shape[0], image.shape[1], image.shape[2], ATLAS_NUM))

    #Top atlas
    for atlas_index in range(ATLAS_NUM):
        np.random.seed()
        left_brain_indexs = np.random.choice(range(202), ATLAS_NUM, replace=False)
        np.random.seed()
        right_brain_indexs = np.random.choice(range(202), ATLAS_NUM, replace=False)

        region_result_RW[np.logical_and(right_barin_mask, single_subject_rw_regions[..., right_brain_indexs[atlas_index]] == 5)
                         , atlas_index] = 5
        region_result_RW[np.logical_and(left_barin_mask, single_subject_rw_regions[..., left_brain_indexs[atlas_index]] == 5)
                         , atlas_index] = 5

        region_result_RW[single_subject_rw_regions[..., right_brain_indexs[atlas_index]] == 1, atlas_index] = 1
        region_result_RW[single_subject_rw_regions[..., left_brain_indexs[atlas_index]] == 2, atlas_index] = 2
        region_result_RW[single_subject_rw_regions[..., right_brain_indexs[atlas_index]] == 3, atlas_index] = 3
        region_result_RW[single_subject_rw_regions[..., left_brain_indexs[atlas_index]] == 4, atlas_index] = 4

    for i in range(20):
        atlas_num = (i + 1) * 10
        rw_atlas_based_aggrator_result = np.zeros((image.shape[0], image.shape[1], image.shape[2], len(ROI) + 1))
        weight = np.ones(atlas_num, dtype=float)
        weight = weight * 1. / atlas_num

        for roi_index in range(len(ROI) + 1):
            temp = np.zeros_like(region_result_RW[..., 0 : atlas_num])
            temp[region_result_RW[..., 0 : atlas_num] == (roi_index + 1)] = 1
            rw_atlas_based_aggrator_result[..., roi_index] = np.average(temp, axis=3, weights=weight)


        #generate the prob result
        temp_image = np.zeros_like(image[..., 0])
        for roi_index in range(len(ROI) + 1):
            temp_image[rw_atlas_based_aggrator_result[..., roi_index] > 0] = 1

        coords = np.array(np.nonzero(temp_image == 1), dtype=np.int32).T
        for i in range(coords.shape[0]):
            temp_image[coords[i, 0], coords[i, 1], coords[i, 2]] = \
                    np.array([rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], 0],
                            rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], 1],
                            rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], 2],
                            rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], 3],
                            rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], 4]]).argmax() + 1

        temp_image[temp_image == 5] = 0
        # nib.save(nib.Nifti1Image(temp_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + 'rw/mv_' + RW_PROB_RESULT_FILE)

        single_subject_rw_result_dir = RW_AGGRAGATOR_RESULT_DATA_DIR + str(atlas_num) + '/'
        if not os.path.exists(single_subject_rw_result_dir):
            os.makedirs(single_subject_rw_result_dir)

        nib.save(nib.Nifti1Image(temp_image, affine), single_subject_rw_result_dir + str(subject_index) + '_' + RW_PROB_RESULT_FILE)

        print 'subject_index: ', subject_index, ' atlas_num: ', atlas_num
    print 'subject_index: ', subject_index, ' end............................'


def generate_rw_prob_result(atlas_num):
    atlas_num = (atlas_num + 1) * 10
    sibgle_subject_result = np.zeros_like(image)
    for subject_index in range(image.shape[3]):
        single_subject_dir = RW_AGGRAGATOR_RESULT_DATA_DIR + str(atlas_num) + '/'
        sibgle_subject_result[..., subject_index] = nib.load(single_subject_dir + str(subject_index) + '_' + RW_PROB_RESULT_FILE).get_data()
    nib.save(nib.Nifti1Image(sibgle_subject_result, affine),   RW_AGGRAGATOR_RESULT_DATA_DIR + '/top_rank_' + str(atlas_num)
                 + '_' + RW_PROB_RESULT_FILE)
    print '-------------------- ', atlas_num, ' ----------------------- '


if __name__ == "__main__":
    starttime = datetime.datetime.now()
    RW_AGGRAGATOR_RESULT_DATA_DIR = RW_AGGRAGATOR_RESULT_DATA_DIR + 'rw/random/'
    if not os.path.exists(RW_AGGRAGATOR_RESULT_DATA_DIR):
        os.makedirs(RW_AGGRAGATOR_RESULT_DATA_DIR)

    # for subject_index in range(image.shape[3]):
    #     atlas_based_aggragator(subject_index)

    # atlas_based_aggragator(0)
    old_dir = RW_AGGRAGATOR_RESULT_DATA_DIR
    for random_index in range(1, 10):
        RW_AGGRAGATOR_RESULT_DATA_DIR = old_dir + str(random_index) + '/'

        #For multi process
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

        endtime = datetime.datetime.now()
        print 'Time cost: ', (endtime - starttime)

        starttime = datetime.datetime.now()
        process_num = 4
        for cycle_index in range(20 / process_num):
            pool = multiprocessing.Pool(processes=process_num)
            pool_outputs = pool.map(generate_rw_prob_result, range(cycle_index * process_num,
                                                              (cycle_index + 1) * process_num))
            pool.close()
            pool.join()

            print 'Cycle index: ', cycle_index, 'Time cost: ', (datetime.datetime.now() - starttime)
            starttime = datetime.datetime.now()

        endtime = datetime.datetime.now()
        print 'Time cost: ', (endtime - starttime)


    print "Program end..."