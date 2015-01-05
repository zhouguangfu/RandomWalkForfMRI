__author__ = 'zgf'

import datetime
import multiprocessing
import numpy as np
import nibabel as nib

from configs import *

SUBJECT_NUM = 10
ATLAS_NUM = 202
BACKGROUND_THR = 0
TOP_RANK = 180 # 0 - 100

def atlas_based_aggragator(subject_index):
    region_result_RW = nib.load(RW_RESULT_DATA_DIR + str(subject_index)+ '_'+ str(TOP_RANK) + '_' + RW_ATLAS_BASED_RESULT_FILE).get_data()
    weight = np.ones(ATLAS_NUM, dtype=float)
    weighted_result = []
    for roi_index in range(len(ROI) + 1):
        temp = np.zeros_like(region_result_RW)
        temp[region_result_RW == (roi_index + 1)] = 1
        weighted_result.append(np.average(temp, axis=3, weights=weight))
        print 'j: ', subject_index, '   roi_index: ', roi_index

    return weighted_result

if __name__ == "__main__":
    starttime = datetime.datetime.now()
    lines = []

    img = nib.load(ALL_PROB_MASK)
    mask = img.get_data()
    affine = img.get_affine()

    rw_atlas_based_aggrator_result = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], SUBJECT_NUM, len(ROI) + 1), dtype=np.float)

    pool = multiprocessing.Pool(processes=SUBJECT_NUM)
    pool_outputs = pool.map(atlas_based_aggragator, range(SUBJECT_NUM))
    pool.close()
    pool.join()

    for i in range(SUBJECT_NUM):
        for roi_index in range(len(ROI) + 1):
            rw_atlas_based_aggrator_result[..., i, roi_index] = pool_outputs[i][roi_index]

    for roi_index in range(len(ROI) + 1):
        if roi_index == len(ROI):
            nib.save(nib.Nifti1Image(rw_atlas_based_aggrator_result[..., roi_index], affine),
                     RW_AGGRAGATOR_RESULT_DATA_DIR + str(TOP_RANK) + '_' + RW_PROB_BACKGROUND_RESULT_FILE)
        else:
            nib.save(nib.Nifti1Image(rw_atlas_based_aggrator_result[..., roi_index], affine),
                     RW_AGGRAGATOR_RESULT_DATA_DIR + ROI[roi_index] + '_' + str(TOP_RANK) + '_' +RW_ATLAS_BASED_AGGRATOR_RESULT_FILE)

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
































