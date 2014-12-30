__author__ = 'zhouguangfu'

import datetime
import multiprocessing

import numpy as np
import nibabel as nib
from skimage.segmentation import random_walker
from scipy.ndimage import morphology

SUBJECT_NUM = 70
ATLAS_NUM = 202

def rw_atlas_based_prob_process(subject_list):
    affine, roi_index = subject_list
    rw_prob_roi = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], SUBJECT_NUM), dtype=np.int32)
    rw_atlas_based_aggrator_result = nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + ROI[roi_index] + '_' +RW_ATLAS_BASED_AGGRATOR_RESULT_FILE).get_data()
    rw_prob_roi[rw_atlas_based_aggrator_result == 0] = -1 #not consider
    for j in range(SUBJECT_NUM):
    # for j in range(3):
        volume = morphology.binary_dilation(rw_atlas_based_aggrator_result[..., j] > 0).astype(rw_atlas_based_aggrator_result.dtype)
        rw_prob_roi[np.logical_and(rw_atlas_based_aggrator_result[..., j] <= 0.1 * rw_atlas_based_aggrator_result[..., j].max() , volume > 0), j] = 1
        rw_prob_roi[rw_atlas_based_aggrator_result[..., j] >= 0.5 * rw_atlas_based_aggrator_result[..., j].max(), j] = 2
        nib.save(nib.Nifti1Image(rw_prob_roi.astype(np.int32), affine), RW_AGGRAGATOR_RESULT_DATA_DIR + ROI[roi_index] + '_' +RW_PROB_RESULT_FILE)
        label = random_walker(rw_atlas_based_aggrator_result[..., j], rw_prob_roi[..., j], beta=10, mode='bf')
        rw_prob_roi[label == 2, j] = 2

        print 'j: ', j, '   roi_index:', roi_index
    nib.save(nib.Nifti1Image((rw_prob_roi == 2).astype(np.int32), affine), RW_AGGRAGATOR_RESULT_DATA_DIR + ROI[roi_index] + '_' +RW_PROB_RESULT_FILE)

if __name__ == "__main__":
    starttime = datetime.datetime.now()
    lines = []

    img = nib.load(ALL_PROB_MASK)
    mask = img.get_data()
    affine = img.get_affine()

    image_list = []
    for i in range(len(ROI)):
        image_list.append((affine, i))

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool_outputs = pool.map(rw_atlas_based_prob_process, image_list)
    pool.close()
    pool.join()

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
































