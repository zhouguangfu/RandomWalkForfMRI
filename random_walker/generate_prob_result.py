__author__ = 'zgf'
__author__ = 'zhouguangfu'

import datetime
import numpy as np
import nibabel as nib

SUBJECT_NUM = 70
ATLAS_NUM = 202

if __name__ == "__main__":
    starttime = datetime.datetime.now()
    lines = []

    img = nib.load(ALL_PROB_MASK)
    mask = img.get_data()
    affine = img.get_affine()

    rw_atlas_based_aggrator_result = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], SUBJECT_NUM, len(ROI) + 1), dtype=np.float)
    for roi_index in range(len(ROI) + 1):
        if roi_index == len(ROI):
            rw_atlas_based_aggrator_result[..., roi_index] = nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + RW_PROB_BACKGROUND_RESULT_FILE).get_data()
        else:
            rw_atlas_based_aggrator_result[..., roi_index] = nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + ROI[roi_index] + '_' +RW_ATLAS_BASED_AGGRATOR_RESULT_FILE).get_data()

    temp_image = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], SUBJECT_NUM))
    for subject_index in range(SUBJECT_NUM):
        for roi_index in range(len(ROI) + 1):
            temp_image[rw_atlas_based_aggrator_result[..., subject_index, roi_index] > 0 , subject_index] = 1

        coords = np.array(np.nonzero(temp_image[..., subject_index] == 1), dtype=np.int32).T
        for i in range(coords.shape[0]):
            temp_image[coords[i, 0], coords[i, 1], coords[i, 2], subject_index] = np.array([rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 0],
                                                                              rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 1],
                                                                              rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 2],
                                                                              rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 3],
                                                                              rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 4]*0.5]).argmax() + 1
        print 'subject_index: ', subject_index
    nib.save(nib.Nifti1Image(temp_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + RW_PROB_RESULT_FILE)

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
































