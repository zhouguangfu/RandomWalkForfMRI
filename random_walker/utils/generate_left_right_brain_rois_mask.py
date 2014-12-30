__author__ = 'zgf'

import datetime
import multiprocessing
import numpy as np
import nibabel as nib

from configs import *

SUBJECT_NUM = 70
ATLAS_NUM = 202

threshold = np.arange(0, 1, 0.05)
if __name__ == "__main__":
    starttime = datetime.datetime.now()
    lines = []

    roi_img = nib.load(PROB_ROI_202_SUB_FILE + ROI[0] + '_prob.nii.gz')
    half_brain_mask = np.zeros_like(roi_img.get_data()).astype(np.int32)
    for roi_index in [0, 2, 1, 3]:
        roi_img = nib.load(PROB_ROI_202_SUB_FILE + ROI[roi_index] + '_prob.nii.gz')
        roi_data = roi_img.get_data()
        affine = roi_img.get_affine()
        half_brain_mask[roi_data > 0] = 1

        if roi_index == 2:
            nib.save(nib.Nifti1Image(half_brain_mask, affine), PROB_ROI_202_SUB_FILE + PROB_LEFT_BRAIN_FILE)
            half_brain_mask = np.zeros_like(roi_data)
        if roi_index == (len(ROI) - 1):
            nib.save(nib.Nifti1Image(half_brain_mask, affine), PROB_ROI_202_SUB_FILE + PROB_RIGHT_BRAIN_FILE)

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
































