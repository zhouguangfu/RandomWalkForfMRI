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

    for roi_index in range(len(ROI)):
        for thr in threshold:
            roi_data = nib.load(PROB_ROI_202_SUB_FILE + ROI[roi_index] + '_prob.nii.gz').get_data()
        #     print 'roi_index: ', ROI[roi_index], '   thr: ', thr, '  roi_size:', (roi_data > thr).sum()
        # print '---------------------------------------'

    # optimal threshold
    # r_OFA: 0.35 -- 201
    # l_OFA: 0.2 -- 255
    # r_pFus: 0.35 -- 237
    # l_pFus: 0.25 -- 183
    optimal_threshold = np.array([0.35, 0.2, 0.35, 0.25])

    #generate the label mask
    for roi_index in range(len(ROI)):
        roi_img = nib.load(PROB_ROI_202_SUB_FILE + ROI[roi_index] + '_prob.nii.gz')
        roi_data = roi_img.get_data()
        affine = roi_img.get_affine()
        nib.save(nib.Nifti1Image((roi_data > optimal_threshold[roi_index]).astype(np.int32), affine), LABEL_ROI_202_SUB_FILE + ROI[roi_index] + '_label.nii.gz')
        print 'roi_index: ', ROI[roi_index], '   optimal_threshold: ', optimal_threshold[roi_index], '  roi_size:', (roi_data > optimal_threshold[roi_index]).sum()

    img = nib.load(ALL_PROB_MASK)
    mask = img.get_data()
    affine = img.get_affine()

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
































