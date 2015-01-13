__author__ = 'zgf'

import numpy as np
import nibabel as nib
from configs import *

image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

DEFAULT_TOP_RANK = 60
SESSION_NUMBERS = 7

def connect_results():
    for roi_index in range(len(ROI)):
        rw_atlas_based_aggrator_result = np.zeros_like(image)
        for i in range(len(SUBJECT_NAMES)):
            for j in range(len(SESSION_NUMBERS)):
                rw_atlas_based_aggrator_result[..., i * SESSION_NUMBERS + j] = \
                    nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + str(i) + '/' + str(DEFAULT_TOP_RANK) + '/' + ROI[roi_index] + '_aggragator.nii.gz')

        nib.save(nib.Nifti1Image(rw_atlas_based_aggrator_result, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + ROI[roi_index] + '_aggragator.nii.gz')