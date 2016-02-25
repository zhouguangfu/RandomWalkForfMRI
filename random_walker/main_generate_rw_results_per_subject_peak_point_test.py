__author__ = 'zhouguangfu'

import datetime
import multiprocessing
import os

import nibabel as nib
import numpy as np
from skimage.segmentation import random_walker

from configs import *

SUBJECTS_SESSION_NUMBERS = 70
ATLAS_NUM = 202 #1 - 202

#Global varibale
image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

complete_atlas_data = nib.load(ATLAS_TOP_DIR + 'complete_atlas_label.nii.gz').get_data()
all_202_image_data = nib.load(ALL_202_SUBJECTS_DATA_DIR).get_data()

#Process subject data.
def process_subjects():
    r_FFA_result = np.zeros((complete_atlas_data.shape[0], complete_atlas_data.shape[1], complete_atlas_data.shape[2]))

    for i in range(complete_atlas_data.shape[3]):
        mask = complete_atlas_data[..., i]
        temp = np.zeros_like(r_FFA_result)
        temp[mask == 3] = all_202_image_data[mask == 3, i]
        peak_cord = np.unravel_index(temp.argmax(), temp.shape)

        r_FFA_result[peak_cord[0], peak_cord[1], peak_cord[2]] = i

        print 'subject_index: ', i

    nib.save(nib.Nifti1Image(r_FFA_result, affine), RW_AGGRAGATOR_RESULT_DATA_DIR  +  'r_FFA_peak_test.nii.gz')





if __name__ == "__main__":
    starttime = datetime.datetime.now()

    process_subjects()

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."

































