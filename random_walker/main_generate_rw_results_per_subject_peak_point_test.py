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
    #Compute ROI mean x, y, z cordinates
    for roi_index in range(len(ROI)):
        result = np.zeros((complete_atlas_data.shape[0], complete_atlas_data.shape[1], complete_atlas_data.shape[2]))
        for i in range(complete_atlas_data.shape[3]):
            mask = complete_atlas_data[..., i]
            temp = np.zeros_like(result)
            temp[mask == (roi_index + 1)] = all_202_image_data[mask == (roi_index + 1), i]
            peak_cord = np.unravel_index(temp.argmax(), temp.shape)

            result[peak_cord[0], peak_cord[1], peak_cord[2]] = 1

        cords = np.asarray(np.nonzero(result))
        x_mean, ymean, z_mean = cords[0].mean(), cords[1].mean(), cords[2].mean()
        print ROI[roi_index], '=> x_mean, ymean, z_mean : ', x_mean, ymean, z_mean

# r_OFA => x_mean, ymean, z_mean :  23.5056818182 23.8977272727 29.4488636364
# l_OFA => x_mean, ymean, z_mean :  64.9556962025 22.6708860759 29.5
# r_pFus => x_mean, ymean, z_mean :  23.8323353293 36.5988023952 25.874251497
# l_pFus => x_mean, ymean, z_mean :  64.7070063694 35.9044585987 25.8662420382

    # nib.save(nib.Nifti1Image(r_FFA_result, affine), RW_AGGRAGATOR_RESULT_DATA_DIR  +  'r_OFA_peak_test.nii.gz')



if __name__ == "__main__":
    starttime = datetime.datetime.now()

    process_subjects()

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."

































