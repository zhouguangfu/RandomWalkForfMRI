__author__ = 'zgf'

import datetime
import os
import numpy as np
import nibabel as nib

from configs import *

SESSION_NUM = 7


image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()


if __name__ == '__main__':
    starttime = datetime.datetime.now()

    # connect_gss_4d_volume()
    # connect_manual_gold_4d_volume()

    mpm_datas = np.zeros_like(image)
    mpm_datas[image > 2.3] = 1
    weight = np.ones(mpm_datas.shape[3], dtype=float)
    mpm_data = np.average(mpm_datas, axis=3, weights=weight)

    nib.save(nib.Nifti1Image(mpm_data, affine), ANALYSIS_DIR + 'face_mpm.nii.gz')

    endtime = datetime.datetime.now()
    print 'Time costs: ', (endtime - starttime)
    print "Program end..."

