__author__ = 'zgf'

'''
Generate single subject vlome.
'''

import numpy as np
import nibabel as nib
import os

from configs import *

SESSION_NUM = 7

image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

all_subject_session_manual_path = ANALYSIS_DIR + 'manual/' + 'manual.nii.gz'
all_subject_session_manual = nib.load(all_subject_session_manual_path).get_data()

all_subject_session_rw = nib.load(ANALYSIS_DIR + 'rw/rw_prob_result_file.nii.gz').get_data()

for i in range(len(SUBJECT_NAMES)):
    manual_path = ANALYSIS_DIR + 'staple/' + str(i) + '_' + SUBJECT_NAMES[i] + '/manual/'
    if not os.path.exists(manual_path):
        os.makedirs(manual_path)

    rw_path = ANALYSIS_DIR + 'staple/' + str(i) + '_' + SUBJECT_NAMES[i] + '/rw/'
    if not os.path.exists(rw_path):
        os.makedirs(rw_path)

    for j in range(i * SESSION_NUM, (i + 1) * SESSION_NUM):
        #save to *.nii.gz
        # nib.save(nib.Nifti1Image(all_subject_session_manual[..., j], affine), manual_path + str(j) + '.nii.gz')
        # nib.save(nib.Nifti1Image(all_subject_session_rw[..., j], affine), rw_path + str(j) + '.nii.gz')

        #save to *.nii
        nib.save(nib.Nifti1Image(all_subject_session_manual[..., j], affine), manual_path + str(j) + '.nii')
        nib.save(nib.Nifti1Image(all_subject_session_rw[..., j], affine), rw_path + str(j) + '.nii')
        print 'Process: ', j