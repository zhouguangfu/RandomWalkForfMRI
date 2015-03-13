__author__ = 'zgf'

import numpy as np
import nibabel as nib
import os

from configs import *

#Convert file format:
#crlConvertBetweenFileFormats -in inputfilename -out outputfilename[-opct] outputPixelComponentType
crlConvertBetweenFileFormats = '/nfs/j3/userhome/zhouguangfu/workingdir/crkit_build/install/bin/crlConvertBetweenFileFormats'

#staple command
#crlSTAPLE -o Weights.nrrd 1.nrrd 2.nrrd 3.nrrd 4.nrrd
crlSTAPLE = '/nfs/j3/userhome/zhouguangfu/workingdir/crkit_build/install/bin/crlSTAPLE'

#crlIndexOfMaxComponent command
#crlIndexOfMaxComponent Weights.nrrd Seg.nrrd
crlIndexOfMaxComponent = '/nfs/j3/userhome/zhouguangfu/workingdir/crkit_build/install/bin/crlIndexOfMaxComponent'

#crlOverlapstats3d command
#crlOverlapstats3d 1.nrrd REF.nrrd 1
crlOverlapstats3d = '/nfs/j3/userhome/zhouguangfu/workingdir/crkit_build/install/bin/crlOverlapstats3d'


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

    manual_para = ''
    rw_para = ''
    for j in range(i * SESSION_NUM, (i + 1) * SESSION_NUM):
        #save to *.nii.gz
        # nib.save(nib.Nifti1Image(all_subject_session_manual[..., j], affine), manual_path + str(j) + '.nii.gz')
        # nib.save(nib.Nifti1Image(all_subject_session_rw[..., j], affine), rw_path + str(j) + '.nii.gz')

        #save to *.nii
        nib.save(nib.Nifti1Image(all_subject_session_manual[..., j], affine), manual_path + str(j) + '.nii')
        nib.save(nib.Nifti1Image(all_subject_session_rw[..., j], affine), rw_path + str(j) + '.nii')
        manual_para += ' ' + manual_path + str(j) + '.nii'
        rw_para += ' ' + rw_path + str(j) + '.nii'

    print 'i: ', i, ' manuanl---'
    manual_output = os.popen(crlSTAPLE + ' -o ' + manual_path + 'weights.nii ' + manual_para)
    manuanl_file = open(manual_path + 'log.txt', 'w+')
    manuanl_file.write(manual_output.read())
    manuanl_file.close()
    os.popen(crlIndexOfMaxComponent + ' ' + manual_path + 'weights.nii ' + manual_path + 'maxinum_component.nii')

    manuanl_dice_file = open(manual_path + 'dice.txt', 'w+')
    for index in range(SESSION_NUM):
        manuanl_dice_file.write('#------------'+ str(index) + '------------')
        for roi_index in range(len(ROI)):
            manuanl_dice_output = os.popen(crlOverlapstats3d + ' ' + manual_path + str(index) + '.nii '
                                  + manual_path + 'maxinum_component.nii' + ' ' + str(roi_index + 1))
            manuanl_dice_file.write(manuanl_dice_output.read())
    manuanl_dice_file.close()

    print 'i: ', i, ' rw---'
    rw_output = os.popen(crlSTAPLE + ' -o ' + rw_path + 'weights.nii ' + rw_para)
    rw_file = open(rw_path + 'log.txt', 'w+')
    rw_file.write(rw_output.read())
    rw_file.close()
    os.popen(crlIndexOfMaxComponent + ' ' + rw_path + 'weights.nii ' + rw_path + 'maxinum_component.nii')

    rw_dice_file = open(rw_path + 'dice.txt', 'w+')
    for index in range(SESSION_NUM):
        rw_dice_file.write('#------------'+ str(index) + '------------')
        for roi_index in range(len(ROI)):
            rw_dice_output = os.popen(crlOverlapstats3d + ' ' + rw_path + str(index) + '.nii '
                             + rw_path + 'maxinum_component.nii' + ' ' + str(roi_index + 1))
            rw_dice_file.write(rw_dice_output.read())
    rw_dice_file.close()










