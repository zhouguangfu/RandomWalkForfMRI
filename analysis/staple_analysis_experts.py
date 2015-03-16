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

for i in range(len(SUBJECT_NAMES)):
    experts_path = ANALYSIS_DIR + 'staple/experts/' + str(i) + '_' + SUBJECT_NAMES[i] + '/'
    if not os.path.exists(experts_path):
        os.makedirs(experts_path)

    zgf_path = ANALYSIS_DIR + 'staple/zgf/' + str(i) + '_' + SUBJECT_NAMES[i] + '/manual/'
    lzg_path = ANALYSIS_DIR + 'staple/lzg/' + str(i) + '_' + SUBJECT_NAMES[i] + '/manual/'
    if not os.path.exists(zgf_path):
        raise ValueError(zgf_path + ' is not exsit!')
    if not os.path.exists(lzg_path):
        raise ValueError(lzg_path + ' is not exsit!')

    experts_para = ''
    for j in range(i * SESSION_NUM, (i + 1) * SESSION_NUM):
        experts_para += ' ' + zgf_path + str(j) + '.nii'
        experts_para += ' ' + lzg_path + str(j) + '.nii'
    print experts_para

    print 'i: ', i, ' experts---'
    experts_output = os.popen(crlSTAPLE + ' -o ' + experts_path + 'weights.nii ' + experts_para)
    experts_file = open(experts_path + 'log.txt', 'w+')
    experts_file.write(experts_output.read())
    experts_file.close()
    os.popen(crlIndexOfMaxComponent + ' ' + experts_path + 'weights.nii ' + experts_path + 'maxinum_component.nii')

    experts_dice_file = open(experts_path + 'dice.txt', 'w+')

    for index in range(SESSION_NUM):
        for roi_index in range(len(ROI)):
            experts_dice_output = os.popen(crlOverlapstats3d + ' ' + zgf_path + str(i * SESSION_NUM + index) + '.nii '
                                           + experts_path + 'maxinum_component.nii' + ' ' + str(roi_index + 1))
            experts_dice_file.write(experts_dice_output.read())
    for index in range(SESSION_NUM):
        for roi_index in range(len(ROI)):
            experts_dice_output = os.popen(crlOverlapstats3d + ' ' + lzg_path + str(i * SESSION_NUM + index) + '.nii '
                                           + experts_path + 'maxinum_component.nii' + ' ' + str(roi_index + 1))
            experts_dice_file.write(experts_dice_output.read())
    experts_dice_file.close()












