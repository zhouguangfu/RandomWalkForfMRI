__author__ = 'zhouguangfu'

import datetime
import numpy as np
import nibabel as nib
from skimage.morphology import skeletonize, medial_axis
from configs import *

ALL_202_IMAGES_SKELETONIZE_DIR = ATLAS_TOP_DIR + 'all_202_image_skeletonize.nii.gz'
ALL_SUBJECT_SESSIONS_SKELETONIZE_FOREGROUND_DIR = ATLAS_TOP_DIR + 'all_sessions_skeletonize_foreground.nii.gz'
ALL_SUBJECT_SESSIONS_SKELETONIZE_BACKGROUND_DIR = ATLAS_TOP_DIR + 'all_sessions_skeletonize_background.nii.gz'


if __name__ == "__main__":
    starttime = datetime.datetime.now()
    all_202_image = nib.load(ALL_202_SUBJECTS_DATA_DIR)
    affine = all_202_image.get_affine()
    all_202_image_data = all_202_image.get_data()


    all_202_image_data[all_202_image_data < 0] = 0
    all_202_image_data[all_202_image_data > 0] = 1

    for j in range(all_202_image_data.shape[3]):
        for i in range(all_202_image_data.shape[2]):
            # all_202_image_data[..., i, j] = skeletonize(all_202_image_data[..., i, j])
            all_202_image_data[..., i, j] = medial_axis(all_202_image_data[..., i, j])
            print 'All 202 subjects foreground => j: ', j, '   i: ', i
    nib.save(nib.Nifti1Image(all_202_image_data, affine), ALL_202_IMAGES_SKELETONIZE_DIR)

    all_subject_sessions_foreground_image = nib.load(ACTIVATION_DATA_DIR)
    all_subject_sessions_foreground_data = all_subject_sessions_foreground_image.get_data()

    all_subject_sessions_background_data = np.copy(all_subject_sessions_foreground_data)
    all_subject_sessions_foreground_data[all_subject_sessions_foreground_data < 0] = 0
    all_subject_sessions_foreground_data[all_subject_sessions_foreground_data > 0] = 1

    for j in range(all_subject_sessions_foreground_data.shape[3]):
        for i in range(all_subject_sessions_foreground_data.shape[2]):
            # all_subject_sessions_foreground_data[..., i, j] = skeletonize(all_subject_sessions_foreground_data[..., i, j])
            all_subject_sessions_foreground_data[..., i, j] = medial_axis(all_subject_sessions_foreground_data[..., i, j])
            print 'Foreground => j: ', j, '   i: ', i

    all_subject_sessions_background_data[all_subject_sessions_background_data > 0] = 0
    all_subject_sessions_background_data[all_subject_sessions_background_data < 0] = 1

    for j in range(all_subject_sessions_background_data.shape[3]):
        for i in range(all_subject_sessions_background_data.shape[2]):
            # all_subject_sessions_background_data[..., i, j] = skeletonize(all_subject_sessions_background_data[..., i, j])
            all_subject_sessions_background_data[..., i, j] = medial_axis(all_subject_sessions_background_data[..., i, j])
            print 'Background => j: ', j, '   i: ', i

    nib.save(nib.Nifti1Image(all_subject_sessions_foreground_data, affine), ALL_SUBJECT_SESSIONS_SKELETONIZE_FOREGROUND_DIR)
    nib.save(nib.Nifti1Image(all_subject_sessions_background_data, affine), ALL_SUBJECT_SESSIONS_SKELETONIZE_BACKGROUND_DIR)

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
