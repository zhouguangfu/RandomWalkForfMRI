__author__ = 'zgf'

import datetime
import numpy as np
import nibabel as nib

from skimage.segmentation import slic
from configs import *

DEFAULT_TOP_RANK = 1
SUBJECT_NUM = 1
SUPERVOXEL_SEGMENTATION = 20000

img = nib.load(ALL_PROB_MASK)
mask = img.get_data()
affine = img.get_affine()

test_data = nib.load('G:\workingdir\zstat1.nii.gz').get_data()

left_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_LEFT_BRAIN_FILE).get_data() > 0
right_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_RIGHT_BRAIN_FILE).get_data() > 0

def compute_parcel_peak(subject_index):
    # localmax_cords = local_maximum(image[..., subject_index], 2)
    # nib.save(nib.Nifti1Image(localmax_cords, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
    #                                               '_supervoxel_localmax.nii.gz')
    slice = test_data
    gray_image = (slice - slice.min()) * 255 / (slice.max() - slice.min())
    localmax_cords = []

    slic_image = slic(gray_image.astype(np.float),
                      n_segments=SUPERVOXEL_SEGMENTATION,
                      slic_zero=True,
                      sigma=2,
                      multichannel =False,
                      enforce_connectivity=True)
    supervoxels = np.unique(slic_image)
    for i in supervoxels:
        temp = slice.copy()
        temp[slic_image != i] = -1000
        peak_cord = np.unravel_index(temp.argmax(), slice.shape)
        localmax_cords.append(peak_cord)

    localmax_cords = np.array(localmax_cords)

    return localmax_cords, slic_image
    # return np.array(np.nonzero(localmax_cords)).T

def compute_background_parcel(slic_image):
    from scipy.ndimage.morphology import binary_dilation

    left_brain_dilation = binary_dilation(left_barin_mask.astype(np.int))
    right_brain_dilation = binary_dilation(right_barin_mask.astype(np.int))

    left_brain_dilation[left_barin_mask] = 0
    right_brain_dilation[right_barin_mask] = 0

    left_brain_supervoxels = slic_image[left_brain_dilation > 0]
    right_brain_supervoxels = slic_image[right_brain_dilation > 0]

    left_brain_background_marker = np.zeros_like(left_brain_dilation).astype(np.int)
    right_brain_background_marker = np.zeros_like(right_brain_dilation).astype(np.int)

    for i in np.unique(left_brain_supervoxels):
        left_brain_background_marker[slic_image == i] = 1
    for j in np.unique(right_brain_supervoxels):
        right_brain_background_marker[slic_image == j] = 1

    print len(np.unique(left_brain_supervoxels))

    nib.save(nib.Nifti1Image(left_brain_dilation.astype(np.int), affine),
             RW_AGGRAGATOR_RESULT_DATA_DIR + 'left_bk_marker_dilation.nii.gz')
    nib.save(nib.Nifti1Image(right_brain_dilation.astype(np.int), affine),
             RW_AGGRAGATOR_RESULT_DATA_DIR + 'right_bk_marker_dilation.nii.gz')
    nib.save(nib.Nifti1Image(left_brain_background_marker, affine),
             RW_AGGRAGATOR_RESULT_DATA_DIR + 'left_bk_marker.nii.gz')
    nib.save(nib.Nifti1Image(right_brain_background_marker, affine),
             RW_AGGRAGATOR_RESULT_DATA_DIR + 'right_bk_marker.nii.gz')

    return left_brain_background_marker, right_brain_background_marker



if __name__ == "__main__":
    starttime = datetime.datetime.now()

    localmax_cords, slic_image = compute_parcel_peak(0)
    # slic_image = nib.load( RW_AGGRAGATOR_RESULT_DATA_DIR + 'slic_image.nii.gz').get_data()
    print 'slic_image end...'
    compute_background_parcel(slic_image)

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."































