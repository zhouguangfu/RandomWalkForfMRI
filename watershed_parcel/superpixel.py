__author__ = 'zgf'

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from configs import *

from skimage.data import lena
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float

r_OFA_mask = nib.load(PROB_ROI_202_SUB_FILE + 'r_OFA_prob.nii.gz').get_data() > 0
l_OFA_mask = nib.load(PROB_ROI_202_SUB_FILE + 'l_OFA_prob.nii.gz').get_data() > 0
r_pFus_mask = nib.load(PROB_ROI_202_SUB_FILE + 'r_pFus_prob.nii.gz').get_data() > 0
l_pFus_mask = nib.load(PROB_ROI_202_SUB_FILE + 'l_pFus_prob.nii.gz').get_data() > 0



# img = img_as_float(lena()[::2, ::2])
# segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
# segments_slic = slic(img, n_segments=250, compactness=10, sigma=1)
# segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)

# print  "Felzenszwalb's number of segments:", len(np.unique(segments_fz))
# print "Slic number of segments: ",  len(np.unique(segments_slic))
# print "Quickshift number of segments: %d",  len(np.unique(segments_quick))
#
# fig, ax = plt.subplots(1, 3)
# fig.set_size_inches(8, 3, forward=True)
# fig.subplots_adjust(0.05, 0.05, 0.95, 0.95, 0.05, 0.05)
#
# ax[0].imshow(mark_boundaries(img, segments_fz))
# ax[0].set_title("Felzenszwalbs's method")
# ax[1].imshow(mark_boundaries(img, segments_slic))
# ax[1].set_title("SLIC")
# ax[2].imshow(mark_boundaries(img, segments_quick))
# ax[2].set_title("Quickshift")
# for a in ax:
#     a.set_xticks(())
#     a.set_yticks(())
# plt.show()

def compute_parcel_peak(subject_index):
    # localmax_cords = local_maximum(image[..., subject_index], 2)
    # nib.save(nib.Nifti1Image(localmax_cords, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
    #                                               '_watershed_localmax.nii.gz')
    frame = image[..., subject_index]
    gray_image = (frame - frame.min()) * 255 / (frame.max() - frame.min())
    localmax_cords = []

    slic_image = slic(gray_image,
                      n_segments=5000,
                      compactness=10,
                      sigma=0.2,
                      multichannel =False,
                      enforce_connectivity=True)
    nib.save(nib.Nifti1Image(slic_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(0) + '_' + str(supervoxel)+ '_test_supervoxel.nii.gz')
    supervoxels = np.unique(slic_image)
    for i in supervoxels:
        temp = frame.copy()
        temp[slic_image != i] = -1000
        peak_cord = np.unravel_index(temp.argmax(), frame.shape)
        localmax_cords.append(peak_cord)

    localmax_cords = np.array(localmax_cords)
    print 'localmax_cords.shape: ', localmax_cords.shape, localmax_cords[100]

    return np.array(np.nonzero(localmax_cords)).T

import nibabel as nib
import datetime
from configs import *

starttime = datetime.datetime.now()

image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

subject_num = 70
supervoxel = 5000
supervoxel_image = np.zeros((image.shape[0], image.shape[1], image.shape[2], subject_num))

# compute_parcel_peak(0)

whole_rois_mask = np.zeros_like(supervoxel_image[..., 0])
whole_rois_mask[r_pFus_mask > 0] = 1
whole_rois_mask[r_OFA_mask > 0] = 1
whole_rois_mask[l_pFus_mask > 0] = 1
whole_rois_mask[l_OFA_mask > 0] = 1

for i in range(subject_num):
    frame = image[..., i]
    gray_image = (frame - frame.min()) * 255 / (frame.max() - frame.min())

    supervoxel_image[..., i] = slic(gray_image,
                                    n_segments=supervoxel,
                                    compactness=10,
                                    sigma=2,
                                    multichannel =False,
                                    enforce_connectivity=True)

    # supervoxel_image[..., i] = slic(gray_image,
    #                                 n_segments=supervoxel,
    #                                 slic_zero=True,
    #                                 sigma=1,
    #                                 multichannel =False,
    #                                 enforce_connectivity=True)

    print "Subject_index: ",i,  "   supervoxel: ", supervoxel
    print "Slic number of segments: ",  len(np.unique(supervoxel_image[..., i]))
supervoxel_image[whole_rois_mask != 1, :] = 0
nib.save(nib.Nifti1Image(supervoxel_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(supervoxel)+ '_supervoxel.nii.gz')



endtime = datetime.datetime.now()
print 'Time cost: ', (endtime - starttime)
print "Program end..."
