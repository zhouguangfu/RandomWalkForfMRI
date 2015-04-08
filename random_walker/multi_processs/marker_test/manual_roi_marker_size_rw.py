__author__ = 'zhouguangfu'

import datetime
import numpy as np
import nibabel as nib
import os

from configs import *
from skimage.segmentation import random_walker
from random_walker.utils.neighbor import *
from rg_algorithm.regiongrowing import *

SUBJECT_SESSION_INDEX = 0 #0, 1, 2, 3, ,4 ,5, 6, 7, 8, 9
SESSION_NUMBERS = 7

DEFAULT_TOP_RANK = 10 # 0 - 100, default
DEFAULT_Z_TOP = 60 #default 60
DEFAULT_BACKGROUND_THR = -1 #default -1

#global varibale
image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

complete_atlas_data = nib.load(ATLAS_TOP_DIR + 'complete_atlas_label.nii.gz').get_data()
left_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_LEFT_BRAIN_FILE).get_data()
right_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_RIGHT_BRAIN_FILE).get_data()

thin_background_image = nib.load(ATLAS_TOP_DIR + 'all_sessions_background_skeletonize.nii.gz').get_data()
thin_foreground_image = nib.load(ATLAS_TOP_DIR + 'all_sessions_skeletonize.nii.gz').get_data()

r_OFA_mask = nib.load(PROB_ROI_202_SUB_FILE + 'r_OFA_prob.nii.gz').get_data() > 0
l_OFA_mask = nib.load(PROB_ROI_202_SUB_FILE + 'l_OFA_prob.nii.gz').get_data() > 0
r_pFus_mask = nib.load(PROB_ROI_202_SUB_FILE + 'r_pFus_prob.nii.gz').get_data() > 0
l_pFus_mask = nib.load(PROB_ROI_202_SUB_FILE + 'l_pFus_prob.nii.gz').get_data() > 0

r_OFA_group_mask = nib.load(LABEL_ROI_202_SUB_FILE + 'r_OFA_label.nii.gz').get_data() > 0
l_OFA_group_mask = nib.load(LABEL_ROI_202_SUB_FILE + 'l_OFA_label.nii.gz').get_data() > 0
r_pFus_group__mask = nib.load(LABEL_ROI_202_SUB_FILE + 'r_pFus_label.nii.gz').get_data() > 0
l_pFus_group_mask = nib.load(LABEL_ROI_202_SUB_FILE + 'l_pFus_label.nii.gz').get_data() > 0

manual_img = nib.load(ANALYSIS_DIR + 'manual/' + 'lzg_manual.nii.gz')
affine = manual_img.get_affine()
manual_data = manual_img.get_data()

DEFAULT_BACKGROUND_THR = 200

def process_single_subject_manual_roi(subject_index):
    global image, left_barin_mask, right_barin_mask, manual_data

    skeletonize_markers_RW = np.zeros_like(manual_data[..., subject_index])
    region_result_RW = np.zeros_like(manual_data[..., subject_index])

    #right brain process
    markers = np.zeros_like(image[..., subject_index])
    markers[thin_background_image[..., subject_index] == 1] = 3
    markers[manual_data[..., subject_index] == 1] = 1
    markers[manual_data[..., subject_index] == 3] = 2

    markers[right_barin_mask == False] = -1
    skeletonize_markers_RW[markers == 1] = 1
    skeletonize_markers_RW[markers == 2] = 3
    skeletonize_markers_RW[markers == 3] = 5

    rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
    rw_labels[rw_labels == -1] = 0
    region_result_RW[rw_labels == 1] = 1
    region_result_RW[rw_labels == 2] = 3
    region_result_RW[rw_labels == 3] = 5

    #left brain process
    markers = np.zeros_like(image[..., subject_index])
    markers[thin_background_image[..., subject_index] == 1] = 3
    markers[left_barin_mask == False] = -1
    markers[manual_data[..., subject_index] == 2] = 1
    markers[manual_data[..., subject_index] == 4] = 2

    skeletonize_markers_RW[markers == 1] = 2
    skeletonize_markers_RW[markers == 2] = 4
    skeletonize_markers_RW[markers == 3] = 5

    rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
    rw_labels[rw_labels == -1] = 0
    region_result_RW[rw_labels == 1] = 2
    region_result_RW[rw_labels == 2] = 4
    region_result_RW[rw_labels == 3] = 5


    nib.save(nib.Nifti1Image(skeletonize_markers_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                              '_skeletonize_markers_manual_roi_rw.nii.gz')
    nib.save(nib.Nifti1Image(region_result_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                        '_skeletonize_regions_manual_roi_rw.nii.gz')

    # print 'subject_index:', subject_index, '    atlas_index:', atlas_index
    print 'subject_index:', subject_index, 'atlas-based rw finished...'

    return region_result_RW

def process_single_subject_manual_peak(subject_index):
    global image, complete_atlas_data, left_barin_mask, right_barin_mask

    peak_data = np.load(RESULT_DATA_DIR + 'peak_points_all_sub.npy')
    skeletonize_markers_RW = np.zeros_like(manual_data[..., subject_index])
    region_result_RW = np.zeros_like(manual_data[..., subject_index])

    #right brain process
    markers = np.zeros_like(image[..., subject_index])
    markers[thin_background_image[..., subject_index] == 1] = 3
    markers[peak_data[subject_index, 0, 0], peak_data[subject_index, 0, 1], peak_data[subject_index, 0, 2]] = 1
    markers[peak_data[subject_index, 2, 0], peak_data[subject_index, 2, 1], peak_data[subject_index, 2, 2]] = 2

    markers[right_barin_mask == False] = -1
    skeletonize_markers_RW[markers == 1] = 1
    skeletonize_markers_RW[markers == 2] = 3
    skeletonize_markers_RW[markers == 3] = 5

    rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
    rw_labels[rw_labels == -1] = 0
    region_result_RW[rw_labels == 1] = 1
    region_result_RW[rw_labels == 2] = 3
    region_result_RW[rw_labels == 3] = 5

    #left brain process
    markers = np.zeros_like(image[..., subject_index])
    markers[thin_background_image[..., subject_index] == 1] = 3
    markers[left_barin_mask == False] = -1
    markers[peak_data[subject_index, 1, 0], peak_data[subject_index, 1, 1], peak_data[subject_index, 1, 2]] = 1
    markers[peak_data[subject_index, 3, 0], peak_data[subject_index, 3, 1], peak_data[subject_index, 3, 2]] = 2

    skeletonize_markers_RW[markers == 1] = 2
    skeletonize_markers_RW[markers == 2] = 4
    skeletonize_markers_RW[markers == 3] = 5

    rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
    rw_labels[rw_labels == -1] = 0
    region_result_RW[rw_labels == 1] = 2
    region_result_RW[rw_labels == 2] = 4
    region_result_RW[rw_labels == 3] = 5


    nib.save(nib.Nifti1Image(skeletonize_markers_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                              '_skeletonize_markers_manual_peak_rw.nii.gz')
    nib.save(nib.Nifti1Image(region_result_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                        '_skeletonize_regions_manual_peak_rw.nii.gz')

    # print 'subject_index:', subject_index, '    atlas_index:', atlas_index
    print 'subject_index:', subject_index, 'atlas-based rw finished...'

    return region_result_RW

def process_single_subject_manual_sphere(subject_index, radius=1):
    global image, complete_atlas_data, left_barin_mask, right_barin_mask

    peak_data = np.load(RESULT_DATA_DIR + 'peak_points_all_sub.npy')
    sphere = SpatialNeighbor('sphere', image[..., subject_index].shape, radius)

    r_OFA_coords = sphere.compute(peak_data[subject_index, 0].reshape(1, 3))
    l_OFA_coords = sphere.compute(peak_data[subject_index, 1].reshape(1, 3))
    r_FFA_coords = sphere.compute(peak_data[subject_index, 2].reshape(1, 3))
    l_FFA_coords = sphere.compute(peak_data[subject_index, 3].reshape(1, 3))

    skeletonize_markers_RW = np.zeros_like(manual_data[..., subject_index])
    region_result_RW = np.zeros_like(manual_data[..., subject_index])

    #right brain process
    markers = np.zeros_like(image[..., subject_index])
    markers[thin_background_image[..., subject_index] == 1] = 3
    markers[r_OFA_coords[:, 0], r_OFA_coords[:, 1], r_OFA_coords[:, 2]] = 1
    markers[r_FFA_coords[:, 0], r_FFA_coords[:, 1], r_FFA_coords[:, 2]] = 2

    markers[right_barin_mask == False] = -1
    skeletonize_markers_RW[markers == 1] = 1
    skeletonize_markers_RW[markers == 2] = 3
    skeletonize_markers_RW[markers == 3] = 5

    rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
    rw_labels[rw_labels == -1] = 0
    region_result_RW[rw_labels == 1] = 1
    region_result_RW[rw_labels == 2] = 3
    region_result_RW[rw_labels == 3] = 5

    #left brain process
    markers = np.zeros_like(image[..., subject_index])
    markers[thin_background_image[..., subject_index] == 1] = 3
    markers[left_barin_mask == False] = -1

    markers[l_OFA_coords[:, 0], l_OFA_coords[:, 1], l_OFA_coords[:, 2]] = 1
    markers[l_FFA_coords[:, 0], l_FFA_coords[:, 1], l_FFA_coords[:, 2]] = 2

    skeletonize_markers_RW[markers == 1] = 2
    skeletonize_markers_RW[markers == 2] = 4
    skeletonize_markers_RW[markers == 3] = 5

    rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
    rw_labels[rw_labels == -1] = 0
    region_result_RW[rw_labels == 1] = 2
    region_result_RW[rw_labels == 2] = 4
    region_result_RW[rw_labels == 3] = 5


    nib.save(nib.Nifti1Image(skeletonize_markers_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                              '_skeletonize_markers_manual_peak_sphere_rw_'+ str(radius) + '.nii.gz')
    nib.save(nib.Nifti1Image(region_result_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                        '_skeletonize_regions_manual_peak_sphere_rw_'+ str(radius) + '.nii.gz')

    # print 'subject_index:', subject_index, '    atlas_index:', atlas_index
    print 'subject_index:', subject_index, 'atlas-based rw finished...'

    return region_result_RW

def process_single_subject_manual_regiongrow(subject_index, size=1):
    global image, complete_atlas_data, left_barin_mask, right_barin_mask

    peak_data = np.load(RESULT_DATA_DIR + 'peak_points_all_sub.npy')
    similarity_criteria = SimilarityCriteria('euclidean')
    stop_criteria = StopCriteria('size')
    threshold = np.array((size, ))
    srg = SeededRegionGrowing(similarity_criteria, stop_criteria)
    neighbor_element = SpatialNeighbor('connected', mask.shape, 26)

    r_OFA_seed_coords = np.array([peak_data[subject_index, 0]]).astype(np.int)
    region = Region(r_OFA_seed_coords, neighbor_element)
    srg_region = srg.compute(region, image[..., subject_index], threshold)
    r_OFA_coords = srg_region[0].get_label()

    l_OFA_seed_coords = np.array([peak_data[subject_index, 1]]).astype(np.int)
    region = Region(l_OFA_seed_coords, neighbor_element)
    srg_region = srg.compute(region, image[..., subject_index], threshold)
    l_OFA_coords = srg_region[0].get_label()

    r_FFA_seed_coords = np.array([peak_data[subject_index, 2]]).astype(np.int)
    region = Region(r_FFA_seed_coords, neighbor_element)
    srg_region = srg.compute(region, image[..., subject_index], threshold)
    r_FFA_coords = srg_region[0].get_label()

    l_FFA_seed_coords = np.array([peak_data[subject_index, 3]]).astype(np.int)
    region = Region(l_FFA_seed_coords, neighbor_element)
    srg_region = srg.compute(region, image[..., subject_index], threshold)
    l_FFA_coords = srg_region[0].get_label()

    skeletonize_markers_RW = np.zeros_like(manual_data[..., subject_index])
    region_result_RW = np.zeros_like(manual_data[..., subject_index])

    #right brain process
    markers = np.zeros_like(image[..., subject_index])
    markers[thin_background_image[..., subject_index] == 1] = 3
    markers[r_OFA_coords[:, 0], r_OFA_coords[:, 1], r_OFA_coords[:, 2]] = 1
    markers[r_FFA_coords[:, 0], r_FFA_coords[:, 1], r_FFA_coords[:, 2]] = 2

    markers[right_barin_mask == False] = -1
    skeletonize_markers_RW[markers == 1] = 1
    skeletonize_markers_RW[markers == 2] = 3
    skeletonize_markers_RW[markers == 3] = 5

    rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
    rw_labels[rw_labels == -1] = 0
    region_result_RW[rw_labels == 1] = 1
    region_result_RW[rw_labels == 2] = 3
    region_result_RW[rw_labels == 3] = 5

    #left brain process
    markers = np.zeros_like(image[..., subject_index])
    markers[thin_background_image[..., subject_index] == 1] = 3
    markers[left_barin_mask == False] = -1

    markers[l_OFA_coords[:, 0], l_OFA_coords[:, 1], l_OFA_coords[:, 2]] = 1
    markers[l_FFA_coords[:, 0], l_FFA_coords[:, 1], l_FFA_coords[:, 2]] = 2

    skeletonize_markers_RW[markers == 1] = 2
    skeletonize_markers_RW[markers == 2] = 4
    skeletonize_markers_RW[markers == 3] = 5

    rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
    rw_labels[rw_labels == -1] = 0
    region_result_RW[rw_labels == 1] = 2
    region_result_RW[rw_labels == 2] = 4
    region_result_RW[rw_labels == 3] = 5


    nib.save(nib.Nifti1Image(skeletonize_markers_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                              '_skeletonize_markers_manual_peak_rg_'+ str(size) + '.nii.gz')
    nib.save(nib.Nifti1Image(region_result_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                        '_skeletonize_regions_manual_peak_rg_'+ str(size) + '.nii.gz')

    # print 'subject_index:', subject_index, '    atlas_index:', atlas_index
    print 'subject_index:', subject_index, 'atlas-based rw finished...'

    return region_result_RW



if __name__ == "__main__":
    starttime = datetime.datetime.now()

    img = nib.load(ALL_PROB_MASK)
    mask = img.get_data()
    affine = img.get_affine()

    if not os.path.exists(RW_AGGRAGATOR_RESULT_DATA_DIR):
        os.makedirs(RW_AGGRAGATOR_RESULT_DATA_DIR)

    # for subject_index in range(0, 7): #by sessions
    for subject_index in range(0, 7): #by sessions
        process_single_subject_manual_roi(subject_index)
        process_single_subject_manual_peak(subject_index)
        process_single_subject_manual_sphere(subject_index, radius=1)
        process_single_subject_manual_sphere(subject_index, radius=2)
        process_single_subject_manual_sphere(subject_index, radius=3)
        process_single_subject_manual_sphere(subject_index, radius=4)
        process_single_subject_manual_sphere(subject_index, radius=5)

        process_single_subject_manual_regiongrow(subject_index, size=10)
        process_single_subject_manual_regiongrow(subject_index, size=30)
        process_single_subject_manual_regiongrow(subject_index, size=50)
        process_single_subject_manual_regiongrow(subject_index, size=100)

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."

































