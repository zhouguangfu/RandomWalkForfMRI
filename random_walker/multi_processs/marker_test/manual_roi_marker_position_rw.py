__author__ = 'zhouguangfu'

import datetime
import numpy as np
import nibabel as nib
import os
import segment

from configs import *
from skimage.segmentation import random_walker
from segment import watershed

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

def process_single_subject_top_one_atlas_roi(subject_index):
    global image, complete_atlas_data, left_barin_mask, right_barin_mask

    region_result_RW = np.zeros_like(image[..., subject_index])
    skeletonize_markers_RW = np.zeros_like(image[..., subject_index])
    atlas_data = np.zeros_like(complete_atlas_data[..., subject_index])

    # r_OFA_indexs =  np.load(ATLAS_TOP_DIR + ROI[0] + '_' + str(subject_index) + '_top_sort.npy')
    # l_OFA_indexs =  np.load(ATLAS_TOP_DIR + ROI[1] + '_' + str(subject_index) + '_top_sort.npy')
    # r_pFus_indexs =  np.load(ATLAS_TOP_DIR + ROI[2] + '_' + str(subject_index) + '_top_sort.npy')
    # l_pFus_indexs =  np.load(ATLAS_TOP_DIR + ROI[3] + '_' + str(subject_index) + '_top_sort.npy')

    atlas_index= 0
    # atlas_data[complete_atlas_data[..., r_OFA_indexs[atlas_index]] == 1] = 1
    # atlas_data[complete_atlas_data[..., l_OFA_indexs[atlas_index]] == 2] = 2
    # atlas_data[complete_atlas_data[..., r_pFus_indexs[atlas_index]] == 3] = 3
    # atlas_data[complete_atlas_data[..., l_pFus_indexs[atlas_index]] == 4] = 4

    top_atlas_data = np.load(ATLAS_TOP_DIR + 'old_threshold_0/' + str(subject_index) + '_top_sort.npy')

    atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 1] = 1
    atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 2] = 2
    atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 3] = 3
    atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 4] = 4

    #right brain process
    markers = np.zeros_like(image[..., subject_index])
    markers[atlas_data == 1] = 1
    markers[atlas_data == 3] = 2
    markers[thin_background_image[..., subject_index] == 1] = 3
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
    markers[atlas_data == 2] = 1
    markers[atlas_data == 4] = 2
    markers[thin_background_image[..., subject_index] == 1] = 3
    markers[left_barin_mask == False] = -1

    skeletonize_markers_RW[markers == 1] = 2
    skeletonize_markers_RW[markers == 2] = 4
    skeletonize_markers_RW[markers == 3] = 5

    rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
    rw_labels[rw_labels == -1] = 0
    region_result_RW[rw_labels == 1] = 2
    region_result_RW[rw_labels == 2] = 4
    region_result_RW[rw_labels == 3] = 5


    nib.save(nib.Nifti1Image(skeletonize_markers_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                              '_skeletonize_markers_top_one_atlas_roi.nii.gz')
    nib.save(nib.Nifti1Image(region_result_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                        '_skeletonize_regions_top_one_atlas_roi.nii.gz')

    print 'subject_index:', subject_index, 'atlas-based rw finished...'

    return region_result_RW

def process_single_subject_top_one_atlas_watershed(subject_index):
    global image, complete_atlas_data, left_barin_mask, right_barin_mask

    region_result_RW = np.zeros_like(image[..., subject_index])
    skeletonize_markers_RW = np.zeros_like(image[..., subject_index])
    atlas_data = np.zeros_like(complete_atlas_data[..., subject_index])

    # r_OFA_indexs =  np.load(ATLAS_TOP_DIR + ROI[0] + '_' + str(subject_index) + '_top_sort.npy')
    # l_OFA_indexs =  np.load(ATLAS_TOP_DIR + ROI[1] + '_' + str(subject_index) + '_top_sort.npy')
    # r_pFus_indexs =  np.load(ATLAS_TOP_DIR + ROI[2] + '_' + str(subject_index) + '_top_sort.npy')
    # l_pFus_indexs =  np.load(ATLAS_TOP_DIR + ROI[3] + '_' + str(subject_index) + '_top_sort.npy')

    atlas_index= 0
    # atlas_data[complete_atlas_data[..., r_OFA_indexs[atlas_index]] == 1] = 1
    # atlas_data[complete_atlas_data[..., l_OFA_indexs[atlas_index]] == 2] = 2
    # atlas_data[complete_atlas_data[..., r_pFus_indexs[atlas_index]] == 3] = 3
    # atlas_data[complete_atlas_data[..., l_pFus_indexs[atlas_index]] == 4] = 4


    top_atlas_data = np.load(ATLAS_TOP_DIR + 'old_threshold_0/' + str(subject_index) + '_top_sort.npy')

    atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 1] = 1
    atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 2] = 2
    atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 3] = 3
    atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 4] = 4

    markers, seg_input, watershed_volume = watershed(image[..., subject_index], 1, 2.3, None, segment.inverse_transformation)
    nib.save(nib.Nifti1Image(watershed_volume, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                    '_watershed.nii.gz')
    nib.save(nib.Nifti1Image(atlas_data, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                        '_watershed_top_one_atlas.nii.gz')

    #---------------------------------------------------------------------------------------------------
    r_OFA_parcels = watershed_volume[atlas_data == 1]
    l_OFA_parcels = watershed_volume[atlas_data == 2]
    r_FFA_parcels = watershed_volume[atlas_data == 3]
    l_FFA_parcels = watershed_volume[atlas_data == 4]

    #r_OFA
    r_OFA_means = []
    r_OFA_unique_values = np.unique(r_OFA_parcels)

    print "subject_index: ", subject_index, "   r_OFA_unique_values: ", r_OFA_unique_values
    nib.save(nib.Nifti1Image((atlas_data == 1).astype(np.int), affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                  'test.nii.gz')
    if(r_OFA_unique_values.size == 1):
        r_OFA_parcels = (atlas_data == 1)
    else:
        for value in r_OFA_unique_values:
            if value != 0:
                r_OFA_means.append(image[watershed_volume == value, subject_index].mean())
        index = np.array(r_OFA_means).argmax() + 1
        r_OFA_parcels = (watershed_volume == r_OFA_unique_values[index])
        print 'r_OFA_unique_values: ', r_OFA_unique_values
        print 'r_OFA_unique_values[index]: ', r_OFA_unique_values[index]

    #l_OFA
    l_OFA_means = []
    l_OFA_unique_values = np.unique(l_OFA_parcels)
    if(l_OFA_unique_values.size == 1):
        l_OFA_parcels = (atlas_data == 2)
    else:
        for value in l_OFA_unique_values:
            if value != 0:
                l_OFA_means.append(image[watershed_volume == value, subject_index].mean())
        index = np.array(l_OFA_means).argmax() + 1
        l_OFA_parcels = (watershed_volume == l_OFA_unique_values[index])
        print 'l_OFA_unique_values[index]: ', l_OFA_unique_values[index]

    #r_FFA
    r_FFA_means = []
    r_FFA_unique_values = np.unique(r_FFA_parcels)
    if(r_FFA_unique_values.size == 1):
        r_FFA_parcels = (atlas_data == 3)
    else:
        for value in r_FFA_unique_values:
            if value != 0:
                r_FFA_means.append(image[watershed_volume == value, subject_index].mean())
        index = np.array(r_FFA_means).argmax() + 1
        r_FFA_parcels = (watershed_volume == r_FFA_unique_values[index])

    #l_FFA
    l_FFA_means = []
    l_FFA_unique_values = np.unique(l_FFA_parcels)
    if(l_FFA_unique_values.size == 1):
        l_FFA_parcels = (atlas_data == 4)
    else:
        for value in l_FFA_unique_values:
            if value != 0:
                l_FFA_means.append(image[watershed_volume == value, subject_index].mean())
        index = np.array(l_FFA_means).argmax() + 1
        l_FFA_parcels = (watershed_volume == l_FFA_unique_values[index])

    #-----------------------------------------------------------------------------------------------------

    #right brain process
    markers = np.zeros_like(image[..., subject_index])
    markers[r_OFA_parcels] = 1
    markers[r_FFA_parcels] = 2
    markers[thin_background_image[..., subject_index] == 1] = 3
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
    markers[l_OFA_parcels] = 1
    markers[l_FFA_parcels] = 2
    markers[thin_background_image[..., subject_index] == 1] = 3
    markers[left_barin_mask == False] = -1

    skeletonize_markers_RW[markers == 1] = 2
    skeletonize_markers_RW[markers == 2] = 4
    skeletonize_markers_RW[markers == 3] = 5

    rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
    rw_labels[rw_labels == -1] = 0
    region_result_RW[rw_labels == 1] = 2
    region_result_RW[rw_labels == 2] = 4
    region_result_RW[rw_labels == 3] = 5


    nib.save(nib.Nifti1Image(skeletonize_markers_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                              '_skeletonize_markers_top_one_atlas_watershed.nii.gz')
    nib.save(nib.Nifti1Image(region_result_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                        '_skeletonize_regions_top_one_atlas_watershed.nii.gz')

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
    for subject_index in range(4, 5): #by sessions
        process_single_subject_top_one_atlas_roi(subject_index)
        process_single_subject_top_one_atlas_watershed(subject_index)



    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."


    line = '\r\n'
    line.rstrip('\r\n')






























