__author__ = 'zgf'
__author__ = 'zhouguangfu'

import datetime
import numpy as np
import nibabel as nib
import os

from random_walker.multi_processs.marker_test.segment import *
from random_walker.multi_processs.marker_test.imtool import *
from configs import *
from skimage.segmentation import random_walker

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

r_OFA_label_mask = nib.load(LABEL_ROI_202_SUB_FILE + 'r_OFA_label.nii.gz').get_data() > 0
l_OFA_label_mask = nib.load(LABEL_ROI_202_SUB_FILE + 'l_OFA_label.nii.gz').get_data() > 0
r_pFus_label__mask = nib.load(LABEL_ROI_202_SUB_FILE + 'r_pFus_label.nii.gz').get_data() > 0
l_pFus_label_mask = nib.load(LABEL_ROI_202_SUB_FILE + 'l_pFus_label.nii.gz').get_data() > 0

manual_img = nib.load(ANALYSIS_DIR + 'manual/' + 'lzg_manual.nii.gz')
affine = manual_img.get_affine()
manual_data = manual_img.get_data()

def compute_OFA_FFA_mean_prob_peak_distance():
    r_OFA_data = np.zeros_like(r_OFA_mask)
    l_OFA_data = np.zeros_like(l_OFA_mask)
    r_FFA_data = np.zeros_like(r_pFus_mask)
    l_FFA_data = np.zeros_like(l_pFus_mask)

    r_OFA_data[r_OFA_label_mask > 0] = r_OFA_mask[r_OFA_label_mask > 0]
    l_OFA_data[l_OFA_label_mask > 0] = l_OFA_mask[l_OFA_label_mask > 0]
    r_FFA_data[r_pFus_label__mask > 0] = r_pFus_mask[r_pFus_label__mask > 0]
    l_FFA_data[l_pFus_label_mask > 0] = l_pFus_mask[l_pFus_label_mask > 0]

    peaks =  np.array([np.unravel_index(r_OFA_data.argmax(), r_OFA_data.shape),
                     np.unravel_index(l_OFA_data.argmax(), l_OFA_data.shape),
                     np.unravel_index(r_FFA_data.argmax(), r_FFA_data.shape),
                     np.unravel_index(l_FFA_data.argmax(), l_FFA_data.shape)])
    print "Prob Peaks: ", peaks
    print 'The distance of r_OFA and r_pFus : ', np.linalg.norm(peaks[0] - peaks[2])
    print 'The distance of l_OFA and l_pFus : ', np.linalg.norm(peaks[1] - peaks[3])
    mean_distance = (np.linalg.norm(peaks[0] - peaks[2]) + np.linalg.norm(peaks[1] - peaks[3])) / 2.0
    print 'The mean distance of l_OFA and l_pFus : ', mean_distance
    return mean_distance

def compute_label_peak(atlas_data, subject_index):
    r_OFA_data = np.zeros_like(atlas_data)
    l_OFA_data = np.zeros_like(atlas_data)
    r_pFus_data = np.zeros_like(atlas_data)
    l_pFus_data = np.zeros_like(atlas_data)

    r_OFA_data[atlas_data == 1] = image[atlas_data == 1, subject_index]
    l_OFA_data[atlas_data == 2] = image[atlas_data == 2, subject_index]
    r_pFus_data[atlas_data == 3] = image[atlas_data == 3, subject_index]
    l_pFus_data[atlas_data == 4] = image[atlas_data == 4, subject_index]

    peaks = np.array([np.unravel_index(r_OFA_data.argmax(), r_OFA_data.shape),
                      np.unravel_index(l_OFA_data.argmax(), l_OFA_data.shape),
                      np.unravel_index(r_pFus_data.argmax(), r_pFus_data.shape),
                      np.unravel_index(l_pFus_data.argmax(), l_pFus_data.shape)])
    print "Peaks of atlas label: ", peaks
    return peaks

def compute_parcel_peak(subject_index):
    localmax_cords = local_maximum(image[..., subject_index], 2)
    nib.save(nib.Nifti1Image(localmax_cords, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                  '_watershed_localmax.nii.gz')
    return np.array(np.nonzero(localmax_cords)).T

def select_optimal_parcel_min_distance(subject_index, size=None):
    #get the atlas data
    atlas_data = np.zeros_like(complete_atlas_data[..., subject_index])
    atlas_index= 0
    top_atlas_data = np.load(ATLAS_TOP_DIR + 'old_threshold_0/' + str(subject_index) + '_top_sort.npy')
    atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 1] = 1
    atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 2] = 2
    atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 3] = 3
    atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 4] = 4

    #get peaks
    atlas_label_peaks = compute_label_peak(atlas_data, subject_index)
    image_peaks = compute_parcel_peak(subject_index)

    print 'image_peaks.shape: ', image_peaks.shape
    r_OFA_distances = np.linalg.norm((image_peaks - atlas_label_peaks[0]), axis=1)
    l_OFA_distances = np.linalg.norm((image_peaks - atlas_label_peaks[1]), axis=1)
    r_pFus_distances = np.linalg.norm((image_peaks - atlas_label_peaks[2]), axis=1)
    l_pFus_distances = np.linalg.norm((image_peaks - atlas_label_peaks[3]), axis=1)

    markers, seg_input, watershed_volume = watershed(image[..., subject_index], 0, 2.3, None, inverse_transformation)
    nib.save(nib.Nifti1Image(watershed_volume, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                        '_watershed.nii.gz')
    nib.save(nib.Nifti1Image(atlas_data, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                  '_watershed_top_one_atlas.nii.gz')

    mean_OFA_FFA_distance = compute_OFA_FFA_mean_prob_peak_distance()
    #r_OFA
    r_OFA_optimal_label_cord = image_peaks[r_OFA_distances.argmin()]
    r_OFA_optimal_label_value = watershed_volume[r_OFA_optimal_label_cord[0],
                                                 r_OFA_optimal_label_cord[1],
                                                 r_OFA_optimal_label_cord[2]]
    if r_OFA_distances.min() > mean_OFA_FFA_distance:
        r_OFA_optimal_label_value = 0
        print '!!!!!!!!!!!r_OFA_distances > mean_OFA_FFA_distance error!'
    #l_OFA
    l_OFA_optimal_label_cord = image_peaks[l_OFA_distances.argmin()]
    l_OFA_optimal_label_value = watershed_volume[l_OFA_optimal_label_cord[0],
                                                 l_OFA_optimal_label_cord[1],
                                                 l_OFA_optimal_label_cord[2]]
    if l_OFA_distances.min() > mean_OFA_FFA_distance:
        l_OFA_optimal_label_value = 0
        print '!!!!!!!!!!!l_OFA_distances > mean_OFA_FFA_distance error!'
    #r_pFus
    r_pFus_optimal_label_cord = image_peaks[r_pFus_distances.argmin()]
    r_pFus_optimal_label_value = watershed_volume[r_pFus_optimal_label_cord[0],
                                                  r_pFus_optimal_label_cord[1],
                                                  r_pFus_optimal_label_cord[2]]
    if r_pFus_distances.min() > mean_OFA_FFA_distance:
        r_pFus_optimal_label_value = 0
        print '!!!!!!!!!!!r_pFus_distances > mean_OFA_FFA_distance error!'
    #l_pFus
    l_pFus_optimal_label_cord = image_peaks[l_pFus_distances.argmin()]
    l_pFus_optimal_label_value = watershed_volume[l_pFus_optimal_label_cord[0],
                                                  l_pFus_optimal_label_cord[1],
                                                  l_pFus_optimal_label_cord[2]]
    if l_pFus_distances.min() > mean_OFA_FFA_distance:
        l_pFus_optimal_label_value = 0
        print '!!!!!!!!!!!l_pFus_distances > mean_OFA_FFA_distance error!'

    #print
    print '------------------------------------------'
    print 'r_OFA_distances.min(): ', r_OFA_distances.min()
    print 'r_OFA_optimal_label_value: ', r_OFA_optimal_label_value
    print 'r_OFA_optimal_label_cord: ', r_OFA_optimal_label_cord
    print '------------------------------------------'
    print 'l_OFA_distances.min(): ', l_OFA_distances.min()
    print 'l_OFA_optimal_label_value: ', l_OFA_optimal_label_value
    print 'l_OFA_optimal_label_cord: ', l_OFA_optimal_label_cord
    print '------------------------------------------'
    print 'r_pFus_distances.min(): ', r_pFus_distances.min()
    print 'r_pFus_optimal_label_value: ', r_pFus_optimal_label_value
    print 'r_pFus_optimal_label_cord: ', r_pFus_optimal_label_cord
    print '------------------------------------------'
    print 'l_pFus_distances.min(): ', l_pFus_distances.min()
    print 'l_pFus_optimal_label_value: ', l_pFus_optimal_label_value
    print 'l_pFus_optimal_label_cord: ', l_pFus_optimal_label_cord

    #----------------------------------------------------------------------------------------------
    region_result_RW = np.zeros_like(image[..., subject_index])
    skeletonize_markers_RW = np.zeros_like(image[..., subject_index])
    atlas_data = np.zeros_like(complete_atlas_data[..., subject_index])

    r_OFA_parcels = (watershed_volume == r_OFA_optimal_label_value)
    l_OFA_parcels = (watershed_volume == l_OFA_optimal_label_value)
    r_pFus_parcels = (watershed_volume == r_pFus_optimal_label_value)
    l_pFus_parcels = (watershed_volume == l_pFus_optimal_label_value)

    #right brain process
    markers = np.zeros_like(image[..., subject_index])
    markers[r_OFA_parcels] = 1
    markers[r_pFus_parcels] = 2
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
    markers[l_pFus_parcels] = 2
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

def select_optimal_parcel_max_region_mean(subject_index, size=None):
    #get the atlas data
    atlas_data = np.zeros_like(complete_atlas_data[..., subject_index])
    atlas_index= 1
    top_atlas_data = np.load(ATLAS_TOP_DIR + 'old_threshold_0/' + str(subject_index) + '_top_sort.npy')
    atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 1] = 1
    atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 2] = 2
    atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 3] = 3
    atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 4] = 4

    #get peaks
    atlas_label_peaks = compute_label_peak(atlas_data, subject_index)
    image_peaks = compute_parcel_peak(subject_index)

    print 'image_peaks.shape: ', image_peaks.shape
    r_OFA_distances = np.linalg.norm((image_peaks - atlas_label_peaks[0]), axis=1)
    l_OFA_distances = np.linalg.norm((image_peaks - atlas_label_peaks[1]), axis=1)
    r_pFus_distances = np.linalg.norm((image_peaks - atlas_label_peaks[2]), axis=1)
    l_pFus_distances = np.linalg.norm((image_peaks - atlas_label_peaks[3]), axis=1)

    markers, seg_input, watershed_volume = watershed(image[..., subject_index], 0, 0, None, inverse_transformation)
    nib.save(nib.Nifti1Image(watershed_volume, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                        '_watershed.nii.gz')
    nib.save(nib.Nifti1Image(atlas_data, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                  '_watershed_top_one_atlas.nii.gz')

    mean_OFA_FFA_distance = compute_OFA_FFA_mean_prob_peak_distance() / 2.0

    #r_OFA
    r_OFA_distances_argsort = r_OFA_distances.argsort()
    r_OFA_max_region_mean_index = 0
    r_OFA_max_region_mean_value = -10000
    for i in range((r_OFA_distances <= mean_OFA_FFA_distance).sum()):
        r_OFA_cord = image_peaks[r_OFA_distances_argsort[i]]
        r_OFA_label_value = watershed_volume[r_OFA_cord[0],
                                             r_OFA_cord[1],
                                             r_OFA_cord[2]]
        region_mean = image[watershed_volume == r_OFA_label_value, subject_index].mean()
        if r_OFA_max_region_mean_value < region_mean:
            r_OFA_max_region_mean_index = r_OFA_distances_argsort[i]
            r_OFA_optimal_label_value = r_OFA_label_value
            r_OFA_max_region_mean_value = region_mean

    if r_OFA_optimal_label_value == 0:
        print '!!!!!!!!!!!r_OFA_distances = 0 error!'

    #l_OFA
    l_OFA_distances_argsort = l_OFA_distances.argsort()
    l_OFA_max_region_mean_index = 0
    l_OFA_max_region_mean_value = -10000
    for i in range((l_OFA_distances <= mean_OFA_FFA_distance).sum()):
        l_OFA_cord = image_peaks[l_OFA_distances_argsort[i]]
        l_OFA_label_value = watershed_volume[l_OFA_cord[0],
                                             l_OFA_cord[1],
                                             l_OFA_cord[2]]
        region_mean = image[watershed_volume == l_OFA_label_value, subject_index].mean()
        if l_OFA_max_region_mean_value < region_mean:
            l_OFA_max_region_mean_index = l_OFA_distances_argsort[i]
            l_OFA_optimal_label_value = l_OFA_label_value
            l_OFA_max_region_mean_value = region_mean

    if l_OFA_optimal_label_value == 0:
        print '!!!!!!!!!!!l_OFA_distances = 0 error!'

    #r_pFus
    r_pFus_distances_argsort = r_pFus_distances.argsort()
    r_pFus_max_region_mean_index = 0
    r_pFus_max_region_mean_value = -10000
    for i in range((r_pFus_distances <= mean_OFA_FFA_distance).sum()):
        r_pFus_cord = image_peaks[r_pFus_distances_argsort[i]]
        r_pFus_label_value = watershed_volume[r_pFus_cord[0],
                                              r_pFus_cord[1],
                                              r_pFus_cord[2]]
        region_mean = image[watershed_volume == r_pFus_label_value, subject_index].mean()
        if r_pFus_max_region_mean_value < region_mean:
            r_pFus_max_region_mean_index = r_pFus_distances_argsort[i]
            r_pFus_optimal_label_value = r_pFus_label_value
            r_pFus_max_region_mean_value = region_mean

    if r_pFus_optimal_label_value == 0:
        print '!!!!!!!!!r_pFus_distances = 0 error!'

    #l_pFus
    l_pFus_distances_argsort = l_pFus_distances.argsort()
    l_pFus_max_region_mean_index = 0
    l_pFus_max_region_mean_value = -10000
    for i in range((l_pFus_distances <= mean_OFA_FFA_distance).sum()):
        l_pFus_cord = image_peaks[l_pFus_distances_argsort[i]]
        l_pFus_label_value = watershed_volume[l_pFus_cord[0],
                                              l_pFus_cord[1],
                                              l_pFus_cord[2]]
        region_mean = image[watershed_volume == l_pFus_label_value, subject_index].mean()
        if l_pFus_max_region_mean_value < region_mean:
            l_pFus_max_region_mean_index = l_pFus_distances_argsort[i]
            l_pFus_optimal_label_value = l_pFus_label_value
            l_pFus_max_region_mean_value = region_mean

    if l_pFus_optimal_label_value == 0:
        print '!!!!!!!!!!!l_pFus_distances = 0 error!'

    #print
    print '------------------------------------------'
    print 'optimal r_OFA_distances: ', r_OFA_distances[r_OFA_max_region_mean_index]
    print 'r_OFA_optimal_label_value: ', r_OFA_optimal_label_value
    print 'r_OFA_optimal_label_cord: ', image_peaks[r_OFA_max_region_mean_index]
    print '------------------------------------------'
    print 'optimal l_OFA_distances: ', l_OFA_distances[l_OFA_max_region_mean_index]
    print 'l_OFA_optimal_label_value: ', l_OFA_optimal_label_value
    print 'l_OFA_optimal_label_cord: ', image_peaks[l_OFA_max_region_mean_index]
    print '------------------------------------------'
    print 'optimal r_pFus_distances: ', r_pFus_distances[r_pFus_max_region_mean_index]
    print 'r_pFus_optimal_label_value: ', r_pFus_optimal_label_value
    print 'r_pFus_optimal_label_cord: ', image_peaks[r_pFus_max_region_mean_index]
    print '------------------------------------------'
    print 'optimal l_pFus_distances: ', l_pFus_distances[l_pFus_max_region_mean_index]
    print 'l_pFus_optimal_label_value: ', l_pFus_optimal_label_value
    print 'l_pFus_optimal_label_cord: ', image_peaks[l_pFus_max_region_mean_index]

    #----------------------------------------------------------------------------------------------
    region_result_RW = np.zeros_like(image[..., subject_index])
    skeletonize_markers_RW = np.zeros_like(image[..., subject_index])
    atlas_data = np.zeros_like(complete_atlas_data[..., subject_index])

    r_OFA_parcels = (watershed_volume == r_OFA_optimal_label_value)
    l_OFA_parcels = (watershed_volume == l_OFA_optimal_label_value)
    r_pFus_parcels = (watershed_volume == r_pFus_optimal_label_value)
    l_pFus_parcels = (watershed_volume == l_pFus_optimal_label_value)

    #right brain process
    markers = np.zeros_like(image[..., subject_index])
    markers[r_OFA_parcels] = 1
    markers[r_pFus_parcels] = 2
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
    markers[l_pFus_parcels] = 2
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
                                                              '_skeletonize_markers_top_one_atlas_watershed_max_region_mean.nii.gz')
    nib.save(nib.Nifti1Image(region_result_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                        '_skeletonize_regions_top_one_atlas_watershed_max_region_mean.nii.gz')

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
    for subject_index in range(0, 1): #by sessions
        # select_optimal_parcel_min_distance(subject_index)
        select_optimal_parcel_max_region_mean(subject_index)

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."































