__author__ = 'zgf'

import datetime
import numpy as np
import nibabel as nib
import os

from configs import *
from skimage.segmentation import random_walker
from skimage.segmentation import slic

DEFAULT_TOP_RANK = 202
SUBJECT_NUM = 3
SUPERVOXEL_SEGMENTATION = 10000

#global varibale
image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

complete_atlas_data = nib.load(ATLAS_TOP_DIR + 'complete_atlas_label.nii.gz').get_data()
complete_image_data = nib.load(ALL_202_SUBJECTS_DATA_DIR).get_data()

left_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_LEFT_BRAIN_FILE).get_data() > 0
right_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_RIGHT_BRAIN_FILE).get_data() > 0

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
    # print "Prob Peaks: ", peaks
    # print 'The distance of r_OFA and r_pFus : ', np.linalg.norm(peaks[0] - peaks[2])
    # print 'The distance of l_OFA and l_pFus : ', np.linalg.norm(peaks[1] - peaks[3])
    mean_distance = (np.linalg.norm(peaks[0] - peaks[2]) + np.linalg.norm(peaks[1] - peaks[3])) / 2.0
    # print 'The mean distance of l_OFA and l_pFus : ', mean_distance
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
    return peaks

def compute_parcel_peak(subject_index):
    # localmax_cords = local_maximum(image[..., subject_index], 2)
    # nib.save(nib.Nifti1Image(localmax_cords, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
    #                                               '_supervoxel_localmax.nii.gz')
    slice = image[..., subject_index]
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

def compute_background_parcel(subject_index, slic_image):
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

    nib.save(nib.Nifti1Image(left_brain_dilation.astype(np.int), affine),
             RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) + '_left_bk_marker_dilation.nii.gz')
    nib.save(nib.Nifti1Image(right_brain_dilation.astype(np.int), affine),
             RW_AGGRAGATOR_RESULT_DATA_DIR +  str(subject_index) + '_right_bk_marker_dilation.nii.gz')
    nib.save(nib.Nifti1Image(left_brain_background_marker, affine),
             RW_AGGRAGATOR_RESULT_DATA_DIR +  str(subject_index) + '_left_bk_marker.nii.gz')
    nib.save(nib.Nifti1Image(right_brain_background_marker, affine),
             RW_AGGRAGATOR_RESULT_DATA_DIR +  str(subject_index) + '_right_bk_marker.nii.gz')

    return left_brain_background_marker, right_brain_background_marker

def select_optimal_parcel_max_region_mean(subject_index, size=None):
    #get the atlas data
    top_atlas_data = np.load(ATLAS_TOP_DIR + 'old_threshold_0/' + str(subject_index) + '_top_sort.npy')
    region_results_RW = np.zeros((image.shape[0], image.shape[1], image.shape[2], DEFAULT_TOP_RANK))
    marker_results_RW = np.zeros((image.shape[0], image.shape[1], image.shape[2], DEFAULT_TOP_RANK))
    supervoxel_top_one_atlas_results_RW = np.zeros((image.shape[0], image.shape[1], image.shape[2], DEFAULT_TOP_RANK))
    top_atlas_image_RW = np.zeros((image.shape[0], image.shape[1], image.shape[2], DEFAULT_TOP_RANK))

    mean_OFA_FFA_distance = compute_OFA_FFA_mean_prob_peak_distance()
    image_peaks, slic_image = compute_parcel_peak(subject_index)
    left_brain_background_marker, right_brain_background_marker = compute_background_parcel(subject_index, slic_image)

    for atlas_index in range(DEFAULT_TOP_RANK):
        atlas_data = np.zeros_like(complete_atlas_data[..., subject_index])

        if (atlas_data == 1).sum() == 0:
            atlas_data[r_OFA_label_mask] = 1
        else:
            atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 1] = 1
        if (atlas_data == 2).sum() == 0:
            atlas_data[l_OFA_label_mask] = 2
        else:
            atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 2] = 2
        if (atlas_data == 3).sum() == 0:
            atlas_data[r_pFus_label__mask] = 3
        else:
            atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 3] = 3
        if (atlas_data == 4).sum() == 0:
            atlas_data[l_pFus_label_mask] = 4
        else:
            atlas_data[complete_atlas_data[..., top_atlas_data[atlas_index]] == 4] = 4

        supervoxel_top_one_atlas_results_RW[..., atlas_index] = atlas_data
        top_atlas_image_RW[..., atlas_index] = complete_image_data[..., top_atlas_data[atlas_index]]
        #get peaks
        atlas_label_peaks = compute_label_peak(atlas_data, subject_index)

        r_OFA_distances = np.linalg.norm((image_peaks - atlas_label_peaks[0]), axis=1)
        l_OFA_distances = np.linalg.norm((image_peaks - atlas_label_peaks[1]), axis=1)
        r_pFus_distances = np.linalg.norm((image_peaks - atlas_label_peaks[2]), axis=1)
        l_pFus_distances = np.linalg.norm((image_peaks - atlas_label_peaks[3]), axis=1)

        #generate the marker
        #r_pFus
        r_pFus_distances_argsort = r_pFus_distances.argsort()
        r_pFus_max_region_mean_value = -10000
        r_pFus_optimal_label_value = -1
        for i in range((r_pFus_distances <= mean_OFA_FFA_distance).sum()):
            r_pFus_cord = image_peaks[r_pFus_distances_argsort[i]]
            if r_pFus_mask[r_pFus_cord[0], r_pFus_cord[1], r_pFus_cord[2]] == 0:
                continue
            r_pFus_label_value = slic_image[r_pFus_cord[0],
                                            r_pFus_cord[1],
                                            r_pFus_cord[2]]
            region_mean = image[slic_image == r_pFus_label_value, subject_index].mean()
            if r_pFus_max_region_mean_value < region_mean:
                r_pFus_optimal_label_value = r_pFus_label_value
                r_pFus_max_region_mean_value = region_mean

        if r_pFus_optimal_label_value == -1:
            print '!!!!!!!!!r_pFus_distances = -1 error!'

        #l_pFus
        l_pFus_distances_argsort = l_pFus_distances.argsort()
        l_pFus_max_region_mean_value = -10000
        l_pFus_optimal_label_value = -1
        for i in range((l_pFus_distances <= mean_OFA_FFA_distance).sum()):
            l_pFus_cord = image_peaks[l_pFus_distances_argsort[i]]
            if l_pFus_mask[l_pFus_cord[0], l_pFus_cord[1], l_pFus_cord[2]] == 0:
                continue
            l_pFus_label_value = slic_image[l_pFus_cord[0],
                                            l_pFus_cord[1],
                                            l_pFus_cord[2]]
            region_mean = image[slic_image == l_pFus_label_value, subject_index].mean()
            if l_pFus_max_region_mean_value < region_mean:
                l_pFus_optimal_label_value = l_pFus_label_value
                l_pFus_max_region_mean_value = region_mean

        if l_pFus_optimal_label_value == -1:
            print '!!!!!!!!!!!l_pFus_distances = -1 error!'

        #r_OFA
        r_OFA_distances_argsort = r_OFA_distances.argsort()
        r_OFA_max_region_mean_value = -10000
        r_OFA_optimal_label_value = -1
        for i in range((r_OFA_distances <= mean_OFA_FFA_distance).sum()):
            r_OFA_cord = image_peaks[r_OFA_distances_argsort[i]]
            if r_OFA_mask[r_OFA_cord[0], r_OFA_cord[1], r_OFA_cord[2]] == 0:
                continue
            r_OFA_label_value = slic_image[r_OFA_cord[0],
                                           r_OFA_cord[1],
                                           r_OFA_cord[2]]
            region_mean = image[slic_image == r_OFA_label_value, subject_index].mean()
            if r_OFA_max_region_mean_value < region_mean:
                if r_OFA_label_value != r_pFus_optimal_label_value:
                    r_OFA_optimal_label_value = r_OFA_label_value
                    r_OFA_max_region_mean_value = region_mean

        if r_OFA_optimal_label_value == -1:
            print '!!!!!!!!!!!r_OFA_distances = -1 error!'

        #l_OFA
        l_OFA_distances_argsort = l_OFA_distances.argsort()
        l_OFA_max_region_mean_value = -10000
        l_OFA_optimal_label_value = -1
        for i in range((l_OFA_distances <= mean_OFA_FFA_distance).sum()):
            l_OFA_cord = image_peaks[l_OFA_distances_argsort[i]]
            if l_OFA_mask[l_OFA_cord[0], l_OFA_cord[1], l_OFA_cord[2]] == 0:
                continue
            l_OFA_label_value = slic_image[l_OFA_cord[0],
                                           l_OFA_cord[1],
                                           l_OFA_cord[2]]
            region_mean = image[slic_image == l_OFA_label_value, subject_index].mean()
            if l_OFA_max_region_mean_value < region_mean:
                if l_OFA_label_value != l_pFus_optimal_label_value:
                    l_OFA_optimal_label_value = l_OFA_label_value
                    l_OFA_max_region_mean_value = region_mean

        if l_OFA_optimal_label_value == -1:
            print '!!!!!!!!!!!l_OFA_distances = -1 error!'

        region_result_RW = np.zeros_like(image[..., subject_index])
        skeletonize_markers_RW = np.zeros_like(image[..., subject_index])
        atlas_data = np.zeros_like(complete_atlas_data[..., subject_index])

        r_OFA_parcels = (slic_image == r_OFA_optimal_label_value)
        l_OFA_parcels = (slic_image == l_OFA_optimal_label_value)
        r_pFus_parcels = (slic_image == r_pFus_optimal_label_value)
        l_pFus_parcels = (slic_image == l_pFus_optimal_label_value)

        #right brain process
        markers = np.zeros_like(image[..., subject_index])
        markers[r_OFA_parcels] = 1
        markers[r_pFus_parcels] = 2
        markers[right_brain_background_marker > 0] = 3
        markers[right_barin_mask == False] = -1

        skeletonize_markers_RW[markers == 1] = 1
        skeletonize_markers_RW[markers == 2] = 3
        skeletonize_markers_RW[markers == 3] = 5

        rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
        rw_labels[rw_labels == -1] = 0

        if (markers == 1).sum() == 0 and (markers == 3).sum() == 0:
            # raise ValueError("atlas_index: " + atlas_index + " r_OFA and r_FFA missing!")
            print "atlas_index: " + str(atlas_index) + " r_OFA and r_FFA missing!"
        elif (markers == 1).sum() == 0:
            region_result_RW[rw_labels == 1] = 3
            region_result_RW[rw_labels == 2] = 5
        elif (markers == 3).sum() == 0:
            region_result_RW[rw_labels == 1] = 1
            region_result_RW[rw_labels == 2] = 5
        else:
            region_result_RW[rw_labels == 1] = 1
            region_result_RW[rw_labels == 2] = 3
            region_result_RW[rw_labels == 3] = 5

        #left brain process
        markers = np.zeros_like(image[..., subject_index])
        markers[l_OFA_parcels] = 1
        markers[l_pFus_parcels] = 2
        markers[left_brain_background_marker > 0] = 3
        markers[left_barin_mask == False] = -1

        skeletonize_markers_RW[markers == 1] = 2
        skeletonize_markers_RW[markers == 2] = 4
        skeletonize_markers_RW[markers == 3] = 5

        rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
        rw_labels[rw_labels == -1] = 0

        if (markers == 2).sum() == 0 and (markers == 4).sum() == 0:
            # raise ValueError("atlas_index: " + atlas_index + " l_OFA and l_FFA missing!")
            print "atlas_index: " + str(atlas_index) + " l_OFA and l_FFA missing!"
        elif (markers == 1).sum() == 0:
            region_result_RW[rw_labels == 1] = 4
            region_result_RW[rw_labels == 2] = 5
        elif (markers == 2).sum() == 0:
            region_result_RW[rw_labels == 1] = 2
            region_result_RW[rw_labels == 2] = 5
        else:
            region_result_RW[rw_labels == 1] = 2
            region_result_RW[rw_labels == 2] = 4
            region_result_RW[rw_labels == 3] = 5

        region_results_RW[..., atlas_index] = region_result_RW
        marker_results_RW[..., atlas_index] = skeletonize_markers_RW

        print 'subject_index:', subject_index, 'atlas_index: ', atlas_index,  'atlas-based rw finished...'

    nib.save(nib.Nifti1Image(marker_results_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                         '_skeletonize_markers_supervoxel.nii.gz')
    nib.save(nib.Nifti1Image(region_results_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                         '_skeletonize_regions_supervoxel.nii.gz')
    nib.save(nib.Nifti1Image(slic_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                  '_supervoxel.nii.gz')
    nib.save(nib.Nifti1Image(supervoxel_top_one_atlas_results_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                                           '_top_atlas.nii.gz')
    nib.save(nib.Nifti1Image(top_atlas_image_RW, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
                                                          '_top_atlas_image.nii.gz')

    region_results_RW[region_results_RW == 5] = 0
    return region_results_RW
    # return marker_results_RW

def atlas_based_aggragator(subject_index):
    region_results_RW = select_optimal_parcel_max_region_mean(subject_index)
    weight = np.ones(DEFAULT_TOP_RANK, dtype=float)
    weighted_result = []

    for roi_index in range(len(ROI) + 1):
        temp = np.zeros_like(region_results_RW)
        temp[region_results_RW == roi_index] = 1

        for i in range(temp.shape[3]):
            temp_data = temp[..., i].copy()
            print 'subject_index: ', subject_index, '   (temp_data == 1).sum(): ', (temp_data == 1).sum()
            if (temp_data == 1).sum() != 0:
                temp[temp_data == 1, i] = (image[temp_data == 1, subject_index] - image[temp_data == 1, subject_index].min()) / \
                                          (image[temp_data == 1, subject_index].max() - image[temp_data == 1, subject_index].min())

        weighted_result.append(np.average(temp, axis=3, weights=weight))

        if roi_index > 0:
            nib.save(nib.Nifti1Image(weighted_result[roi_index], affine), RW_AGGRAGATOR_RESULT_DATA_DIR +
                                                                          ROI[roi_index-1] + '_' + str(subject_index) + '_aggragator.nii.gz')
        else:
            nib.save(nib.Nifti1Image(weighted_result[roi_index], affine), RW_AGGRAGATOR_RESULT_DATA_DIR +
                                                                          'background_' + str(subject_index) + '_aggragator.nii.gz')
        print 'subject_index: ', subject_index, '   roi_index: ', roi_index

    return weighted_result

def generate_rw_prob_result(rw_atlas_based_aggrator_result, subject_sum):
    #generate the prob result
    temp_image = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], subject_sum))
    for subject_index in range(subject_sum):
        for roi_index in range(len(ROI) + 1):
            temp_image[rw_atlas_based_aggrator_result[..., subject_index, roi_index] > 0 , subject_index] = 1

        coords = np.array(np.nonzero(temp_image[..., subject_index] == 1), dtype=np.int32).T
        for i in range(coords.shape[0]):
            temp_image[coords[i, 0], coords[i, 1], coords[i, 2], subject_index] = \
                np.array([rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 0],
                          rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 1],
                          rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 2],
                          rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 3],
                          rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 4]]).argmax()
        print 'generate subject_index: ', subject_index
        # temp_image[temp_image == 5] = 0
    nib.save(nib.Nifti1Image(temp_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + RW_PROB_RESULT_FILE)

    return temp_image


if __name__ == "__main__":
    starttime = datetime.datetime.now()

    img = nib.load(ALL_PROB_MASK)
    mask = img.get_data()
    affine = img.get_affine()

    if not os.path.exists(RW_AGGRAGATOR_RESULT_DATA_DIR):
        os.makedirs(RW_AGGRAGATOR_RESULT_DATA_DIR)

    rw_atlas_based_aggrator_result = np.zeros((image.shape[0], image.shape[1], image.shape[2], SUBJECT_NUM, len(ROI) + 1))
    # for subject_index in range(0, 7): #by sessions
    for subject_index in range(0, SUBJECT_NUM): #by sessions
        # select_optimal_parcel_min_distance(subject_index)
        # select_optimal_parcel_max_region_mean(subject_index)
        weighted_result = atlas_based_aggragator(subject_index)
        for i in range(len(ROI) + 1):
            rw_atlas_based_aggrator_result[..., subject_index, i] = weighted_result[i]

    generate_rw_prob_result(rw_atlas_based_aggrator_result, SUBJECT_NUM)

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."































