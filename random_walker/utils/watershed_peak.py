__author__ = 'zgf'

import nibabel as nib
from sklearn.cluster import KMeans

from configs import *
from imtool import *

DEFAULT_TOP_RANK = 10
WATERSHED_THRESHOLD = 0
SUBJECT_NUM = 14

#global varibale
image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

import datetime
import os

all_202_image_data = nib.load(ALL_202_SUBJECTS_DATA_DIR).get_data()

complete_atlas_data = nib.load(ATLAS_TOP_DIR + 'complete_atlas_label.nii.gz').get_data()
left_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_LEFT_BRAIN_FILE).get_data()
right_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_RIGHT_BRAIN_FILE).get_data()

r_OFA_mask = nib.load(PROB_ROI_202_SUB_FILE + 'r_OFA_prob.nii.gz').get_data() > 0
l_OFA_mask = nib.load(PROB_ROI_202_SUB_FILE + 'l_OFA_prob.nii.gz').get_data() > 0
r_pFus_mask = nib.load(PROB_ROI_202_SUB_FILE + 'r_pFus_prob.nii.gz').get_data() > 0
l_pFus_mask = nib.load(PROB_ROI_202_SUB_FILE + 'l_pFus_prob.nii.gz').get_data() > 0

r_OFA_label_mask = nib.load(LABEL_ROI_202_SUB_FILE + 'r_OFA_label.nii.gz').get_data() > 0
l_OFA_label_mask = nib.load(LABEL_ROI_202_SUB_FILE + 'l_OFA_label.nii.gz').get_data() > 0
r_pFus_label__mask = nib.load(LABEL_ROI_202_SUB_FILE + 'r_pFus_label.nii.gz').get_data() > 0
l_pFus_label_mask = nib.load(LABEL_ROI_202_SUB_FILE + 'l_pFus_label.nii.gz').get_data() > 0


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
    # print "Peaks of atlas label: ", peaks
    return peaks

def compute_parcel_peak(subject_index, mask):
    localmax_cords = local_maximum(image[..., subject_index], 2)
    # nib.save(nib.Nifti1Image(localmax_cords, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) +
    #                                               '_watershed_localmax.nii.gz')
    localmax_cords[mask == 0] = 0
    return np.array(np.nonzero(localmax_cords)).astype(np.float).T


def compute_atlas_rois_center_id():
    #Compute ROI mean x, y, z cordinates
    rois_center_cords = []

    for roi_index in range(len(ROI)):
        result = np.zeros((complete_atlas_data.shape[0], complete_atlas_data.shape[1], complete_atlas_data.shape[2]))
        for i in range(complete_atlas_data.shape[3]):
            mask = complete_atlas_data[..., i]
            temp = np.zeros_like(result)
            temp[mask == (roi_index + 1)] = all_202_image_data[mask == (roi_index + 1), i]
            peak_cord = np.unravel_index(temp.argmax(), temp.shape)

            result[peak_cord[0], peak_cord[1], peak_cord[2]] = 1

        cords = np.asarray(np.nonzero(result))
        x_mean, ymean, z_mean = cords[0].mean(), cords[1].mean(), cords[2].mean()
        print ROI[roi_index], '=> x_mean, ymean, z_mean : ',x_mean, ymean, z_mean
        rois_center_cords.append((x_mean, ymean, z_mean))

    return rois_center_cords

# r_OFA => x_mean, ymean, z_mean :  23.5056818182 23.8977272727 29.4488636364
# l_OFA => x_mean, ymean, z_mean :  64.9556962025 22.6708860759 29.5
# r_pFus => x_mean, ymean, z_mean :  23.8323353293 36.5988023952 25.874251497
# l_pFus => x_mean, ymean, z_mean :  64.7070063694 35.9044585987 25.8662420382


if __name__ == "__main__":
    starttime = datetime.datetime.now()

    if not os.path.exists(RW_AGGRAGATOR_RESULT_DATA_DIR):
        os.makedirs(RW_AGGRAGATOR_RESULT_DATA_DIR)

    rois_center_cords = np.asarray(compute_atlas_rois_center_id())
    peak_point_result = np.zeros_like(image).astype(np.int)
    for subject_index in range(image.shape[3]):
        #Left brain
        localmax_cords = compute_parcel_peak(subject_index, left_barin_mask)
        print localmax_cords.shape
        print np.array([rois_center_cords[0], rois_center_cords[2]])
        k_means = KMeans(n_clusters=2,init=np.array([rois_center_cords[0], rois_center_cords[2]]), n_init=1).fit(localmax_cords)
        # break
        labels = k_means.labels_

        temp = np.zeros_like(image[..., subject_index]).astype(np.int)
        for i in range(labels.shape[0]):
            if labels[i] == 0:
                temp[localmax_cords[i, 0], localmax_cords[i, 1], localmax_cords[i, 2]] = 1
            elif labels[i] == 1:
                temp[localmax_cords[i, 0], localmax_cords[i, 1], localmax_cords[i, 2]] = 2
            else:
                print 'error!'
        data = np.zeros_like(temp)
        data[temp == 1] = image[temp == 1, subject_index]
        r_OFA_peak_points = np.unravel_index(data.argmax(), data.shape)

        data = np.zeros_like(temp)
        data[temp == 2] = image[temp == 2, subject_index]
        r_FFA_peak_points = np.unravel_index(data.argmax(), data.shape)

        peak_point_result[r_OFA_peak_points[0], r_OFA_peak_points[1], r_OFA_peak_points[2], subject_index] = 2
        peak_point_result[r_FFA_peak_points[0], r_FFA_peak_points[1], r_FFA_peak_points[2], subject_index] = 4

        #Right brain
        localmax_cords = compute_parcel_peak(subject_index, right_barin_mask)
        k_means = KMeans(n_clusters=2,init=np.array([rois_center_cords[1], rois_center_cords[3]]), n_init=1).fit(localmax_cords)
        # break
        labels = k_means.labels_

        temp = np.zeros_like(image[..., subject_index]).astype(np.int)
        for i in range(labels.shape[0]):
            if labels[i] == 0:
                temp[localmax_cords[i, 0], localmax_cords[i, 1], localmax_cords[i, 2]] = 1
            elif labels[i] == 1:
                temp[localmax_cords[i, 0], localmax_cords[i, 1], localmax_cords[i, 2]] = 2
            else:
                print 'error!'
        data = np.zeros_like(temp)
        data[temp == 1] = image[temp == 1, subject_index]
        r_OFA_peak_points = np.unravel_index(data.argmax(), data.shape)

        data = np.zeros_like(temp)
        data[temp == 2] = image[temp == 2, subject_index]
        r_FFA_peak_points = np.unravel_index(data.argmax(), data.shape)

        peak_point_result[r_OFA_peak_points[0], r_OFA_peak_points[1], r_OFA_peak_points[2], subject_index] = 1
        peak_point_result[r_FFA_peak_points[0], r_FFA_peak_points[1], r_FFA_peak_points[2], subject_index] = 3

        #Draw sphere.
        center_data = np.zeros_like(temp)
        center_data = peak_point_result[..., subject_index]
        coord_list, value_list = nonzero_coord(center_data)
        data = center_data.copy()
        for idx in range(len(coord_list)):
            data = sphere_roi(data,
                              coord_list[idx][0],
                              coord_list[idx][1],
                              coord_list[idx][2],
                              [3, 3, 3],
                              value_list[idx])
        peak_point_result[..., subject_index] = data

        print 'Peak point, subject_index => ', subject_index

    nib.save(nib.Nifti1Image(peak_point_result, affine), RW_AGGRAGATOR_RESULT_DATA_DIR  +  'a_peak_points_result.nii.gz')
    print RW_AGGRAGATOR_RESULT_DATA_DIR  +  'a_peak_points_result.nii.gz'


    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
