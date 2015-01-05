__author__ = 'zgf'

import datetime
import numpy as np
import nibabel as nib
from configs import *

SUBJECT_NUM = 4
ATLAS_NUM = 202
BACKGROUND_THR = np.arange(0.0, 2.3, 0.2)

if __name__ == "__main__":
    starttime = datetime.datetime.now()
    lines = []

    left_brain_mask = nib.load('/nfs/j3/userhome/zhouguangfu/workingdir/nfs/4Ddata/roi_prob/prob_left_brain.nii.gz').get_data()
    right_brain_mask = nib.load('/nfs/j3/userhome/zhouguangfu/workingdir/nfs/4Ddata/roi_prob/prob_right_brain.nii.gz').get_data()

    image = nib.load(ACTIVATION_DATA_DIR).get_data()

    img = nib.load(ALL_PROB_MASK)
    mask = img.get_data()
    affine = img.get_affine()

    vals = []

                # elif self.opt_type == 'OTSU':
                # for i in range(len(region)):
                #     label_last = region[i][len(region[0]) - 1].get_label()
                #     neighbour_last = region[i][len(region[0]) - 1].get_neighbor()
                #     region_mask = np.array(list(label_last) + list(neighbour_last))
                #     u = np.mean(image[region_mask[:, 0], region_mask[:, 1], region_mask[:, 2]])
                #     for j in range(len(region[0])):
                #         label = region[i][j].get_label()
                #         neighbor = region[i][j].get_neighbor()
                #         region_val = np.mean(image[label[:, 0], label[:, 1], label[:, 2]])
                #         per_val = np.mean(image[neighbor[:, 0], neighbor[:, 1], neighbor[:, 2]])
                #
                #         # Otsu method
                #         w0 = label.shape[0] * 1. / region_mask.shape[0]
                #         w1 = 1. - w0
                #         u0 = region_val
                #         u1 = (u - w0 * u0) / w1
                #         con_val[i, j] = w0 * w1 * (u0 - u1) * (u0 - u1)

    for thr in BACKGROUND_THR:
        rw_atlas_based_aggrator_result = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], SUBJECT_NUM, len(ROI) + 1), dtype=np.float)
        for roi_index in range(len(ROI) + 1):
            if roi_index == len(ROI):
                rw_atlas_based_aggrator_result[..., roi_index] = nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + str(thr) + '_' + RW_PROB_BACKGROUND_RESULT_FILE).get_data()
            else:
                rw_atlas_based_aggrator_result[..., roi_index] = nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + ROI[roi_index] + '_' + str(thr) + '_' +RW_ATLAS_BASED_AGGRATOR_RESULT_FILE).get_data()

        temp_image = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], SUBJECT_NUM))
        for subject_index in range(SUBJECT_NUM):
            for roi_index in range(len(ROI) + 1):
                temp_image[rw_atlas_based_aggrator_result[..., subject_index, roi_index] > 0 , subject_index] = 1

            coords = np.array(np.nonzero(temp_image[..., subject_index] == 1), dtype=np.int32).T
            for i in range(coords.shape[0]):
                temp_image[coords[i, 0], coords[i, 1], coords[i, 2], subject_index] = np.array([rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 0],
                                                                                rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 1],
                                                                                rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 2],
                                                                                rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 3],
                                                                                rw_atlas_based_aggrator_result[coords[i, 0], coords[i, 1], coords[i, 2], subject_index, 4]]).argmax() + 1

            left_u = image[left_brain_mask > 0, subject_index].mean()
            left_w0 = ((temp_image[..., subject_index] == 1).sum() + (temp_image[..., subject_index] == 3).sum()) * 1. \
                      / (temp_image[left_brain_mask > 0, subject_index] == 5).sum()
            left_w1 = 1.0 - left_w0
            left_label_mask = np.logical_or(temp_image[..., subject_index == 1], temp_image[..., subject_index] == 3)
            left_u0 = image[left_label_mask, subject_index].mean()
            left_u1 = (left_u - left_w0 * left_u0) / left_w1
            left_val = left_w0 * left_w1 * (left_u0 - left_u1) * (left_u0 - left_u1)

            right_u = image[right_brain_mask > 0, subject_index].mean()
            right_w0 = ((temp_image[..., subject_index] == 1).sum() + (temp_image[..., subject_index] == 3).sum()) *1. \
                      / (temp_image[right_brain_mask > 0, subject_index] == 5).sum()
            right_w1 = 1.0 - right_w0
            right_label_mask = np.logical_or(temp_image[..., subject_index == 1], temp_image[..., subject_index] == 3)
            right_u0 = image[right_label_mask, subject_index].mean()
            right_u1 = (right_u - right_w0 * right_u0) / right_w1
            right_val = right_w0 * right_w1 * (right_u0 - right_u1) * (right_u0 - right_u1)

            vals.append([left_val, right_val])

            print 'thr: ', thr, '   subject_index: ', subject_index, '  left_u: ', left_u, '  right_u: ', right_u, \
                '   left_val: ', left_val, '  right_val: ', right_val

        temp_image[temp_image == 5] = 0


        nib.save(nib.Nifti1Image(temp_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + str(thr) + '_' +RW_PROB_RESULT_FILE)

    print '-------------------------------------------------------------------------------------------'
    print 'vals :', vals

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
































