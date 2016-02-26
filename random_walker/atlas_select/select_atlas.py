__author__ = 'zhouguangfu'

import datetime
import numpy as np
import nibabel as nib
import csv

from configs import *
from nipype.interfaces.nipy.utils import Similarity

TOP_RANK = 30 # 0 - 100
ATLAS_NUM = 202
#Compute the similarity threshold.
IMAGE_THRESHOLD = 0

#global varibale
all_202_label_data = nib.load(ATLAS_SUBJECTS_LABELS_DIR).get_data()
all_202_image_data = nib.load(ALL_202_SUBJECTS_DATA_DIR).get_data()

image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

all_prob_mask = nib.load(ALL_PROB_MASK).get_data().astype(np.bool)

r_OFA_mask = nib.load(PROB_ROI_202_SUB_FILE + 'r_OFA_prob.nii.gz').get_data() > 0
l_OFA_mask = nib.load(PROB_ROI_202_SUB_FILE + 'l_OFA_prob.nii.gz').get_data() > 0
r_pFus_mask = nib.load(PROB_ROI_202_SUB_FILE + 'r_pFus_prob.nii.gz').get_data() > 0
l_pFus_mask = nib.load(PROB_ROI_202_SUB_FILE + 'l_pFus_prob.nii.gz').get_data() > 0

left_brain_mask = nib.load(PROB_ROI_202_SUB_FILE + 'prob_left_brain.nii.gz').get_data() > 0
right_brain_mask = nib.load(PROB_ROI_202_SUB_FILE + 'prob_right_brain.nii.gz').get_data() > 0

roi_mask_list = [r_OFA_mask, l_OFA_mask, r_pFus_mask, l_pFus_mask]
left_right_brain_mask_list = [left_brain_mask, right_brain_mask]
LEFT_RIGHT_BRAIN_NAME = ['left_brain', 'right_brain']


def compute_similarity(volume1_filepath, volume2_filepath, mask_file_path, metric='nmi'):
    similarity = Similarity()
    similarity.inputs.volume1 = volume1_filepath
    similarity.inputs.volume2 = volume2_filepath
    similarity.inputs.mask1 = mask_file_path
    similarity.inputs.mask2 = mask_file_path
    similarity.inputs.metric = metric
    res = similarity.run()

    print 'Similarity: ', res

def generate_atlas_top_index_per_roi(all_image_data):
    for i in range(image.shape[3]):
        for roi_index in range(len(ROI)):
            writer = csv.writer(file(ATLAS_TOP_DIR + ROI[roi_index] + '_' + str(i) + '_top_sort.csv', 'wb'))
            writer.writerow(['index', 'similarity'])

            # vector1 = image[roi_mask_list[roi_index], i] > IMAGE_THRESHOLD
            simility_vals = np.zeros((all_image_data.shape[3]))
            for j in range(all_image_data.shape[3]):
                # vector2 = all_image_data[roi_mask_list[roi_index], j] > IMAGE_THRESHOLD
                # inter_mask = np.logical_or(vector1, vector2)
                #
                # print 'inter_mask.sum(): ', inter_mask.sum()
                # simility_vals[j] = 0.5 + 0.5 * np.corrcoef(vector1[inter_mask], vector2[inter_mask])[0, 1]

                volume1 = image[..., i]
                volume2 = all_image_data[..., j]
                mask = roi_mask_list[roi_index]
                mask[mask] = 1
                nib.save(nib.Nifti1Image(volume1, affine), RW_AGGRAGATOR_RESULT_DATA_DIR +'volume1.nii.gz')
                nib.save(nib.Nifti1Image(volume2, affine), RW_AGGRAGATOR_RESULT_DATA_DIR +'volume2.nii.gz')
                nib.save(nib.Nifti1Image(mask.astype(int), affine), RW_AGGRAGATOR_RESULT_DATA_DIR +'mask.nii.gz')

                simility_vals[j] = compute_similarity(RW_AGGRAGATOR_RESULT_DATA_DIR +'volume1.nii.gz',
                                                      RW_AGGRAGATOR_RESULT_DATA_DIR +'volume2.nii.gz',
                                                      RW_AGGRAGATOR_RESULT_DATA_DIR +'mask.nii.gz')

                print 'i: ', i, '   roi_index: ', roi_index, ' j: ', j, '   simility_vals[j]: ',  simility_vals[j]

            index = np.argsort(-simility_vals)
            np.save(ATLAS_TOP_DIR + ROI[roi_index] + '_' + str(i) + '_top_sort.npy', index)

            per_roi_label_data = np.zeros((r_OFA_mask.shape[0], r_OFA_mask.shape[1], r_OFA_mask.shape[2], TOP_RANK))
            for top_index in range(TOP_RANK):
                per_roi_label_data[..., top_index] = all_202_label_data[..., index[top_index]]
                if (per_roi_label_data[..., top_index] == (roi_index + 1)).sum() <= 0:
                    print 'subject_index: ', i, '----roi_index: ', roi_index, '-- ',  ' top_index: ', top_index, '----------'

            nib.save(nib.Nifti1Image(per_roi_label_data, affine), ATLAS_TOP_DIR + ROI[roi_index] + '_' + str(i) +
                                                            '_top_rank_' + str(TOP_RANK) + '.nii.gz')

            for k in range(index.shape[0]):
                writer.writerow([index[k], round(simility_vals[index[k]], 4)])



def generate_atlas_top_index_half_brain(all_image_data):
    for i in range(image.shape[3]):
        for half_brain_index in range(len(LEFT_RIGHT_BRAIN_NAME)):
            writer = csv.writer(file(ATLAS_TOP_DIR + LEFT_RIGHT_BRAIN_NAME[half_brain_index] + '_' + str(i) + '_top_sort.csv', 'wb'))
            writer.writerow(['index', 'similarity'])

            vector1 = image[left_right_brain_mask_list[half_brain_index], i] > IMAGE_THRESHOLD
            simility_vals = np.zeros((all_image_data.shape[3]))
            for j in range(all_image_data.shape[3]):
                vector2 = all_image_data[left_right_brain_mask_list[half_brain_index], j] > IMAGE_THRESHOLD
                inter_mask = np.logical_or(vector1, vector2)

                print 'inter_mask.sum(): ', inter_mask.sum()
                simility_vals[j] = 0.5 + 0.5 * np.corrcoef(vector1[inter_mask], vector2[inter_mask])[0, 1]
                print 'i: ', i, '   half_brain_index: ', half_brain_index, ' j: ', j, '  simility_vals[j]: ', simility_vals[j]

            index = np.argsort(-simility_vals)
            np.save(ATLAS_TOP_DIR + LEFT_RIGHT_BRAIN_NAME[half_brain_index] + '_' + str(i) + '_top_sort.npy', index)

            left_right_brain_label_data = np.zeros((left_brain_mask.shape[0], left_brain_mask.shape[1], left_brain_mask.shape[2], TOP_RANK))
            for top_index in range(TOP_RANK):
                left_right_brain_label_data[..., top_index] = all_202_label_data[..., index[top_index]]
                if (left_right_brain_label_data[..., top_index] == (half_brain_index + 1)).sum() <= 0:
                    print 'subject_index: ', i, '----roi_index: ', half_brain_index, '-- ',  ' top_index: ', top_index, '----------'

            nib.save(nib.Nifti1Image(left_right_brain_label_data, affine), ATLAS_TOP_DIR + LEFT_RIGHT_BRAIN_NAME[half_brain_index] + '_' + str(i) +
                                                                  '_top_rank_' + str(TOP_RANK) + '.nii.gz')

            for k in range(index.shape[0]):
                writer.writerow([index[k], round(simility_vals[index[k]], 4)])

if __name__ == "__main__":
    starttime = datetime.datetime.now()

    all_202_image_data[all_202_image_data < 0] = 0
    generate_atlas_top_index_per_roi(all_202_image_data)
    # all_202_image_data[all_202_image_data < 0] = 0
    # generate_atlas_top_index_half_brain(all_202_image_data)

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
































