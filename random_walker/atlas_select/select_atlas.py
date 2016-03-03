__author__ = 'zhouguangfu'

import datetime
import numpy as np
import nibabel as nib
import csv
import multiprocessing
import os

from configs import *
from nipype.interfaces.nipy.utils import Similarity


TOP_RANK = 30 # 0 - 100
ATLAS_NUM = 202

#global varibale
all_202_label_data = nib.load(ATLAS_SUBJECTS_LABELS_DIR).get_data()
all_202_image_data = nib.load(ALL_202_SUBJECTS_DATA_DIR).get_data()
all_202_image_data[all_202_image_data < 0] = 0


image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

all_prob_mask = nib.load(ALL_PROB_MASK).get_data().astype(np.bool)

left_brain_mask = nib.load(PROB_ROI_202_SUB_FILE + 'prob_left_brain.nii.gz').get_data() > 0
right_brain_mask = nib.load(PROB_ROI_202_SUB_FILE + 'prob_right_brain.nii.gz').get_data() > 0

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

    return  res.outputs.similarity

def generate_atlas_top_index_half_brain(subject_index):
    volume1 = image[..., subject_index]
    volume1_filepath = RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) + '_volume1.nii.gz'
    nib.save(nib.Nifti1Image(volume1, affine), volume1_filepath)

    for half_brain_index in range(len(LEFT_RIGHT_BRAIN_NAME)):
        writer = csv.writer(file(ATLAS_TOP_DIR + LEFT_RIGHT_BRAIN_NAME[half_brain_index] + '_' +
                                 str(subject_index) + '_top_sort.csv', 'wb'))
        writer.writerow(['index', 'similarity'])
        mask = left_right_brain_mask_list[half_brain_index]
        mask[mask] = 1

        mask_filepath = RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) + '_mask.nii.gz'
        if not os.path.exists(mask_filepath):
            nib.save(nib.Nifti1Image(mask.astype(int), affine), mask_filepath)

        simility_vals = np.zeros((ATLAS_NUM))
        for j in range(ATLAS_NUM):
            volume2 = all_202_image_data[..., j]
            volume2_filepath = RW_AGGRAGATOR_RESULT_DATA_DIR + str(subject_index) + '_volume2.nii.gz'
            nib.save(nib.Nifti1Image(volume2, affine), volume2_filepath)

            simility_vals[j] = compute_similarity(volume1_filepath, volume2_filepath, mask_filepath)
            os.remove(volume2_filepath)

        os.remove(mask_filepath)
        print '------------------------------ i: ', subject_index, ' -----------------------------'

        index = np.argsort(-simility_vals)
        np.save(ATLAS_TOP_DIR + LEFT_RIGHT_BRAIN_NAME[half_brain_index] + '_' + str(subject_index) + '_top_sort.npy', index)

        left_right_brain_label_data = np.zeros((left_brain_mask.shape[0],
                                                left_brain_mask.shape[1],
                                                left_brain_mask.shape[2],
                                                TOP_RANK))
        for top_index in range(TOP_RANK):
            left_right_brain_label_data[..., top_index] = all_202_label_data[..., index[top_index]]

        nib.save(nib.Nifti1Image(left_right_brain_label_data, affine), ATLAS_TOP_DIR +
                 LEFT_RIGHT_BRAIN_NAME[half_brain_index] + '_' + str(subject_index) + '_top_rank_' + str(TOP_RANK) + '.nii.gz')

        for k in range(index.shape[0]):
            writer.writerow([index[k], round(simility_vals[index[k]], 4)])

    os.remove(volume1_filepath)


if __name__ == "__main__":
    starttime = datetime.datetime.now()
    # generate_atlas_top_index_per_roi(all_202_image_data)

    # for i in range(image.shape[3]):
    #     generate_atlas_top_index_half_brain(i)

    # for i in range(70):
    #     generate_atlas_top_index_half_brain(i)

    process_num = 14
    for cycle_index in range(image.shape[3] / process_num):
        pool = multiprocessing.Pool(processes=process_num)
        pool_outputs = pool.map(generate_atlas_top_index_half_brain, range(cycle_index * process_num,
                                                              (cycle_index + 1) * process_num))
        pool.close()
        pool.join()

        print 'Cycle index: ', cycle_index, 'Time cost: ', (datetime.datetime.now() - starttime)


    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
































