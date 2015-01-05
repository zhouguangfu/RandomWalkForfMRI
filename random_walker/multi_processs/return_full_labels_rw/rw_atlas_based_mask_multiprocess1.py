__author__ = 'zhouguangfu'

import datetime
import multiprocessing
import numpy as np
import nibabel as nib

from docx import Document
from skimage.segmentation import random_walker
from configs import *

TOP_RANK = 10 # 0 - 100
SUBJECT_NUM = 70

#global varibale
image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

complete_atlas_data = nib.load(ATLAS_TOP_DIR + 'complete_atlas_label.nii.gz').get_data()
left_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_LEFT_BRAIN_FILE).get_data()
right_barin_mask = nib.load(PROB_ROI_202_SUB_FILE + PROB_RIGHT_BRAIN_FILE).get_data()

def process_single_subject(subject_index):
    global image, complete_atlas_data, left_barin_mask, right_barin_mask, TOP_RANK

    indexs =  np.load(ATLAS_TOP_DIR + str(subject_index) + '_top_sort.npy')
    region_result_RW = np.zeros_like(complete_atlas_data)

    for atlas_index in range(TOP_RANK):
        atlas_data = complete_atlas_data[..., indexs[atlas_index]]

        #process right brain
        markers = np.zeros_like(image[..., subject_index])

        markers[right_barin_mask == 0] = -1
        temp_volume = image[..., subject_index].copy()
        temp_volume[right_barin_mask == 0] = 1
        temp_volume[temp_volume > 0] = 1
        markers[temp_volume <= 0] = 3 #background
        markers[ atlas_data == 1] = 1
        markers[ atlas_data == 3] = 2

        rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
        region_result_RW[rw_labels == 1, atlas_index] = 1
        region_result_RW[rw_labels == 2, atlas_index] = 3
        region_result_RW[rw_labels == 3, atlas_index] = 5

        #process left brain
        markers = np.zeros_like(image[..., subject_index])

        markers[left_barin_mask == 0] = -1
        temp_volume = image[..., subject_index].copy()
        temp_volume[left_barin_mask == 0] = 1
        temp_volume[temp_volume > 0] = 1
        markers[temp_volume <= 0] = 3 #background
        markers[ atlas_data == 2] = 1
        markers[ atlas_data == 4] = 2

        rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf')
        region_result_RW[rw_labels == 1, atlas_index] = 2
        region_result_RW[rw_labels == 2, atlas_index] = 4
        region_result_RW[rw_labels == 3, atlas_index] = 5

        # print 'subject_index:', subject_index, '    atlas_index:', atlas_index
    print 'subject_index:', subject_index, 'atlas-based rw finished...'

    return region_result_RW

if __name__ == "__main__":
    starttime = datetime.datetime.now()

    image_list = []
    for k in range(SUBJECT_NUM):
    # for k in range(1):
        image_list.append(k)

    pool = multiprocessing.Pool(processes=15)
    pool_outputs = pool.map(process_single_subject, image_list)
    pool.close()
    pool.join()

    for j in range(0, SUBJECT_NUM):
    # for j in range(1):
        nib.save(nib.Nifti1Image((pool_outputs[j]).astype(np.int32), affine), RW_RESULT_DATA_DIR + str(j)+ '_'+ RW_ATLAS_BASED_RESULT_FILE)
        print 'Result:  ', j

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
































