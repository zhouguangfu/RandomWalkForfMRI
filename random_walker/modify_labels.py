__author__ = 'zgf'

import datetime
import multiprocessing

import numpy as np
import nibabel as nib
from configs import *

SUBJECT_NUM = 70
ATLAS_NUM = 202
TOP_RANK = 30

def remove_background(subject_list):
    subject_index, mask, affine = subject_list
    region_result_RW = nib.load(RW_RESULT_DATA_DIR + str(subject_index)+ '_'+ RW_ATLAS_BASED_RESULT_FILE).get_data()
    region_result_RW_back = np.zeros_like(region_result_RW)

    for i in range(ATLAS_NUM):
        labels_bool = [(mask[..., i] == 1).sum() > TOP_RANK, (mask[..., i] == 2).sum() > TOP_RANK,
                       (mask[..., i] == 3).sum() > TOP_RANK, (mask[..., i] == 4).sum() > TOP_RANK]
        cnt = 1
        background_val = np.array(labels_bool).astype(np.int32).sum() + 1
        region_result_RW_back[region_result_RW[..., i] == background_val, i] = 5

        for label_index in range(len(labels_bool)):
            if labels_bool[label_index]:
                region_result_RW_back[region_result_RW[..., i] == cnt, i] = (label_index + 1)
                cnt += 1


    print 'subject_index: ', subject_index
    nib.save(nib.Nifti1Image(region_result_RW_back, affine), RW_RESULT_DATA_DIR + str(subject_index)+ '_'+ RW_ATLAS_BASED_RESULT_FILE)

if __name__ == "__main__":
    starttime = datetime.datetime.now()
    lines = []

    img = nib.load(ATLAS_SUBJECTS_LABELS_DIR)
    mask = img.get_data()
    affine = img.get_affine()

    image_list = []
    for i in range(SUBJECT_NUM):
    # for i in range(1):
        image_list.append((i, mask, affine))

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool_outputs = pool.map(remove_background, image_list)
    pool.close()
    pool.join()

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
































