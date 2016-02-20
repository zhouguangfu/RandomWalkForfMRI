__author__ = 'zhouguangfu'
'''
Detect the size of the label which is less than 10 in the Top Atlas and print the correspongding atlas index.
'''

import datetime
import numpy as np
import nibabel as nib

from configs import *

#global varibale
mask = nib.load(ATLAS_SUBJECTS_LABELS_DIR).get_data()

image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

all_prob_mask = nib.load(ALL_PROB_MASK).get_data().astype(np.bool)
complete_atlas_data = nib.load(ATLAS_TOP_DIR + 'complete_atlas_label.nii.gz').get_data()

def process_single_subject(subject_index):
    global image, complete_atlas_data, left_barin_mask, right_barin_mask

    indexs =  np.load(ATLAS_TOP_DIR + "\old_threshold_0/" + str(subject_index) + '_top_sort.npy')
    region_result_RW = np.zeros((complete_atlas_data.shape[0], complete_atlas_data.shape[1], complete_atlas_data.shape[2], TOP_RANK))

    for atlas_index in range(TOP_RANK):
        atlas_data = complete_atlas_data[..., indexs[atlas_index]]
        if (atlas_data == 1).sum() <= 10 or \
                        (atlas_data == 2).sum() <= 10 or \
                        (atlas_data == 3).sum() <= 10 or \
                        (atlas_data == 4).sum() <= 10:
                print 'subject_index: ', subject_index,  '  atlas_index: ', atlas_index
    print '--------------------------------', subject_index, '--------------------------------'

if __name__ == "__main__":
    starttime = datetime.datetime.now()


    for subject_index in range(image.shape[3]):
        process_single_subject(subject_index)


    endtime = datetime.datetime.now()
    print "Detect all the labels's time cost: ", (endtime - starttime)
    print "Program end..."
































