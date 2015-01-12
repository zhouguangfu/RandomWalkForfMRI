__author__ = 'zhouguangfu'

import datetime
import numpy as np
import nibabel as nib
import csv

from configs import *

TOP_RANK = 30 # 0 - 100
ATLAS_NUM = 202

#global varibale
mask = nib.load(ATLAS_SUBJECTS_LABELS_DIR).get_data()

all_202_image_data = nib.load(ALL_202_SUBJECTS_DATA_DIR).get_data()

image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

all_prob_mask = nib.load(ALL_PROB_MASK).get_data().astype(np.bool)

def generate_complete_atlas_label():
    complete_atlas_list = []
    complete_image_list = []

    for i in range(ATLAS_NUM):
        if (mask[..., i] == 1).sum() == 0 or \
                        (mask[..., i] == 2).sum() == 0 or \
                        (mask[..., i] == 3).sum() == 0 or \
                        (mask[..., i] == 4).sum() == 0:
            pass
        else:
            complete_atlas_list.append(mask[..., i])
            complete_image_list.append(all_202_image_data[..., i])

    complete_atlas_data = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], len(complete_atlas_list)))
    complete_image_data = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], len(complete_atlas_list)))
    for j in range(len(complete_atlas_list)):
        complete_atlas_data[..., j] = complete_atlas_list[j]
        complete_image_data[..., j] = complete_image_list[j]

    nib.save(nib.Nifti1Image(complete_atlas_data, affine), ATLAS_TOP_DIR +  'complete_atlas_label.nii.gz')
    nib.save(nib.Nifti1Image(complete_image_data, affine), ATLAS_TOP_DIR +  'complete_image_data.nii.gz')
    return complete_image_data

def generate_atlas_top_index(all_image_data):
    for i in range(image.shape[3]):
        writer = csv.writer(file(ATLAS_TOP_DIR + str(i) + '_top_sort.csv', 'wb'))
        writer.writerow(['index', 'similarity'])

        vector1 = image[all_prob_mask, i]
        simility_vals = np.zeros((all_image_data.shape[3]))
        for j in range(all_image_data.shape[3]):
            vector2 = all_image_data[all_prob_mask, j]
            simility_vals[j] = np.corrcoef(vector1, vector2)[0, 1]
            print 'i: ', i, '   j: ', j

        index = np.argsort(-simility_vals)
        np.save(ATLAS_TOP_DIR + str(i) + '_top_sort.npy', index)

        for k in range(index.shape[0]):
            writer.writerow([index[k], round(simility_vals[index[k]], 4)])

        print simility_vals

if __name__ == "__main__":
    starttime = datetime.datetime.now()

    lines = []
    for line in open(SUBJECT_ID_DIR):
        if line is not '':
            lines.append(line.rstrip('\r\n'))

    all_image_data = generate_complete_atlas_label()
    all_image_data[all_image_data < 0] = 0
    generate_atlas_top_index(all_image_data)
    # all_202_image_data[all_202_image_data < 0] = 0
    # generate_atlas_top_index(all_202_image_data)

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
































