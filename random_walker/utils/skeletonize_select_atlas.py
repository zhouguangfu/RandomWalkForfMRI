__author__ = 'zhouguangfu'

import datetime
import numpy as np
import nibabel as nib
import csv

from configs import *

TOP_RANK = 30 # 0 - 100
ATLAS_NUM = 202

#global varibale
all_202_label_data = nib.load(ATLAS_SUBJECTS_LABELS_DIR).get_data()

# all_202_image_data = nib.load(ALL_202_SUBJECTS_DATA_DIR).get_data()
all_202_image_data = nib.load(ATLAS_TOP_DIR + 'all_202_image_skeletonize.nii.gz').get_data()

image = nib.load(ATLAS_TOP_DIR + 'all_sessions_skeletonize.nii.gz')
# image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

all_prob_mask = nib.load(ALL_PROB_MASK).get_data().astype(np.bool)

r_OFA_mask = nib.load(PROB_ROI_202_SUB_FILE + 'r_OFA_prob.nii.gz').get_data() > 0
l_OFA_mask = nib.load(PROB_ROI_202_SUB_FILE + 'l_OFA_prob.nii.gz').get_data() > 0
r_pFus_mask = nib.load(PROB_ROI_202_SUB_FILE + 'r_pFus_prob.nii.gz').get_data() > 0
l_pFus_mask = nib.load(PROB_ROI_202_SUB_FILE + 'l_pFus_prob.nii.gz').get_data() > 0

roi_mask_list = [r_OFA_mask, l_OFA_mask, r_pFus_mask, l_pFus_mask]

def generate_atlas_top_index(all_image_data):
    for i in range(image.shape[3]):
        for roi_index in range(len(ROI)):
            writer = csv.writer(file(ATLAS_TOP_DIR + ROI[roi_index] + '_' + str(i) + '_top_sort.csv', 'wb'))
            writer.writerow(['index', 'similarity'])

            vector1 = image[roi_mask_list[roi_index], i]
            simility_vals = np.zeros((all_image_data.shape[3]))
            for j in range(all_image_data.shape[3]):
                vector2 = all_image_data[roi_mask_list[roi_index], j]
                simility_vals[j] = np.corrcoef(vector1, vector2)[0, 1]
                # print 'i: ', i, '   roi_index: ', roi_index, ' j: ', j

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

if __name__ == "__main__":
    starttime = datetime.datetime.now()

    all_202_image_data[all_202_image_data < 0] = 0
    generate_atlas_top_index(all_202_image_data)
    # all_202_image_data[all_202_image_data < 0] = 0
    # generate_atlas_top_index(all_202_image_data)

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
































