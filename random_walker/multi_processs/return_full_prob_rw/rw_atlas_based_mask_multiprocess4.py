__author__ = 'zhouguangfu'

import datetime
import multiprocessing
import numpy as np
import nibabel as nib

from docx import Document
from scipy.ndimage import morphology
from skimage.segmentation import random_walker

TOP_RANK = 30 # 0 - 100
ATLAS_NUM = 202

#global varibale
mask = nib.load(ATLAS_SUBJECTS_LABELS_DIR).get_data()
# image = nib.load(FOUR_D_DATA_DIR + ALL_SESSION_AVEARGE_FILE)
image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()
all_prob_mask = nib.load(ALL_PROB_MASK).get_data()


def process_single_marker(subject_index):
    global mask, image, affine

    all_atlas_based_markers = []
    top_mask = np.zeros_like(mask)
    for i in range(len(ROI)):
        marker_roi = []
        for atlas_index in range(mask.shape[3]):
            z_atlas_mask = np.copy(mask[..., atlas_index] == (i + 1))
            if z_atlas_mask.sum() <= TOP_RANK:
                marker_roi.append(np.zeros((TOP_RANK, 3)))
                continue
            temp_image = image[z_atlas_mask, subject_index]
            threshold = -np.sort(-temp_image)[TOP_RANK]
            z_atlas_mask[image[..., subject_index]<= threshold] = False
            marker = np.array(np.nonzero(z_atlas_mask), dtype=np.int32).T
            marker_roi.append(marker)
            top_mask[z_atlas_mask, atlas_index] = (i + 1)
        all_atlas_based_markers.append(marker_roi)
    nib.save(nib.Nifti1Image(top_mask, affine), RW_RESULT_DATA_DIR + str(subject_index) + '_top_mask.nii.gz')
    print 'subject_index: ', subject_index, ' marker finished...'

    return all_atlas_based_markers

def process_single_subject(subject_image_marker):
    global mask, image, all_prob_mask

    subject_index, all_atlas_based_markers = subject_image_marker
    region_result_RW = np.zeros_like(mask)

    for atlas_index in range(ATLAS_NUM):
        for i in range(len(all_atlas_based_markers)):
            if all_atlas_based_markers[i][atlas_index][:, 0].sum() == 0:
                continue
            region_result_RW[all_atlas_based_markers[i][atlas_index][:, 0],
                             all_atlas_based_markers[i][atlas_index][:, 1],
                             all_atlas_based_markers[i][atlas_index][:, 2], atlas_index] = i + 1

        markers = np.zeros_like(image[..., subject_index])
        volume = morphology.binary_dilation(all_prob_mask).astype(all_prob_mask.dtype)
        # markers[volume == 0] = -1 #not consider -------------!!!!!!
        markers[volume == 0] = -1
        volume[all_prob_mask != 0] = 0
        markers[volume != 0] = 5 #backgroud
        markers[region_result_RW[..., atlas_index] != 0] = region_result_RW[region_result_RW[..., atlas_index] != 0, atlas_index]
        rw_labels = random_walker(image[..., subject_index], markers, beta=10, mode='bf', return_full_prob=True)
        region_result_RW[rw_labels > 0, atlas_index] = rw_labels[rw_labels > 0]
        print 'subject_index:', subject_index, '    atlas_index:', atlas_index

    print 'subject_index:', subject_index, 'atlas-based rw finished...'

    return region_result_RW

if __name__ == "__main__":
    starttime = datetime.datetime.now()

    lines = []
    for line in open(SUBJECT_ID_DIR):
        if line is not '':
            lines.append(line.rstrip('\r\n'))

    document_RW = Document()
    document_RW.add_heading('All prob mask ROIs' + ' RW Analysis', 0)

    image_list = []
    markers_list = []
    for i in range(30, 40):
    # for k in range(1):
        markers_list.append(i)

    markers_pool = multiprocessing.Pool(processes=10)
    markers_pool_outputs = markers_pool.map(process_single_marker, markers_list)
    markers_pool.close()
    markers_pool.join()

    for k in range(30, 40):
    # for k in range(1):
        image_list.append((k, markers_pool_outputs[k - 30]))

    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    # pool_outputs = pool.map(process_single_subject, image_list)
    # pool.close()
    # pool.join()

    # rw_atlas_based_aggrator_result = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3], len(ROI)))

    for j in range(30, 40):
    # for j in range(1):
    #     region_result_RW = pool_outputs[j]

        region_result_RW = process_single_subject(image_list[j - 30])
        weight = np.ones(ATLAS_NUM, dtype=float)
        nib.save(nib.Nifti1Image((region_result_RW).astype(np.int32), affine), RW_RESULT_DATA_DIR + str(j)+ '_'+ RW_ATLAS_BASED_RESULT_FILE)
        print 'Result:  ', j

    #     ROI_sizes = []
    #     for roi_index in range(len(ROI)):
    #         temp = np.zeros_like(region_result_RW)
    #         temp[region_result_RW == (roi_index + 1)] = 1
    #         rw_atlas_based_aggrator_result[..., j, roi_index] = (np.average(temp, aix=3, weights=weight))
    #         nib.save(nib.Nifti1Image(rw_atlas_based_aggrator_result[..., j, roi_index].astype(np.int32), affine),
    #                  RW_RESULT_DATA_DIR + ROI[roi_index] + '_' + str(j)+ '_'+ RW_ATLAS_BASED_AGGRATOR_RESULT_FILE)
    #
    #     print 'j: ', j, '  subject: ', lines[j]
    # for roi_index in range(len(ROI)):
    #     nib.save(nib.Nifti1Image(region_result_RW, affine), RW_RESULT_DATA_DIR + ROI[roi_index] + '_' +RW_ATLAS_BASED_AGGRATOR_RESULT_FILE)

    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."
































