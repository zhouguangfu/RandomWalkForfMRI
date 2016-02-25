__author__ = 'zgf'

import datetime

import nibabel as nib
import numpy as np
from docx import Document
from docx.shared import Inches

from configs import *
from matplot_case.inter_subject_bar import show_barchart

SUBJECT_SESSION_NUM = 7

def dice(volume1, volume2):
    if volume1.shape != volume2.shape:
        raise ValueError("Shape mismatch: volume1 and volume2 must have the same shape.")
    intersection = np.logical_and(volume1, volume2)
    if volume1.sum() + volume2.sum() == 0:
        return 0.0
    else:
        return 2. * intersection.sum() / (volume1.sum() + volume2.sum())

    return dices


def compute_dice():
    manual_all_subject_session = nib.load(ANALYSIS_DIR + 'manual/' + 'manual.nii.gz').get_data()
    # rw_all_subject_session = nib.load(ANALYSIS_DIR + 'rw_prob_result_file.nii.gz').get_data()

    all_rois_dice_mean = []
    all_rois_dice_std= []
    for roi_index in range(len(ROI)):
        roi_dice = np.zeros((manual_all_subject_session.shape[3], ))
        # data = rw_all_subject_session[..., subject_index]

        #ARG
        data = nib.load(ANALYSIS_DIR + 'asrg/' + ROI[roi_index] + '_' + AC_OPTIMAL_FILE).get_data()

        #GSS
        # data = nib.load(ANALYSIS_DIR + 'gss/' + ROI[roi_index] + '_gss.nii.gz').get_data()

        for subject_index in range(manual_all_subject_session.shape[3]):
            dice_value = dice(data[..., subject_index] == 1, manual_all_subject_session[..., subject_index] == (roi_index + 1))

            print 'subject_index: ', subject_index, '  roi_index: ', roi_index
            print dice_value
            roi_dice[subject_index] = dice_value

        print '------------------- ', subject_index, ' ----------------------'
        all_rois_dice_mean.append(roi_dice.mean())
        all_rois_dice_std.append(roi_dice.std())

    return all_rois_dice_mean, all_rois_dice_std


if __name__ == "__main__":
    starttime = datetime.datetime.now()

    all_rois_dice_mean, all_rois_dice_std = compute_dice()
    print 'all_rois_dice_mean: ', all_rois_dice_mean
    print 'all_rois_dice_std: ', all_rois_dice_std

    endtime = datetime.datetime.now()
    print 'Time costs: ', (endtime - starttime)
    print "Program end..."


#Compare to the manual segmentation...

#RW method dice :
# all_rois_dice_mean:  [0.36026110427318925, 0.28633489339739487, 0.54198482563799788, 0.50848599454180332]
# all_rois_dice_std:  [0.26296877028703725, 0.2833425607203679, 0.23079355897693135, 0.24334520330330828]

#GSS method dice :
# all_rois_dice_mean:  [0.57651501446295739, 0.4054730454791749, 0.76699057512641489, 0.62777441364131081]
# all_rois_dice_std:  [0.25508285230296185, 0.33672038815818417, 0.24079069221934857, 0.33759774999614256]

#ARG method dice :
# all_rois_dice_mean:  [0.42643644385691837, 0.45345652892851684, 0.43652831976571843, 0.48440739932965449]
# all_rois_dice_std:  [0.29883630292637126, 0.37621551207848314, 0.26660482476701081, 0.27638676043808591]
