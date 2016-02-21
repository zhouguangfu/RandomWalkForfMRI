__author__ = 'zgf'

import numpy as np
import nibabel as nib
from configs import *

SESSION_NUM = 7

if __name__ == "__main__":
    all_experts_dice_array = np.zeros((len(SUBJECT_NAMES), len(ROI)))
    all_rw_dice_array = np.zeros((len(SUBJECT_NAMES), len(ROI)))

    for i in range(len(SUBJECT_NAMES)):
        experts_dice_path = ANALYSIS_DIR + 'staple/' + 'experts' + '/' + str(i) + '_' + SUBJECT_NAMES[i] + '/dice.txt'
        rw_dice_path = ANALYSIS_DIR + 'staple/' + 'zgf' + '/' + str(i) + '_' + SUBJECT_NAMES[i] + '/rw/dice.txt'

        experts_dice = [float(line.split(' ')[8]) for line in open(experts_dice_path)]
        rw_dice = [float(line.split(' ')[8]) for line in open(rw_dice_path)]
        experts_dice_array = np.array(experts_dice).reshape(SESSION_NUM * 2, len(ROI))
        rw_dice_array = np.array(rw_dice).reshape(SESSION_NUM, len(ROI))

        all_experts_dice_array[i, :] = np.mean(experts_dice_array, axis=0)
        all_rw_dice_array[i, :] = np.mean(rw_dice_array, axis=0)


        # print experts_dice
        print experts_dice_path
        print experts_dice
        print np.round(np.mean(experts_dice_array, axis=0), 3)
        print np.round(np.mean(rw_dice_array, axis=0), 3)
        print '---------------------------------------', i ,'--------------------------------------------'
    print '-----------------------all-------------------------'
    print np.round(np.mean(all_experts_dice_array, axis=0), 3)
    print np.round(np.mean(rw_dice_array, axis=0), 3)
    print 'experts: ', np.round(np.std(all_experts_dice_array, axis=0), 3) / np.round(np.mean(all_experts_dice_array, axis=0), 3)
    print 'rw: ', np.round(np.std(rw_dice_array, axis=0), 3) / np.round(np.mean(rw_dice_array, axis=0), 3)

























