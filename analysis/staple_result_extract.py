__author__ = 'zgf'

import numpy as np
import nibabel as nib
from configs import *

SESSION_NUM = 7

if __name__ == "__main__":
    all_maunal_dice_array = np.zeros((len(SUBJECT_NAMES), len(ROI)))
    all_rw_dice_array = np.zeros((len(SUBJECT_NAMES), len(ROI)))

    expert_name = 'lzg' # or 'lzg'

    for i in range(len(SUBJECT_NAMES)):
        manual_dice_path = ANALYSIS_DIR + 'staple/' + expert_name + '/' + str(i) + '_' + SUBJECT_NAMES[i] + '/manual/dice.txt'
        rw_dice_path = ANALYSIS_DIR + 'staple/' + expert_name + '/' + str(i) + '_' + SUBJECT_NAMES[i] + '/rw/dice.txt'

        manual_dice = [float(line.split(' ')[8]) for line in open(manual_dice_path)]
        rw_dice = [float(line.split(' ')[8]) for line in open(rw_dice_path)]
        manual_dice_array = np.array(manual_dice).reshape(SESSION_NUM, len(ROI))
        rw_dice_array = np.array(rw_dice).reshape(SESSION_NUM, len(ROI))

        all_maunal_dice_array[i, :] = np.mean(manual_dice_array, axis=0)
        all_rw_dice_array[i, :] = np.mean(rw_dice_array, axis=0)


        # print manual_dice
        print manual_dice_path
        print manual_dice
        print np.round(np.mean(manual_dice_array, axis=0), 3)
        print np.round(np.mean(rw_dice_array, axis=0), 3)
        print '---------------------------------------', i ,'--------------------------------------------'
    print '-----------------------all-------------------------'
    print np.round(np.mean(all_maunal_dice_array, axis=0), 3)
    print np.round(np.mean(rw_dice_array, axis=0), 3)
    print 'manual: ', np.round(np.std(all_maunal_dice_array, axis=0), 3) / np.round(np.mean(all_maunal_dice_array, axis=0), 3)
    print 'rw: ', np.round(np.std(rw_dice_array, axis=0), 3) / np.round(np.mean(rw_dice_array, axis=0), 3)

























