__author__ = 'zgf'

import datetime
import multiprocessing
import os

import nibabel as nib
import numpy as np

from configs import *

#Convert file format:
#crlConvertBetweenFileFormats -in inputfilename -out outputfilename[-opct] outputPixelComponentType
crlConvertBetweenFileFormats = '/nfs/j3/userhome/zhouguangfu/workingdir/crkit_build/install/bin/crlConvertBetweenFileFormats'

#staple command
#crlSTAPLE -o Weights.nrrd 1.nrrd 2.nrrd 3.nrrd 4.nrrd
crlSTAPLE = '/nfs/j3/userhome/zhouguangfu/workingdir/crkit_build/install/bin/crlSTAPLE'

#crlIndexOfMaxComponent command
#crlIndexOfMaxComponent Weights.nrrd Seg.nrrd
crlIndexOfMaxComponent = '/nfs/j3/userhome/zhouguangfu/workingdir/crkit_build/install/bin/crlIndexOfMaxComponent'

#crlOverlapstats3d command
#crlOverlapstats3d 1.nrrd REF.nrrd 1
crlOverlapstats3d = '/nfs/j3/userhome/zhouguangfu/workingdir/crkit_build/install/bin/crlOverlapstats3d'


ATLAS_NUM = 10
SUBJECTS_SESSION_NUMBERS = 70
LEFT_RIGHT_BRAIN_NAME = ['left_brain', 'right_brain']

image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

all_laebls_exsit_atlas_data_index = np.load(ATLAS_TOP_DIR + 'all_laebls_exsit_atlas_data_150.npy')

def generate_single_subject_nii_volume(subject_index):
    all_subject_session_rw = nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + 'subjects_rw_all_atlas_results/' +
                                      str(subject_index) + '_regions_rw.nii.gz').get_data()
    all_file_path = ' '
    # all_laebls_exsit_atlas_data = nib.load(ATLAS_TOP_DIR + 'all_laebls_exsit_atlas_data_150.nii.gz').get_data()
    # all_laebls_exsit_atlas_data_index = np.load(ATLAS_TOP_DIR + 'all_laebls_exsit_atlas_data_150.npy')


    rw_path = RW_AGGRAGATOR_RESULT_DATA_DIR + 'staple/random/' + str(ATLAS_NUM) + '/'+ str(subject_index) + '/'
    if not os.path.exists(rw_path):
        os.makedirs(rw_path)

    # for atlas_index in range(ATLAS_NUM):
    #     nib.save(nib.Nifti1Image(all_subject_session_rw[..., atlas_index], affine), rw_path + str(atlas_index) + '_session.nii')
    #     all_file_path += ' ' + rw_path + str(atlas_index) + '_session.nii'
    np.random.seed()
    indexs_right = np.random.choice(range(150), ATLAS_NUM, replace=False)
    indexs_left = np.random.choice(range(150), ATLAS_NUM, replace=False)

    # print '-------------------------- ', subject_index, ' -------------------------------'
    # print 'indexs_right: ', indexs_right
    # print 'indexs_left: ', indexs_left

    for top_index in range(ATLAS_NUM):
        left_brain_rois_index = indexs_left[top_index]
        right_brain_rois_index = indexs_right[top_index]

        atlas_data = np.zeros((all_subject_session_rw.shape[0],
                               all_subject_session_rw.shape[1],
                               all_subject_session_rw.shape[2]))
        # atlas_data[all_subject_session_rw[..., right_brain_rois_index] == 1] = 1
        # atlas_data[all_subject_session_rw[..., right_brain_rois_index] == 3] = 3
        # atlas_data[all_subject_session_rw[..., left_brain_rois_index] == 2] = 2
        # atlas_data[all_subject_session_rw[..., left_brain_rois_index] == 4] = 4

        atlas_data[all_subject_session_rw[..., all_laebls_exsit_atlas_data_index[right_brain_rois_index]] == 1] = 1
        atlas_data[all_subject_session_rw[..., all_laebls_exsit_atlas_data_index[right_brain_rois_index]] == 3] = 3
        atlas_data[all_subject_session_rw[..., all_laebls_exsit_atlas_data_index[left_brain_rois_index]] == 2] = 2
        atlas_data[all_subject_session_rw[..., all_laebls_exsit_atlas_data_index[left_brain_rois_index]] == 4] = 4

        nib.save(nib.Nifti1Image(atlas_data, affine), rw_path + str(top_index) + '_session.nii')
        all_file_path += ' ' + rw_path + str(top_index) + '_session.nii'

    print 'Generate single subject session_nii_volume: ', subject_index

    return all_file_path


def random_walker_staple_analysis(subject_index):
    rw_path = RW_AGGRAGATOR_RESULT_DATA_DIR + 'staple/rw_random/' + str(ATLAS_NUM) + '/' + str(subject_index) + '/'
    if not os.path.exists(rw_path):
        os.makedirs(rw_path)
    all_file_path = generate_single_subject_nii_volume(subject_index)


    # print 'STAPLE Processing...'
    rw_staple_output = os.popen(crlSTAPLE + ' -o ' + rw_path + 'weights.nii ' + all_file_path)
    rw_staple_file = open(rw_path + 'log.txt', 'w+')
    rw_staple_file.write(rw_staple_output.read())
    rw_staple_file.close()

    # print 'STAPLE end. Compute the maxinum_component...'
    os.popen(crlIndexOfMaxComponent + ' ' + rw_path + 'weights.nii ' + rw_path + 'maxinum_component.nii')

    # rw_staple_dice_file = open(rw_path + 'dice.txt', 'w+')
        
    # #Compute the dice value between per seeion rw segmentation result and maxinum_component
    # for atlas_index in range(ATLAS_NUM):
    #     for roi_index in range(len(ROI) + 1):
    #         rw_staple_dice_output = os.popen(crlOverlapstats3d + ' ' + rw_path + str(atlas_index) + '.nii '
    #                                        + rw_path + 'maxinum_component.nii' + ' ' + str(roi_index + 1))
    #         rw_staple_dice_file.write(rw_staple_dice_output.read())
    #         print 'Dice => atlas_index: ', atlas_index, '  roi_index: ', roi_index
    # rw_staple_dice_file.close()

    #Delete files
    print 'random_walker_staple_analysis => Deleteing...'
    rw_path = RW_AGGRAGATOR_RESULT_DATA_DIR + 'staple/random/' + str(ATLAS_NUM) + '/'+ str(subject_index) + '/'
    os.popen('rsync -avh --delete /tmp/test/ ' + rw_path)

def generate_rw_prob_result():
    temp_image = np.zeros_like(image).astype(np.int)

    for subject_index in range(SUBJECTS_SESSION_NUMBERS):
        rw_path = RW_AGGRAGATOR_RESULT_DATA_DIR + 'staple/rw_random/' + str(ATLAS_NUM) + '/' + str(subject_index) + '/'
        temp_image[..., subject_index] = nib.load(rw_path + 'maxinum_component.nii').get_data()

        #Delete files
        print 'generate_rw_prob_result => Deleteing...'
        os.popen('rsync -avh --delete /tmp/test/ ' + rw_path)

    temp_image[temp_image == 5] = 0
    nib.save(nib.Nifti1Image(temp_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + 'staple/rw/random_' + str(ATLAS_NUM)
             + '_' + RW_PROB_RESULT_FILE)


if __name__ == "__main__":
    starttime = datetime.datetime.now()

    # #For single process
    # for subject_index in range(SUBJECTS_SESSION_NUMBERS):
    #     random_walker_staple_analysis(subject_index)
    #     print 'Staple process => subject_index: ', subject_index

    #For multi process
    for i in range(15):
        ATLAS_NUM = (i + 1) * 10
        print '------------------------------- ', ATLAS_NUM, ' ------------------------------------'

        starttime = datetime.datetime.now()
        process_num = 14
        for cycle_index in range(SUBJECTS_SESSION_NUMBERS / process_num):
            pool = multiprocessing.Pool(processes=process_num)
            pool_outputs = pool.map(random_walker_staple_analysis, range(cycle_index * process_num,
                                                                (cycle_index + 1) * process_num))
            pool.close()
            pool.join()

            print 'Cycle index: ', cycle_index, 'Time cost: ', (datetime.datetime.now() - starttime)
            starttime = datetime.datetime.now()

        generate_rw_prob_result()
        endtime = datetime.datetime.now()
        print 'Time cost: ', (endtime - starttime)
    print "Program end..."













