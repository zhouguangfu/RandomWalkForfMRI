__author__ = 'zgf'

import datetime
import os
import numpy as np
import nibabel as nib

from configs import *

SESSION_NUM = 7

GSS_FILE_PATH = ANALYSIS_DIR + 'gss/' + 'GSS_thr_0.1_zgf.nii.gz '
GSS_GOLD_FILE_PATH = ANALYSIS_DIR + 'gold/'+ 'CV_GSS_gold_thr0.1.nii.gz'

MARW_GOLD_FILE_PATH = ANALYSIS_DIR + 'gold/' + 'MARW_gold.nii.gz'

CV_FEAT_FILE_PATH = ANALYSIS_DIR + 'gold/' + 'CV_Feat_gold.nii.gz'

MANUAL_GOLD_FILE_PATH = ANALYSIS_DIR + 'gold/' + 'manual_gold_z2.3.nii.gz'

FACE_LABEL_FILE_PATH = ANALYSIS_DIR + 'face_label.nii.gz'

SUBJECT_ID = '/nfs/t2/fmricenter/obj_reliability/sessid'

image = nib.load(ACTIVATION_DATA_DIR)
affine = image.get_affine()
image = image.get_data()

def dice(volume1,volume2):
    """
    volume1,volume2: array-like, bool
    return dice value
    """
    if volume1.shape != volume2.shape:
        raise ValueError("Shape can't match")
    intersection = np.logical_and(volume1,volume2)
    if (volume1.sum() + volume2.sum()==0):
        return 0
    return 2.*intersection.sum() / (volume1.sum() + volume2.sum())

def connect_gss_4d_volume():
    prefix_dir = '/nfs/t2/fmricenter/obj_reliability/4d/'
    suffix = '_GSS_thr0.1.nii.gz'

    GSS_data = np.zeros_like(image)
    for i in range(len(SUBJECT_NAMES)):
        filepath = prefix_dir + SUBJECT_NAMES[i] + suffix
        GSS_data[..., i * SESSION_NUM : (i+1) * SESSION_NUM] = nib.load(filepath).get_data()

    if not os.path.exists(GSS_FILE_PATH):
        os.makedirs(GSS_FILE_PATH)

    nib.save(nib.Nifti1Image(GSS_data, affine), GSS_FILE_PATH)

def generate_gss_4d_volume():
    face_label_data = nib.load(FACE_LABEL_FILE_PATH).get_data()
    all_gss_gold_data = np.zeros_like(image)
    for i in range(image.shape[3]):
        for roi_index in range(len(ROI)):
            mask = np.logical_and(face_label_data == (roi_index + 1), image[..., i] > 2.3)
            all_gss_gold_data[mask, i] = (roi_index + 1)

    nib.save(nib.Nifti1Image(all_gss_gold_data, affine), GSS_FILE_PATH)

def connect_manual_gold_4d_volume():
    prefix_dir = '/nfs/t2/fmricenter/obj_reliability/cv/'
    suffix = 'manual_z2.3.nii.gz'
    all_gold_data = np.zeros_like(image)

    subject_id_file = open(SUBJECT_ID)
    lines = subject_id_file.readlines()
    index = 0
    for line in lines:
        line = line.replace("\n","")
        gold_file = prefix_dir + 'cv_' + line + '.gfeat/cope1.feat/stats/' + suffix
        all_gold_data[..., index] = nib.load(gold_file).get_data()
        index += 1

    nib.save(nib.Nifti1Image(all_gold_data, affine), MANUAL_GOLD_FILE_PATH)

def generate_gss_gold_4d_volume():
    prefix_dir = '/nfs/t2/fmricenter/obj_reliability/cv/'
    suffix = 'zstat1.nii.gz'

    subject_id_file = open(SUBJECT_ID)
    lines = subject_id_file.readlines()
    cv_zstat_file_path = []
    for line in lines:
        line = line.replace("\n", "")
        gold_file = prefix_dir + 'cv_' + line + '.gfeat/cope1.feat/stats/' + suffix
        cv_zstat_file_path.append(gold_file)

    # print 'len(cv_zstat_file_path): ', len(cv_zstat_file_path)
    # print cv_zstat_file_path

    face_label_data = nib.load(FACE_LABEL_FILE_PATH).get_data()
    all_gss_gold_data = np.zeros_like(image)
    for i in range(len(cv_zstat_file_path)):
        for roi_index in range(len(ROI)):
            image_data = nib.load(cv_zstat_file_path[i]).get_data()
            mask = np.logical_and(face_label_data == (roi_index + 1), image_data > 2.3)
            all_gss_gold_data[mask, i] = (roi_index + 1)

    nib.save(nib.Nifti1Image(all_gss_gold_data, affine), GSS_GOLD_FILE_PATH)

def generate_cv_4d_image_volume():
    prefix_dir = '/nfs/t2/fmricenter/obj_reliability/cv/'
    suffix = 'zstat1.nii.gz'

    subject_id_file = open(SUBJECT_ID)
    lines = subject_id_file.readlines()
    cv_zstat_file_path = []
    for line in lines:
        line = line.replace("\n", "")
        gold_file = prefix_dir + 'cv_' + line + '.gfeat/cope1.feat/stats/' + suffix
        cv_zstat_file_path.append(gold_file)

    # print 'len(cv_zstat_file_path): ', len(cv_zstat_file_path)
    print cv_zstat_file_path

    all_gss_gold_data = np.zeros_like(image)
    for i in range(len(cv_zstat_file_path)):
        all_gss_gold_data[..., i] = nib.load(cv_zstat_file_path[i]).get_data()

    nib.save(nib.Nifti1Image(all_gss_gold_data, affine), CV_FEAT_FILE_PATH)

if __name__ == '__main__':
    starttime = datetime.datetime.now()

    # connect_gss_4d_volume()
    # connect_manual_gold_4d_volume()
    generate_gss_gold_4d_volume()
    generate_cv_4d_image_volume()
    exit(0)

    gss_data = nib.load(GSS_FILE_PATH).get_data()
    gold_data = nib.load(MANUAL_GOLD_FILE_PATH).get_data()

    for roi_index in range(len(ROI)):
        print '----------roi_index: ', ROI[roi_index], '-----------------'

        region_size = []
        gold_size = []
        dice_list = []

        for i in range(gss_data.shape[3]):
            region_size.append((gss_data[..., i] == (roi_index + 1)).sum())
            gold_size.append((gold_data[..., i] == (roi_index + 1)).sum())
            sess_dice = dice(gss_data[..., i] == (roi_index + 1), gold_data[..., i] == (roi_index + 1))
            dice_list.append(sess_dice)

        # square root mean squared error(RMSE) of volume ****************************
        RMSEs = []
        for subject_index in range(len(SUBJECT_NAMES)):
            region_size_variace = []
            for session_index in range(SESSION_NUM * subject_index, SESSION_NUM * (subject_index+1)):
                region_size_variace.append((gold_size[session_index] - region_size[session_index])**2)
            RMSEs.append(8 * np.sqrt(np.mean(region_size_variace)))      # region size to volume

        print 'RMSEs: ', RMSEs
        print 'Mean:', '%.4f' % np.mean(RMSEs)
        print 'MSE:', '%.4f' % (np.std(RMSEs))/(np.sqrt(10))
        # square root mean squared error(RMSE) of volume ****************************




        # # the variability (CV) of volume in precision  ****************************
        #cv_list = []
        #for i in range(len(data)):
        #    withinmean = np.mean((region_size[i],gold_size[i]))
        #    withinstd = np.std((region_size[i],gold_size[i]))
        #    cv = withinstd/withinmean
        #    cv_list.append(cv)
        #for m in range(10):
        #    withinCVmean = np.mean(cv_list[7*m : 7*(m+1)])
        #    print '%.4f' % withinCVmean
        #print 'Mean:',(np.mean(cv_list))
        # # the variability (CV) of volume in precision  ****************************


        # # the location of precision (dice coefficients) **************************
        #for s in range(10):
        #    print '%.4f' %  (np.mean(dice_list[7*s: 7*(s+1)]))
        #    volumevar = 8 * np.mean(sizevar[7*s: 7*(s+1)])
        #    print  '%d' % volumevar
        # # the location of precision (dice coefficients) **************************


    endtime = datetime.datetime.now()
    print 'Time costs: ', (endtime - starttime)
    print "Program end..."

