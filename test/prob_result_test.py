__author__ = 'zgf'
import datetime
import numpy as np
import nibabel as nib
from configs import *

SUBJECT_NUM = 70
ATLAS_NUM = 202

if __name__ == "__main__":
    starttime = datetime.datetime.now()
    lines = []

    data = nib.load(ANALYSIS_DIR + 'rw/rw_prob_result_file.nii.gz').get_data()

    for i in range(SUBJECT_NUM):
        labels_size = [(data[..., i] == 1).sum(), (data[..., i] == 2).sum(), (data[..., i] == 3).sum(), (data[..., i] == 4).sum()]
        print 'i: ', i, '   ', labels_size


    endtime = datetime.datetime.now()
    print 'Time cost: ', (endtime - starttime)
    print "Program end..."




