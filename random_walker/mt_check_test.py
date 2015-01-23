__author__ = 'zgf'

import numpy as np
import nibabel as nib


if __name__ == "__main__":
    lines = []
    for line in open('/nfs/j3/userhome/zhouguangfu/workingdir/BAA/mt/2006subjID'):
        if line is not '':
            lines.append(line.rstrip('\r\n'))
            subject_zstat_path = '/nfs/t2/BAA/SSR/' + line.rstrip('\r\n') + '/mt/motion/'
            print 'Subject zstat path: ', subject_zstat_path

            lzg_data = nib.load(subject_zstat_path + 'lzg_mt_z5.0.nii.gz').get_data()
            zgf_data = nib.load(subject_zstat_path + 'zgf_mt_z5.0.nii.gz').get_data()

            similarity = np.corrcoef(lzg_data[lzg_data > 0], zgf_data[zgf_data > 0])[0, 1]
            if similarity != 1.0:
                print 'Similarity: ', similarity, ' subject num:', line.rstrip('\r\n')



