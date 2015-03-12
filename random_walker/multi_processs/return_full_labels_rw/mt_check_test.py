__author__ = 'zhouguangfu'

import numpy as np
import nibabel as nib
import os


if __name__ == "__main__":
    lines = []
    whole_brain_mask = nib.load('/nfs/t2/BAA/SSR/S0001/mt/motion/zstat1.nii.gz').get_data()

    for line in open('/nfs/j3/userhome/zhouguangfu/workingdir/BAA/mt/2006subjID'):
        if line is not '':
            lines.append(line.rstrip('\r\n'))
            subject_zstat_path = '/nfs/t2/BAA/SSR/' + line.rstrip('\r\n') + '/mt/motion/'
            # print 'Subject zstat path: ', subject_zstat_path

            if  not os.path.exists(subject_zstat_path + 'mt_z5.0_ff.nii.gz'):
                print 'Rename to mt_z5.0_ff.nii.gz.', ' subject num:', line.rstrip('\r\n')
                image = nib.load(subject_zstat_path + 'zgf_mt_z5.0.nii.gz')
                affine = image.get_affine()
                data = image.get_data()
                nib.save(nib.Nifti1Image(data, affine), subject_zstat_path + 'mt_z5.0_ff.nii.gz')
            else:
                print subject_zstat_path + 'mt_z5.0_ff.nii.gz', ' subject num:', line.rstrip('\r\n')


            # if  not os.path.exists(subject_zstat_path + 'zgf_mt_z5.0.nii.gz'):
            #     print '--------------------zgf_mt_z5.0.nii.gz not found.', ' subject num:', line.rstrip('\r\n')
            #     continue

            # lzg_data = nib.load(subject_zstat_path + 'lzg_mt_z5.0.nii.gz').get_data()
            # zgf_data = nib.load(subject_zstat_path + 'zgf_mt_z5.0.nii.gz').get_data()
            #
            # #similarity = np.corrcoef(lzg_data[whole_brain_mask > 0], zgf_data[whole_brain_mask > 0])[0, 1]
            # #if similarity != 1.0:
            # #    print '--------------------Similarity: ', similarity, ' subject num:', line.rstrip('\r\n')
            #
            # if (lzg_data.sum() - zgf_data.sum() ) > 5:
            #     print line.rstrip('\r\n')



