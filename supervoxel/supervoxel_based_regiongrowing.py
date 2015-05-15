__author__ = 'zgf'

import nibabel as nib
import numpy as np

from skimage.segmentation import slic
from scipy.ndimage.morphology import binary_dilation
from configs import *

SUPERVOXEL_SEGMENTATION = 50000
RW_AGGRAGATOR_RESULT_DATA_DIR = "/Users/zgf/Documents/github/data/"

img = nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + "zstat1.nii.gz")
affine = img.get_affine()
volume = img.get_data()

def compute_supervoxel(volume):
    gray_image = (volume - volume.min()) * 255 / (volume.max() - volume.min())

    slic_image = slic(gray_image.astype(np.float),
                      n_segments=SUPERVOXEL_SEGMENTATION,
                      slic_zero=True,
                      sigma=2,
                      multichannel =False,
                      enforce_connectivity=True)
    # nib.save(nib.Nifti1Image(slic_image, affine), RW_AGGRAGATOR_RESULT_DATA_DIR + 'supervoxel.nii.gz')

    return slic_image

def compute_slic_max_region_mean(volume, region_volume, slic_image):
    neighbor_slic = binary_dilation(region_volume)
    neighbor_slic[region_volume > 0] = 0

    neighbor_values = np.unique(slic_image[neighbor_slic > 0])
    # print 'neighbor_values: ', neighbor_values

    region_means = np.zeros((len(neighbor_values - 1), ))
    for i in range(len(neighbor_values)):
        if neighbor_values[i] !=0 :
            neighbor_slic[slic_image == neighbor_values[i]] = 1
            region_means[i] = volume[slic_image == neighbor_values[i]].mean()
    # print 'region_means: ',region_means
    # print 'neighbor_values[region_means.argmax(): ', neighbor_values[region_means.argmax()]

    return neighbor_slic, slic_image == neighbor_values[region_means.argmax()]



def supervoxel_based_regiongrowing(volume, seed, size=10):
    '''
    :param volume: 3D volume
    :param seed: a list of cordinates
    :param size: region size
    :return:
    '''
    seed = np.array(seed)
    slic_image = compute_supervoxel(volume)
    seed_region = np.zeros_like(slic_image)
    seed_region[slic_image == slic_image[seed[0], seed[1], seed[2]]] = 1

    seed_regions = np.zeros((seed_region.shape[0], seed_region.shape[1], seed_region.shape[2], size))
    seed_regions[..., 0] = seed_region

    neighbor_slics = np.zeros((seed_region.shape[0], seed_region.shape[1], seed_region.shape[2], size))

    for i in range(0, size-1):
        neighbor_slic, best_parcel = compute_slic_max_region_mean(volume, seed_region, slic_image)
        seed_region[best_parcel] = 1
        seed_regions[..., i + 1] = seed_region
        neighbor_slics[..., i] = neighbor_slic

    neighbor_slics[..., size - 1] = compute_slic_max_region_mean(volume, seed_region, slic_image)[0]

    # nib.save(nib.Nifti1Image(seed_regions, affine),
    #          RW_AGGRAGATOR_RESULT_DATA_DIR + str(SUPERVOXEL_SEGMENTATION) + '_slic_regiongrowing.nii.gz')
    # nib.save(nib.Nifti1Image(neighbor_slics, affine),
    #          RW_AGGRAGATOR_RESULT_DATA_DIR + str(SUPERVOXEL_SEGMENTATION) + '_neighbor_slic_regiongrowing.nii.gz')

    return neighbor_slics, seed_regions

def compute_optional_region_based_AC_value(volume, regions, neighbor_slics):
    AC_values = np.zeros((regions.shape[3], ))
    for i in range(regions.shape[3]):
        AC_values[i] = volume[regions[..., i] > 0].mean() - volume[neighbor_slics[..., i] > 0].mean()

    # print 'AC_values: ', AC_values
    return regions[..., AC_values.argmax()]



if __name__ == "__main__":
    import datetime
    starttime = datetime.datetime.now()

    roi_peak_points = np.load(RW_AGGRAGATOR_RESULT_DATA_DIR + "peak_points_all_sub.npy")
    all_volumes = nib.load(RW_AGGRAGATOR_RESULT_DATA_DIR + "all_session.nii.gz").get_data()
    result_vlomes = np.zeros_like(all_volumes)

    for j in range(0, all_volumes.shape[3]):
    # for j in range(0, 2):
        for i in range(len(ROI)):
            seed = np.array([roi_peak_points[j, i, :]]).astype(np.int)[0]

            neighbor_slics, regions = supervoxel_based_regiongrowing(all_volumes[..., j], seed, size=10)
            optimal_region = compute_optional_region_based_AC_value(all_volumes[..., j], regions, neighbor_slics)

            result_vlomes[optimal_region > 0, j] = i + 1

            print 'j: ', j, ' i: ', i, ' ROI: ', ROI[i]

    nib.save(nib.Nifti1Image(result_vlomes, affine),
             RW_AGGRAGATOR_RESULT_DATA_DIR + str(SUPERVOXEL_SEGMENTATION) + '_result_regions.nii.gz')

    endtime = datetime.datetime.now()
    print 'time: ', (endtime - starttime)


















