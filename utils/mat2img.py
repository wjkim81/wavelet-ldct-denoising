from skimage.external.tifffile import imsave as t_imsave
from skimage.io import imsave
from scipy.io.matlab.mio import savemat, loadmat
import numpy as np
import os


if __name__ == "__main__":

    #sidd1
    noisy1_fn = 'siddplus_valid_noisy_raw.mat'
    noisy1_key = 'siddplus_valid_noisy_raw'
    noisy1_mat = loadmat(os.path.join('D:/data/denoising/test/sidd1',  noisy1_fn))[noisy1_key]
    n_im1, h1, w1 = noisy1_mat.shape
    noisy1_dir = 'D:/data/denoising/test/sidd1/valid'

    if not os.path.exists(noisy1_dir):
        os.makedirs(noisy1_dir)

    for i in range(n_im1):
        print('\n[*]PROCESSING..{}/{}'.format(i,n_im1))

        noisy1 = np.reshape(noisy1_mat[i, :, :], (h1, w1))
        noisy1_name = noisy1_dir + '/' + str(i).zfill(4) + '.tiff'
        t_imsave(noisy1_name, noisy1)

    #sidd2
    noisy2_fn = 'siddplus_valid_noisy_srgb.mat'
    noisy2_key = 'siddplus_valid_noisy_srgb'
    noisy2_mat = loadmat(os.path.join('D:/data/denoising/test/sidd2', noisy2_fn))[noisy2_key]
    n_im2, h2, w2, c2 = noisy2_mat.shape
    noisy2_dir = 'D:/data/denoising/test/sidd2/valid'

    if not os.path.exists(noisy2_dir):
        os.makedirs(noisy2_dir)

    for i in range(n_im2):
        print('\n[*]PROCESSING..{}/{}'.format(i,n_im2))

        noisy2 = np.reshape(noisy2_mat[i, :, :, :], (h2, w2, c2))
        noisy2_name = noisy2_dir + '/' + str(i).zfill(4) + '.png'
        imsave(noisy2_name, noisy2)
    
    





