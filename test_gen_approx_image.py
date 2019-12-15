import pywt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from dwt_serial import *
import scipy.misc as misc

dirpath = os.getcwd()
inpath = os.path.join(dirpath, 'ApproximateImage')
f_img = 'test.png'
f_gray = 'test_gray.png'
f_out = 'test_gray_approx.png'

picpath = os.path.join(inpath, f_img)
graypath = os.path.join(inpath, f_gray)
outpath = os.path.join(inpath, f_out)

color_img = mpimg.imread(picpath).astype(np.float32)

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

gray_img = rgb2gray(color_img)
plt.imsave(str(graypath), gray_img, cmap='gray')

wav = gen_wavelet()
cA, cH, cV, cD, _ = run_DWT(gray_img, wav, False, mode='zero')

cH_empty = np.zeros(cH.shape)
cV_empty = np.zeros(cV.shape)
cD_empty = np.zeros(cD.shape)

approx_img = run_iDWT(wav, cA, cH_empty, cV_empty, cD_empty, mode='zero')
plt.imsave(str(outpath), approx_img, cmap='gray')