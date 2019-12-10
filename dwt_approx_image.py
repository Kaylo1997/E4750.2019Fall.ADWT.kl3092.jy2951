import pywt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from dwt_serial import *
import scipy.misc as misc

dirpath = os.getcwd()
inpath = os.path.join(dirpath, 'Image')
f_img = 'test_gray.png'
f_out = 'test_gray_approx.png'

picpath = os.path.join(inpath, f_img)
outpath = os.path.join(inpath, f_out)
gray_img = misc.imread(picpath, mode='F').astype(np.float32)

wav = gen_wavelet()
cA, cH, cV, cD, _ = run_DWT(gray_img, wav, False, mode='zero')

cH_empty = np.zeros(cH.shape)
cV_empty = np.zeros(cV.shape)
cD_empty = np.zeros(cD.shape)

approx_img = run_iDWT(wav, cA, cH_empty, cV_empty, cD_empty, mode='zero')
approx_img = approx_img * 1.1
plt.imsave(str(outpath), approx_img, cmap='gray')
print('nothing')