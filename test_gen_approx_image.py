import pywt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from dwt_serial import *

# Obtain the path
dirpath = os.getcwd()
inpath = os.path.join(dirpath, 'ApproximateImage')
f_img = 'test.png'
f_gray = 'test_gray.png'
f_out = 'test_gray_approx.png'

picpath = os.path.join(inpath, f_img)
graypath = os.path.join(inpath, f_gray)
outpath = os.path.join(inpath, f_out)

# Load the image in RGB format
color_img = mpimg.imread(picpath).astype(np.float32)

# Define a function that calculates the grayscale
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

# Convert the RBG image to scale and saving it in the destination directory
gray_img = rgb2gray(color_img)
plt.imsave(str(graypath), gray_img, cmap='gray')

# Obtain the CDF wavelet object
wav = gen_wavelet()

# Calculate the approximation and detail coefficients
cA, cH, cV, cD, _ = run_DWT(gray_img, wav, False, mode='zero')

# Turn the detail into zeros
cH_empty = np.zeros(cH.shape)
cV_empty = np.zeros(cV.shape)
cD_empty = np.zeros(cD.shape)

# Reconstruct the approximated grayscale image using IDWT
approx_img = run_iDWT(wav, cA, cH_empty, cV_empty, cD_empty, mode='zero')
plt.imsave(str(outpath), approx_img, cmap='gray')