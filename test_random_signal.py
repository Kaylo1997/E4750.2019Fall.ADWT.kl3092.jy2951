#File to initialize 2D image, pass input image to kernel, and peform timing analyis on output image.
#Authors. Kaylo Littlejohn and Desmond Yao 2019.

import numpy as np
import pywt
from dwt_serial import *
from dwt_naive_separable_parallel import *
from dwt_tiled_separable_parallel import *
from dwt_nonseparable_parallel import *


BLOCK_WIDTH = 32
"""
1. Test serial with some random array
"""
signal = np.random.rand(4000, 4000).astype(np.float32)

wav = gen_wavelet()
cA, cH, cV, cD, serial_time = run_DWT(signal, wav, False, mode='zero')

"""
2. Test naive parallel with some random array
"""

# Define the coefficients for the CDF9/7 filters
factor = 1

# FORWARD FILTER COEFFICIENTS
# Forward Decomposition filter: lowpass
cdf97_an_lo = factor * np.array([0, 0.026748757411, -0.016864118443, -0.078223266529, 0.266864118443,
                                 0.602949018236, 0.266864118443, -0.078223266529, -0.016864118443,
                                 0.026748757411])

# Forward Decomposition filter: highpass
cdf97_an_hi = factor * np.array([0, 0.091271763114, -0.057543526229, -0.591271763114, 1.11508705,
                                 -0.591271763114, -0.057543526229, 0.091271763114, 0, 0])

# INVERSE FILTER COEFFICIENTS
# Inverse Reconstruction filter: lowpass
cdf97_syn_lo = factor * np.array([0, -0.091271763114, -0.057543526229, 0.591271763114, 1.11508705,
                                  0.591271763114, -0.057543526229, -0.091271763114, 0, 0])

# Inverse Reconstruction filter: highpass
cdf97_syn_hi = factor * np.array([0, 0.026748757411, 0.016864118443, -0.078223266529, -0.266864118443,
                                  0.602949018236, -0.266864118443, -0.078223266529, 0.016864118443,
                                  0.026748757411])
filters = np.vstack((cdf97_an_lo, cdf97_an_hi, cdf97_syn_lo, cdf97_syn_hi)).astype(np.float32)


dwt_naive = DWT_naive_separable()
h_cA, h_cH, h_cV, h_cD, kernel_time = dwt_naive.dwt_gpu_naive_separable(signal, filters, BLOCK_WIDTH)

"""
Test tiled parallel kernel with some random array
"""
# Optimized shared memory of 2D DWT using shared memory
dwt_tiled = DWT_tiled_separable()
h_cA_tiled, h_cH_tiled, h_cV_tiled, h_cD_tiled, kernel_time_tiled = dwt_tiled.dwt_gpu_tiled_separable(signal, filters, BLOCK_WIDTH)

"""
Test nonseparable parallel kernel with some random array
"""
# Nonseparable version of 2D DWT
dwt_nonseparable = DWT_nonseparable()
h_cAo, h_cHo, h_cVo, h_cDo, kernel_time_o = dwt_nonseparable.dwt_gpu_nonseparable(signal, filters, BLOCK_WIDTH)

print('naive same as serial c_A: {}'.format(np.allclose(cA, h_cA, atol=5e-7)))
print('naive same as serial c_H: {}'.format(np.allclose(cH, h_cH, atol=5e-7)))
print('naive same as serial c_V: {}'.format(np.allclose(cV, h_cV, atol=5e-7)))
print('naive same as serial c_D: {}'.format(np.allclose(cD, h_cD, atol=5e-7)))

print('\ntiled same as serial c_A: {}'.format(np.allclose(cA, h_cA_tiled, atol=5e-7)))
print('tiled same as serial c_H: {}'.format(np.allclose(cH, h_cH_tiled, atol=5e-7)))
print('tiled same as serial c_V: {}'.format(np.allclose(cV, h_cV_tiled, atol=5e-7)))
print('tiled same as serial c_D: {}'.format(np.allclose(cD, h_cD_tiled, atol=5e-7)))

print('\nNon-separable same as serial c_A: {}'.format(np.allclose(cA, h_cAo, atol=5e-7)))
print('Non-separable same as serial c_H: {}'.format(np.allclose(cH, h_cHo, atol=5e-7)))
print('Non-separable same as serial c_V: {}'.format(np.allclose(cV, h_cVo, atol=5e-7)))
print('Non-separable same as serial c_D: {}'.format(np.allclose(cD, h_cDo, atol=5e-7)))

print('\nSerial time: {}'.format(serial_time))
print('Parallel time: {}'.format(kernel_time))
print('Tiled parallel time: {}'.format(kernel_time_tiled))
print('Non-separable parallel time: {}'.format(kernel_time_o))
