#File to initialize 2D image, pass input image to kernel, and peform timing analyis on output image.
#Authors. Kaylo Littlejohn and Desmond Yao 2019.

import numpy as np
from dwt_serial import *
from dwt_naive_parallel import *
from dwt_optimized_parallel import *

"""
1. Test serial with some random array
"""
signal = np.random.rand(2000, 2000).astype(np.float32)

wav = gen_wavelet()
cA, cH, cV, cD, serial_time = run_DWT(signal, wav, False, mode='zero')
# rec_signal = run_iDWT(wav, cA, cH, cV, cD, mode='zero')
#
# print('Rec same as original: {}'.format(np.allclose(signal, rec_signal, atol=5e-7)))

"""
2. Test parallel with some random array
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
dwt = DWT_naive()
h_cA, h_cH, h_cV, h_cD, kernel_time = dwt.dwt_gpu_naive(signal, filters)

#implement optimized separable version of 2D dwt
dwt_opt = DWT_optimized()
h_cAo, h_cHo, h_cVo, h_cDo, kernel_time_o = dwt_opt.dwt_gpu_optimized(signal,filters)

print('naive same as serial c_A: {}'.format(np.allclose(cA, h_cA, atol=5e-7)))
print('naive same as serial c_H: {}'.format(np.allclose(cH, h_cH, atol=5e-7)))
print('naive same as serial c_V: {}'.format(np.allclose(cV, h_cV, atol=5e-7)))
print('naive same as serial c_D: {}'.format(np.allclose(cD, h_cD, atol=5e-7)))

print('optimized same as serial c_A: {}'.format(np.allclose(h_cA, h_cAo, atol=5e-7)))
print('optimized same as serial c_Hs: {}'.format(np.allclose(h_cH, h_cHo, atol=5e-7)))
print('optimized same as serial c_Vs: {}'.format(np.allclose(h_cV, h_cVo, atol=5e-7)))
print('optimized same as serial c_Ds: {}'.format(np.allclose(h_cD, h_cDo, atol=5e-7)))

print(h_cA).shape[0]

print(h_cA)
print("------------------------------")
print(h_cHo)

print('\nSerial time: {}'.format(serial_time))
print('Parallel time: {}'.format(kernel_time))
