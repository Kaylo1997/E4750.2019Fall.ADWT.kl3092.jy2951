import numpy as np
import pywt
from dwt_serial import *
from dwt_naive_parallel import *
from dwt_shared_mem import *
from dwt_optimized_parallel import *
import os
import matplotlib.pyplot as plt

"""
Parameters to experiment with
"""
BLOCK_WIDTH = 32
M = 100
N = 100
L_max = 50
f_name_full = 'Benchmark_BlockSize{}_Random.png'.format(BLOCK_WIDTH)
f_name_parallel = 'Benchmark_Parallel_BlockSize{}_Random.png'.format(BLOCK_WIDTH)

"""
Setup
"""
wav = gen_wavelet()

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
dwt_naive = DWT_naive()
dwt_tiled = DWT_optimized_shared_mem()
dwt_separable = DWT_optimized()

# Path setup
dirpath = os.getcwd()
outpath = os.path.join(dirpath, 'Results')
if not os.path.exists(outpath):
    os.mkdir(outpath)

outputpath_full = os.path.join(outpath, f_name_full)
outputpath_parallel = os.path.join(outpath, f_name_parallel)

# Create list for benchmark results
vec_serial_time = []
vec_kernel_time = []
vec_kernel_time_tiled = []
vec_kernel_time_separable = []

for L in np.arange(1, L_max + 1, 1):
    signal_i = np.random.rand(L * M, L * N).astype(np.float32)

    cA_i, cH_i, cV_i, cD_i, serial_time_i = run_DWT(signal_i, wav, False, mode='zero')
    h_cA_i, h_cH_i, h_cV_i, h_cD_i, kernel_time_i = dwt_naive.dwt_gpu_naive(signal_i, filters, BLOCK_WIDTH)
    h_cA_tiled_i, h_cH_tiled_i, h_cV_tiled_i, h_cD_tiled_i, kernel_time_tiled_i = dwt_tiled.dwt_gpu_optimized(signal_i, filters,
                                                                                                    BLOCK_WIDTH)
    h_cAo_i, h_cHo_i, h_cVo_i, h_cDo_i, kernel_time_o_i = dwt_separable.dwt_gpu_optimized(signal_i, filters, BLOCK_WIDTH)

    print('\n\n\n###########################################################################')
    print('##                            Iteration {}                               ##'.format(L + 1))
    print('###########################################################################\n')

    print('naive same as serial c_A: {}'.format(np.allclose(cA_i, h_cA_i, atol=5e-7)))
    print('naive same as serial c_H: {}'.format(np.allclose(cH_i, h_cH_i, atol=5e-7)))
    print('naive same as serial c_V: {}'.format(np.allclose(cV_i, h_cV_i, atol=5e-7)))
    print('naive same as serial c_D: {}'.format(np.allclose(cD_i, h_cD_i, atol=5e-7)))
    if not np.allclose(cA_i, h_cA_i, atol=5e-7) and np.allclose(cH_i, h_cH_i, atol=5e-7) \
            and np.allclose(cV_i, h_cV_i, atol=5e-7) and np.allclose(cD_i, h_cD_i, atol=5e-7):
        raise Exception('Naive parallel outputs not same as serial')

    print('\ntiled same as serial c_A: {}'.format(np.allclose(cA_i, h_cA_tiled_i, atol=5e-7)))
    print('tiled same as serial c_H: {}'.format(np.allclose(cH_i, h_cH_tiled_i, atol=5e-7)))
    print('tiled same as serial c_V: {}'.format(np.allclose(cV_i, h_cV_tiled_i, atol=5e-7)))
    print('tiled same as serial c_D: {}'.format(np.allclose(cD_i, h_cD_tiled_i, atol=5e-7)))
    if not np.allclose(cA_i, h_cA_tiled_i, atol=5e-7) and np.allclose(cH_i, h_cH_tiled_i, atol=5e-7) \
            and np.allclose(cV_i, h_cV_tiled_i, atol=5e-7) and np.allclose(cD_i, h_cD_tiled_i, atol=5e-7):
        raise Exception('Tiled parallel outputs not same as serial')

    print('\nseparable same as serial c_A: {}'.format(np.allclose(cA_i, h_cAo_i, atol=5e-7)))
    print('separable same as serial c_H: {}'.format(np.allclose(cH_i, h_cHo_i, atol=5e-7)))
    print('separable same as serial c_V: {}'.format(np.allclose(cV_i, h_cVo_i, atol=5e-7)))
    print('separable same as serial c_D: {}'.format(np.allclose(cD_i, h_cDo_i, atol=5e-7)))

    print('\nSerial time: {}'.format(serial_time_i))
    print('Parallel time: {}'.format(kernel_time_i))
    print('Tiled parallel time: {}'.format(kernel_time_tiled_i))
    print('Separable parallel time: {}'.format(kernel_time_o_i))

    vec_serial_time.append(serial_time_i)
    vec_kernel_time.append(kernel_time_i)
    vec_kernel_time_tiled.append(kernel_time_tiled_i)
    vec_kernel_time_separable.append(kernel_time_o_i)

plt.figure()
n_iter = np.arange(1, L_max + 1, 1)
plt.title('Serial vs Parallel\nBLOCK WIDTH: {}, BASE SIZE: ({},{}), SQUARE MATRICES'.format(BLOCK_WIDTH, M, N))
plt.plot(n_iter, vec_serial_time, label='Serial')
plt.plot(n_iter, vec_kernel_time, label='Naive Parallel')
plt.plot(n_iter, vec_kernel_time_tiled, label='Tiled Parallel')
plt.plot(n_iter, vec_kernel_time_separable, label='Separable Parallel')
plt.xlabel('Scale')
plt.ylabel('Runtime (s)')
plt.legend()
plt.savefig(str(outputpath_full))

plt.figure()
n_iter = np.arange(1, L_max + 1, 1)
plt.title('Parallel Only\nBLOCK WIDTH: {}, BASE SIZE: ({},{}), SQUARE MATRICES'.format(BLOCK_WIDTH, M, N))
plt.plot(n_iter, vec_kernel_time, label='Naive Parallel')
plt.plot(n_iter, vec_kernel_time_tiled, label='Tiled Parallel')
plt.plot(n_iter, vec_kernel_time_separable, label='Separable Parallel')
plt.xlabel('Scale')
plt.ylabel('Runtime (s)')
plt.legend()
plt.savefig(str(outputpath_parallel))