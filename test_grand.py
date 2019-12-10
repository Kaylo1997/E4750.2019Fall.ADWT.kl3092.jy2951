#File to initialize 2D image, pass input image to kernel, and peform timing analyis on output image.
#Authors. Kaylo Littlejohn and Desmond Yao 2019.

import numpy as np
import time
import math
from dwt_serial import *
from dwt_naive_parallel import *
from dwt_optimized_parallel import *
import os, os.path
import matplotlib.image as mpimg

# Data set specifications (750 images total):
#     Square .jpg images ranging from 100x100 to 1000x1000 in set of 25 per 100 pixel increments (250 total)
#     Rect .jpg images  matricies ranging from 200x100 / 100x200 to 2000x1000 / 1000x2000 in set of 25 per 100 pixel increments (500 total)

#file path assuming images are stored in same directory as project
projdir = os.getcwd()

#create list to hold rgb_images
imgs = []

#for every image
for i in range(30):
    
    #decide which type of image (square, rect wide, rect tall) to load and load image into RGB components
    if(i<10):
        #square
        
        #get directory dimension value
        dimdir = str(np.int32((i+1)*100))
        
        #load rgb matrix into list
        path = projdir+'/images/square/square'+dimdir+'/'
        valid_images = [".jpg"]
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            #ignore non-image files
            if ext.lower() not in valid_images:
                continue
            #append image list
            cur_img =mpimg.imread(os.path.join(path,f))
            imgs.append(cur_img)
        
    if 10 <= i and i < 20:
        #rect wide
                        
        #get directory dimension value
        dimdir = str(np.int32((i+1-10)*100))
                        
        #load rgb matrix into list
        path = projdir+'/images/rect_wide/rect'+dimdir+'/'
        valid_images = [".jpg"]
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            #ignore non-image files
            if ext.lower() not in valid_images:
                continue
            #append image list
            cur_img =mpimg.imread(os.path.join(path,f))
            imgs.append(cur_img)
        
    if i >= 20:
        #rect tall
        
        #get directory dimension value
        dimdir = str(np.int32((i+1-20)*100))
                        
        #load rgb matrix into list
        path = projdir+'/images/rect_tall/rect'+dimdir+'/'
        valid_images = [".jpg"]
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            #ignore non-image files
            if ext.lower() not in valid_images:
                continue
            #append image list
            cur_img =mpimg.imread(os.path.join(path,f))
            imgs.append(cur_img)

# Define the coefficients for the CDF9/7 filters
factor = 1

# Forward Decomposition filter: lowpass
cdf97_an_lo = factor * np.array([0, 0.026748757411, -0.016864118443, -0.078223266529, 0.266864118443,
                                 0.602949018236, 0.266864118443, -0.078223266529, -0.016864118443,
                                 0.026748757411])

# Forward Decomposition filter: highpass
cdf97_an_hi = factor * np.array([0, 0.091271763114, -0.057543526229, -0.591271763114, 1.11508705,
                                 -0.591271763114, -0.057543526229, 0.091271763114, 0, 0])

# Inverse Reconstruction filter: lowpass
cdf97_syn_lo = factor * np.array([0, -0.091271763114, -0.057543526229, 0.591271763114, 1.11508705,
                                  0.591271763114, -0.057543526229, -0.091271763114, 0, 0])

# Inverse Reconstruction filter: highpass
cdf97_syn_hi = factor * np.array([0, 0.026748757411, 0.016864118443, -0.078223266529, -0.266864118443,
                                  0.602949018236, -0.266864118443, -0.078223266529, 0.016864118443,
                                  0.026748757411])
filters = np.vstack((cdf97_an_lo, cdf97_an_hi, cdf97_syn_lo, cdf97_syn_hi)).astype(np.float32)

#define arrays to hold our execution times
times_serial = [] # store the running time of the serial algorithm
times_naive = [] # store the running time of the naive parallel algorithms
sizes = [] # store sizes of images
temp_s = 0
temp_naive = 0 #temp execution time variables before getting the average execution time for each matrix size
temp_o = 0
times_opt = []

# define the block width to be used
BLOCK_WIDTH = 32

#for each image in our grand list
for i in range(750):
    
    #decompose image into RGB
    rgb_cpu = imgs[i]/255
    rsig = np.ascontiguousarray(rgb_cpu[:,:,0], dtype=np.float32)
    gsig = np.ascontiguousarray(rgb_cpu[:,:,1], dtype=np.float32)
    bsig = np.ascontiguousarray(rgb_cpu[:,:,2], dtype=np.float32)
    size = rgb_cpu.shape[0]*rgb_cpu.shape[1]
    
    """
    1. Test serial with r,g,b components of image.
    """
    #generate wavelet
    wav = gen_wavelet()
    
    #perform serial 2D DWT on r g b component matricies
    rcA, rcH, rcV, rcD, serial_time_r = run_DWT(rsig, wav, False, mode='zero')
    gcA, gcH, gcV, gcD, serial_time_g = run_DWT(gsig, wav, False, mode='zero')
    bcA, bcH, bcV, bcD, serial_time_b = run_DWT(bsig, wav, False, mode='zero')
    
    #concatenate combine serial execution times to get a final average value for execution time across 2D dwts
    serial_time = serial_time_r + serial_time_g + serial_time_b
    temp_s = temp_s + serial_time
    
    if(np.mod((i+1),25)==0):
        #append arrays holding serial times and size results
        times_serial.append(temp_s/25)
        sizes.append(size)
        temp_s = 0

    """
    2. Test parallel with some random array

    """

    #implement naive separable version of 2D dwt
    dwt = DWT_naive()
    rh_cA, rh_cH, rh_cV, rh_cD, kernel_time_r = dwt.dwt_gpu_naive(rsig, filters, BLOCK_WIDTH)
    gh_cA, gh_cH, gh_cV, gh_cD, kernel_time_g = dwt.dwt_gpu_naive(gsig, filters, BLOCK_WIDTH)
    bh_cA, bh_cH, bh_cV, bh_cD, kernel_time_b = dwt.dwt_gpu_naive(bsig, filters, BLOCK_WIDTH)

    #implement optimized separable version of 2D dwt
    dwt_opt = DWT_optimized()
    rh_cAo, rh_cHo, rh_cVo, rh_cDo, kernel_time_or = dwt_opt.dwt_gpu_optimized(rsig,filters, BLOCK_WIDTH)
    gh_cAo, gh_cHo, gh_cVo, gh_cDo, kernel_time_og = dwt_opt.dwt_gpu_optimized(gsig,filters, BLOCK_WIDTH)
    bh_cAo, bh_cHo, bh_cVo, bh_cDo, kernel_time_ob = dwt_opt.dwt_gpu_optimized(bsig,filters, BLOCK_WIDTH)
    
    #concatenate combine kernel execution times to get a final average value for execution time across 2D dwts
    kernel_time = kernel_time_r + kernel_time_g + kernel_time_b
    temp_naive = temp_naive + kernel_time
    kernel_time_o = kernel_time_or + kernel_time_og + kernel_time_ob
    temp_o = temp_o + kernel_time_o
    if(np.mod((i+1),25)==0):
        times_naive.append(temp_naive/25)
        times_opt.append(temp_o/25)
        temp_naive = 0
        temp_o = 0

    #print outputs and timing results
    print('naive same as serial rc_A: {}'.format(np.allclose(rcA, rh_cA, atol=5e-7)))
    print('naive same as serial rc_H: {}'.format(np.allclose(rcH, rh_cH, atol=5e-7)))
    print('naive same as serial rc_V: {}'.format(np.allclose(rcV, rh_cV, atol=5e-7)))
    print('naive same as serial rc_D: {}'.format(np.allclose(rcD, rh_cD, atol=5e-7)))
    print('naive same as serial gc_A: {}'.format(np.allclose(gcA, gh_cA, atol=5e-7)))
    print('naive same as serial gc_H: {}'.format(np.allclose(gcH, gh_cH, atol=5e-7)))
    print('naive same as serial gc_V: {}'.format(np.allclose(gcV, gh_cV, atol=5e-7)))
    print('naive same as serial gc_D: {}'.format(np.allclose(gcD, gh_cD, atol=5e-7)))
    print('naive same as serial bc_A: {}'.format(np.allclose(bcA, bh_cA, atol=5e-7)))
    print('naive same as serial bc_H: {}'.format(np.allclose(bcH, bh_cH, atol=5e-7)))
    print('naive same as serial bc_V: {}'.format(np.allclose(bcV, bh_cV, atol=5e-7)))
    print('naive same as serial bc_D: {}'.format(np.allclose(bcD, bh_cD, atol=5e-7)))
    
    print('optimized same as serial rc_A: {}'.format(np.allclose(rcA, rh_cAo, atol=5e-7)))
    print('optimized same as serial rc_H: {}'.format(np.allclose(rcH, rh_cHo, atol=5e-7)))
    print('optimized same as serial rc_V: {}'.format(np.allclose(rcV, rh_cVo, atol=5e-7)))
    print('optimized same as serial rc_D: {}'.format(np.allclose(rcD, rh_cDo, atol=5e-7)))
    print('optimized same as serial gc_A: {}'.format(np.allclose(gcA, gh_cAo, atol=5e-7)))
    print('optimized same as serial gc_H: {}'.format(np.allclose(gcH, gh_cHo, atol=5e-7)))
    print('optimized same as serial gc_V: {}'.format(np.allclose(gcV, gh_cVo, atol=5e-7)))
    print('optimized same as serial gc_D: {}'.format(np.allclose(gcD, gh_cDo, atol=5e-7)))
    print('optimized same as serial bc_A: {}'.format(np.allclose(bcA, bh_cAo, atol=5e-7)))
    print('optimized same as serial bc_H: {}'.format(np.allclose(bcH, bh_cHo, atol=5e-7)))
    print('optimized same as serial bc_V: {}'.format(np.allclose(bcV, bh_cVo, atol=5e-7)))
    print('optimized same as serial bc_D: {}'.format(np.allclose(bcD, bh_cDo, atol=5e-7)))

    print('\nSerial time: {}'.format(serial_time))
    print('Naive time: {}'.format(kernel_time))
    print('Optimized time time: {}'.format(kernel_time_o))
    
plt.figure()
plt.title('Execution Time Comparison Graph')
plt.plot(sizes, times_naive, label='Naive')
plt.plot(sizes, times_opt, label='Optimized')
plt.plot(sizes, times_serial, label='Serial')
plt.xlabel('size')
plt.ylabel('run time/s')
plt.legend(loc='upper right')
plt.savefig('execution_times.png')
