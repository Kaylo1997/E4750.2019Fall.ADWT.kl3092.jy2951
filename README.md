# E4750.2019Fall.ADWT.kl3092.jy2951
Authors: 
Kaylo Littlejohn, kl3092@columbia.edu;
Desmond Yao, jy2951@columbia.edu

## Description
This directory contains the scripts for accelerated Discrete Wavelet Transform (DWT)  project for EECS E4750 
Heterogeneous Computing. In this package, we created the serial implementation of 2D DWT using the `pywavelets` package 
as well as multiple parallel  kernels in the CUDA language using both the separable scheme and non-separable scheme for 
computing 2D DWT. In  particular, in the case of the separable scheme, we also created a version of the parallel
kernel that utilizes shared memory. 

## Installation
All that is required is to clone the repository from github from  `https://github.com/Kaylo1997/eecs4750project` or 
download the entire repository as a zip file. 

### Key Dependencies
#### CUDA
To execute the script, CUDA 10.0 needs to be installed on the computer. Instructions for installing CUDA can be found at 
`https://developer.nvidia.com/cuda-toolkit`

#### Python
The entire program is written using the Python wrapper for CUDA known as `pycuda`, which runs on `Python 2.7`. Therefore,
the following Python version and packages are required for executing the script correctly
```
Python==2.7
pywt
pycuda
```

## Usage
The entire script is organized in the following way.
```
E4750.2019Fall.ADWT.kl3092.jy2951
├──AproximateImage
├──images
|   ├──rect_tall
|   ├──rect_wide
|   ├──square
├──Results
├──benchmark_actual_image.py
├──benchmark_random_signal.py
├──dwt_naive_separable_parallel.py
├──dwt_nonseparable_parallel.py
├──dwt_serial.py
├──dwt_tiled_separable_parallel.py
├──readme.md
├──test_gen_approx_image.py
└──test_random_signal.py
```

### DWT Computing Scripts
The four files `dwt_serial.py`, `dwt_naive_separable_parallel.py`, `dwt_tiled_separable_parallel.py` and 
`dwt_nonseparable_parallel.py` contains the scripts that calculate the 2D DWT and record the runtime. All the DWT methods
are encapsulated as class methods for each of the separable class. 

In particular,
`dwt_serial.py` contains the serial implementation using `pywavelets`, `dwt_naive_separable_parallel.py` contains the
naive separable parallel kernel using CUDA, `dwt_tiled_separable_parallel.py` contains the tiled separable parallel 
kernel using CUDA and `dwt_nonseparable_parallel.py` contains the non-separable parallel kernel using CUDA.

### Testing Scripts
The two files `test_gen_approx_image.py` and `test_random_signal.py` contain some testing functionality. 

The script in `test_gen_approx_image.py` generates the grayscale image and approximate grayscale image using 2D DWT and
IDWT for an image that's stored in `ApproximateImage`. The script in `test_random_signal.py` calls all four DWT
computing scripts and 