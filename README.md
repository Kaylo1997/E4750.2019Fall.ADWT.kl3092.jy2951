# E4750.2019Fall.ADWT.kl3092.jy2951
Authors: 
Kaylo Littlejohn, kl3092@columbia.edu;
Desmond Yao, jy2951@columbia.edu

Supported under the MIT license.

## Description
This directory contains the scripts for accelerated Discrete Wavelet Transform (DWT)  project for EECS E4750 
Heterogeneous Computing. In this package, we created the serial implementation of 2D DWT using the `pywavelets` package 
as well as multiple parallel  kernels in the CUDA language using both the separable scheme and non-separable scheme for 
computing 2D DWT. In  particular, in the case of the separable scheme, we also created a version of the parallel
kernel that utilizes shared memory. The script currently supports the Cohen-Daubechies–Feauveau 9/7 (CDF9/7) wavelet
using zero-padding mode. 

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
numpy
matplotlib
Pillow
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
computing scripts, runs 2D DWT on a random array generated by `numpy` and tests the equality of the generated wavelet 
coefficients as well as displaying the execution time for all four methods.

### Benchmark Scripts
The two files `benchmark_actual_image.py` and `benchmark_random_signal.py` are the benchmarking scripts that generated
the results section for our report.

The script `benchmark_random_signal.py` uses `numpy` package to generate a random array of a base size specified by the 
user. It then iteratively scales the array up and runs all four DWT computing methods on the array and records the runtime
taken for all methods as the size of the array increases. For the parallel kernels, the block size used is specified by
the user. Finally, the program plots the time taken for all four methods and for all three parallel kernels separately
and stores the image in the `Results` folder.

The script `benchmark_actual_image.py` takes in images stored in the `images` folder, which contains images of various
sizes and different shapes, and runs all four DWT computing scripts on images of each shape separately to calculate and 
compare the cumulative time taken for each script to execute. It then plots the cumulative time taken and saves it in
the `Results` folder.

### Running the Sript
To generate the benchmark results, then first move the entire folder into the tesseract server. Then `cd` to the directory
of the scripts and simply run `sbatch --gres=gpu:1 --time=6 --wrap="nvprof python benchmark_random_signal.py"`
for the random signal results once you have configured the parameters you wish to test or run `sbatch --gres=gpu:1 --time=6 --wrap="nvprof python benchmark_actual_image.py"`
if you wish to generate the benchmark results for actual images. 

## License
MIT License

Copyright (c) [2019] [Kaylo Littlejohn, Jiaang Yao]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.