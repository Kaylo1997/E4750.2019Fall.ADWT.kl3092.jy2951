import matplotlib as mpl
mpl.use('agg')
import numpy as np
from pycuda import compiler, gpuarray, tools
import pycuda.driver as cuda
import time
import matplotlib.pyplot as plt

class DWT_naive:
    def __init__(self):
        self.dwt_naive_template = """
        """

    def dwt_gpu_naive(self, h_input):
        # Obtain the shape of the input matrix
        dim_M = h_input.shape[0]
        dim_N = h_input.shape[1]

        # Set the width of the block
        BLOCK_WIDTH = 16

        # Calculate the number of blocks
        # Note that final output has shape (M, N)
        BLOCK_X = int(np.ceil(dim_N / float(BLOCK_WIDTH)))
        BLOCK_Y = int(np.ceil(dim_M / float(BLOCK_WIDTH)))

        h_input = h_input.astype(np.float32)
        h_out = np.zeros(shape=(dim_M, dim_N), dtype=np.float32)

        # Transfer data to device
        d_input = gpuarray.to_gpu(h_input)
        d_out = gpuarray.to_gpu(h_out)

        # Call kernel
        dwt_naive = self.dwt_naive_template % {

        }

        # Call kernel function
        prg_dwt_naive = compiler.SourceModule(dwt_naive)
        dwtNaive = prg_dwt_naive.get_function("convolution_2D_naive_kernel")
        tic = cuda.Event()
        toc = cuda.Event()

        tic.record()
        dwtNaive(d_input, d_out, block=(BLOCK_WIDTH, BLOCK_WIDTH, 1), grid=(BLOCK_X, BLOCK_Y, 1))
        toc.record()
        toc.synchronize()

        kernel_time = tic.time_till(toc)*1e-3
        h_out = d_out.get()

        return h_out, kernel_time

