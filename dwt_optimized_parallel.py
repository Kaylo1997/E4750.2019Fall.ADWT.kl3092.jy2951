import matplotlib as mpl
mpl.use('agg')
import numpy as np
from pycuda import compiler, gpuarray, tools
import pycuda.driver as cuda
import time
import matplotlib.pyplot as plt

# Initialize the device
import pycuda.autoinit
plt.ioff()

def create2Dfilter(a,b,length):
        result = np.zeros((length,length), dtype=np.float32)
        for i in range(length):
            for j in range(length):
                result[j][i]=a[i]*b[j]
        return result

class DWT_optimized:
    def __init__(self):
        # Grid size should be (ceil((N + maskwidth - 1)/2), ceil((M + maskwidth - 1)/2)) for input image shape (M, N)
        # to avoid wasting threads
        
        self.dwt_forward_opt = """
       
        __global__ void w_kern_forward(float* input, float* c_a, float* c_h, float* c_v, float* c_d, float* LL, float* LH, float* HL, float* HH) {
            
            //define row and column indicies
            int Col = threadIdx.x + blockIdx.x*blockDim.x;
            int Row = threadIdx.y + blockIdx.y*blockDim.y;
            
            // Obtain the dimension of the problem
            // size of the mask, width and height of input image
            // int maskwidth: length of the filter (default is 10 for CDF9/7)
            // int H: number of rows for input (height) (equals to M)
            // int W: number of columns for input (width) (equals to N)
            int maskwidth = %(M)s;
            int H = %(H)s;
            int W = %(W)s;

            // Obtain half of the width and height
            int W_half = (W + maskwidth - 1)/2;
            int H_half = (H + maskwidth - 1)/2;

            if (Row < H_half && Col < W_half) {
            
                //get center for even length kernel: default is length 10 for CDF97
                int c;
                c = maskwidth/2 + 3;
                
                //perform zero padding
                float res_a = 0, res_h = 0, res_v = 0, res_d = 0;
                float val = 0;

                // Convolution with periodic boundaries extension. perform horizontal AND vertical convolution
                for (int y = 0; y < maskwidth; y++) {
                
                    //get vertical index for input image
                    int ty = Row * 2 - c + y;
                    
                    for (int x = 0; x < maskwidth; x++) {
                    
                        //get horizonal index for input image
                        int tx = Col*2 - c + x;
                        
                        //get kernel index
                        int keridx = (maskwidth-1-y)*maskwidth + (maskwidth-1 - x);
                        
                        // Apply the zero-padding via the conditional
                        if ((ty > -1) && (ty < H) && (tx > -1) && (tx < W)){
                            //perform convolution with filters and input image
                            val = input[ty*W + tx];
                            res_a += val * LL[keridx];
                            res_h += val * LH[keridx];
                            res_v += val * HL[keridx];
                            res_d += val * HH[keridx];
                        }
                    }
                }
                
                //output coeefficients
                c_a[Row* W_half + Col] = res_a;
                c_h[Row* W_half + Col] = res_h;
                c_v[Row* W_half + Col] = res_v;
                c_d[Row* W_half + Col] = res_d;
            }
        }
        """
    
    def dwt_gpu_optimized(self, h_input, filters, BLOCK_WIDTH):
        
        # Obtain the shape of the input matrix
        dim_M = h_input.shape[0]
        dim_N = h_input.shape[1]
        maskwidth = filters[0].shape[0]

        # Obtain the 1D filters
        filters = filters.astype(np.float32)
        h_filter_lo = filters[0, :]
        h_filter_hi = filters[1, :]
        
        # Create 2D filters from 1D filters
        LL = create2Dfilter(h_filter_lo, h_filter_lo, 10)
        LH = create2Dfilter(h_filter_lo, h_filter_hi, 10)
        HL = create2Dfilter(h_filter_hi, h_filter_lo, 10)
        HH = create2Dfilter(h_filter_hi, h_filter_hi, 10)
        
        # Compute the size of the output of the wavelet transform
        dim_R = int(np.ceil(((dim_M + maskwidth - 1)/2)))
        dim_C = int(np.ceil(((dim_N + maskwidth - 1)/2)))

        # Calculate the number of blocks
        # Note that final output has shape (M, N)
        BLOCK_X = int(np.ceil(dim_C / float(BLOCK_WIDTH)))
        BLOCK_Y = int(np.ceil(dim_R / float(BLOCK_WIDTH)))

        h_input = h_input.astype(np.float32)
        h_cA = np.zeros(shape=(dim_R, dim_C), dtype=np.float32)
        h_cH = np.zeros(shape=(dim_R, dim_C), dtype=np.float32)
        h_cV = np.zeros(shape=(dim_R, dim_C), dtype=np.float32)
        h_cD = np.zeros(shape=(dim_R, dim_C), dtype=np.float32)

        # Transfer data to device
        d_input = gpuarray.to_gpu(h_input)
        d_cA = gpuarray.to_gpu(h_cA)
        d_cH = gpuarray.to_gpu(h_cH)
        d_cV = gpuarray.to_gpu(h_cV)
        d_cD = gpuarray.to_gpu(h_cD)
        d_LL = gpuarray.to_gpu(LL)
        d_LH = gpuarray.to_gpu(LH)
        d_HL = gpuarray.to_gpu(HL)
        d_HH = gpuarray.to_gpu(HH)

        # Call kernel
        dwt_forward_optimized_kernel = self.dwt_forward_opt % {
            'M': maskwidth,
            'H': dim_M,
            'W': dim_N
        }

        # Call kernel function
        prg_dwt_forward_optimized = compiler.SourceModule(dwt_forward_optimized_kernel)
        dwt_forward_optimized = prg_dwt_forward_optimized.get_function("w_kern_forward")
        tic = cuda.Event()
        toc = cuda.Event()

        tic.record()
        dwt_forward_optimized(d_input, d_cA, d_cH, d_cV, d_cD, d_LL, d_LH, d_HL, d_HH,
                           block=(BLOCK_WIDTH, BLOCK_WIDTH, 1), grid=(BLOCK_X, BLOCK_Y, 1))
        toc.record()
        toc.synchronize()

        kernel_time = tic.time_till(toc)*1e-3
        h_cA = d_cA.get()
        h_cH = d_cH.get()
        h_cV = d_cV.get()
        h_cD = d_cD.get()

        return h_cA, h_cH, h_cV, h_cD, kernel_time
