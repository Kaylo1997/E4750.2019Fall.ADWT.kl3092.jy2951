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

class DWT_naive_separable:
    def __init__(self):

        # Grid size should be (ceil((N + maskwidth - 1)/2), M) for input image shape (M, N) to avoid wasting threads
        # and to make indexing work
        self.dwt_forward1 = """
        __global__ void w_kernel_forward1(float* input, float* tmp_a1, float* tmp_a2, float* filter_lo, float* filter_hi){
            // params:
            // float* input: input image of shape (M, N)
            // float* tmp_a1: subband 1 subject to second forward pass
            // float* tmp_a2: subband 2 subject to second forward pass
            // float* filter_lo: LPF coefficients for approximation of shape (hlen,)
            // float* filter_hi: HPF coefficients for detail of shape (hlen,)
            
            // Obtain the global indices
            int Row = threadIdx.y + blockIdx.y*blockDim.y;
            int Col = threadIdx.x + blockIdx.x*blockDim.x;
            
            // Obtain the dimension of the problem
            // size of the mask, width and height of input image
            // int maskwidth: length of the filter (default is 10 for CDF9/7)
            // int H: number of rows for input (height) (equals to M)
            // int W: number of columns for input (width) (equals to N)
            int maskwidth = %(M)s;
            int H = %(H)s;
            int W = %(W)s;

            // Obtain half of the width
            int W_half = (W + maskwidth - 1)/2;
            
            // 1D Convolution with zeropadding boundary constraints and convolution is performed along each row
            // Perform vertical downsampling by half (separable method for DWT) and output is of shape 
            // (M, ceil((N + maskwidth - 1)/2))
            
            // Boundary condition check: If row index is less than M and column index is less than ceil((N + maskwidth - 1)/2)
            if (Row < H && Col < W_half){
                
                // Create variables for the output
                float res_tmp_a1 = 0, res_tmp_a2 = 0;
                
                // Obtain the index corresponding to the start of the filter
                // Note the downsampling via multiplication with 2 and (maskwidth - 2) is the amount of
                //  zero-padding (halo cells) that was padded to each side of the array
                int N_start_col = Col * 2 - (maskwidth - 2);
                
                // Loop through all elements in the filter
                for (int j = 0; j < maskwidth; j++) {
                
                    // Obtain the index of the current element
                    int curCol = N_start_col + j;
                    int kerIdx = maskwidth - j - 1;
                    
                    // Boundary condition check: if column index is less than the input width then add to the
                    // placeholder variable
                    if ((curCol > -1) && (curCol < W)){
                        // Perform the convolution with both filters
                        res_tmp_a1 += input[Row * W + curCol] * filter_lo[kerIdx];
                        res_tmp_a2 += input[Row * W + curCol] * filter_hi[kerIdx];
                    }
                }
                
                // Write the value to the output array
                tmp_a1[Row * W_half + Col] = res_tmp_a1;
                tmp_a2[Row * W_half + Col] = res_tmp_a2;
            }
        }
        """
        # Grid size should be (ceil((N + maskwidth - 1)/2), ceil((M + maskwidth - 1)/2)) for input image shape (M, N)
        # to avoid wasting threads
        self.dwt_forward2 = """
        __global__ void w_kernel_forward2(float* tmp_a1, float* tmp_a2, float* c_a, float* c_h, float* c_v, float* c_d, float* filter_lo, float* filter_hi){
            // params:
            // float* tmp_a1: subband 1 subject to second forward pass
            // float* tmp_a2: subband 2 subject to second forward pass
            // float* c_a: approximation coefficients, shape (ceil((M + maskwidth - 1)/2), ceil((N + maskwidth - 1)/2))
            // float* c_h: horizontal detail coefficients, shape (ceil((M + maskwidth - 1)/2), ceil((N + maskwidth - 1)/2))
            // float* c_v: vertical detail coefficients, shape (ceil((M + maskwidth - 1)/2), ceil((N + maskwidth - 1)/2))
            // float* c_d: diagonal detail coefficients, shape (ceil((M + maskwidth - 1)/2), ceil((N + maskwidth - 1)/2))
            // float* filter_lo: LPF coefficients for approximation of shape (hlen,)
            // float* filter_hi: HPF coefficients for detail of shape (hlen,)
            
            // Obtain the global indices
            int Row = threadIdx.y + blockIdx.y*blockDim.y;
            int Col = threadIdx.x + blockIdx.x*blockDim.x;

            // Obtain the dimension of the problem
            // size of the mask, width and height of input image
            // int maskwidth: length of the filter (default is 10 for CDF9/7)
            // int H: number of rows for input (height) (equals to M)
            // int W: number of columns for input (width) (equals to N)
            int maskwidth = %(M)s;
            int H = %(H)s;
            int W = %(W)s;

            // Obtain half of the height and width
            int H_half = (H + maskwidth - 1)/2;
            int W_half = (W + maskwidth - 1)/2;

            // 1D Convolution with zeropadding boundary constraints and convolution is performed along each row
            // Perform vertical downsampling by half (separable method for DWT) and output is of shape 
            // (ceil((M + maskwidth - 1)/2), ceil((N + maskwidth - 1)/2))
            
            // Boundary condition check: If row index is less than ceil((M + maskwidth - 1)/2) and 
            // column index is less than ceil((N + maskwidth - 1)/2)
            if (Row < H_half && Col < W_half){

                // Create variables for the output
                float res_a = 0, res_h = 0, res_v = 0, res_d = 0;

                // Obtain the index corresponding to the start of the filter
                // Note the downsampling via multiplication with 2 and (maskwidth - 2) is the amount of
                //  zero-padding (halo cells) that was padded to each side of the array
                int N_start_row = Row * 2 - (maskwidth - 2);
                
                // Loop through all elements in the filter
                for (int i = 0; i < maskwidth; i++) {
                    // Obtain the index of the current element
                    int curRow = N_start_row + i;
                    int kerIdx = maskwidth - i - 1;

                    // Boundary condition check: if row index is less than the input height then add to the
                    // placeholder variable
                    if ((curRow > -1) && (curRow < H)){
                        // Perform the convolution with both filters using both subbands
                        res_a += tmp_a1[curRow * W_half + Col] * filter_lo[kerIdx];
                        res_h += tmp_a1[curRow * W_half + Col] * filter_hi[kerIdx];
                        res_v += tmp_a2[curRow * W_half + Col] * filter_lo[kerIdx];
                        res_d += tmp_a2[curRow * W_half + Col] * filter_hi[kerIdx];
                    }
                }
                
                // Write the value to the output array
                c_a[Row * W_half + Col] = res_a;
                c_h[Row * W_half + Col] = res_h;
                c_v[Row * W_half + Col] = res_v;
                c_d[Row * W_half + Col] = res_d;
            }
        }
        """

    def dwt_gpu_naive_separable(self, h_input, filters, BLOCK_WIDTH):
        # Obtain the shape of the input matrix
        dim_M = h_input.shape[0]
        dim_N = h_input.shape[1]
        maskwidth = filters[0].shape[0]

        # Obtain the filters for DWT
        filters = filters.astype(np.float32)
        h_filter_lo = filters[0, :]
        h_fitler_hi = filters[1, :]

        # Compute the size of the output of the wavelet transform
        dim_R = int(np.ceil(((dim_M + maskwidth - 1)/2)))
        dim_C = int(np.ceil(((dim_N + maskwidth - 1)/2)))

        # Calculate the number of blocks
        # Note that final output has shape (R, C)
        BLOCK_X = int(np.ceil(dim_C / float(BLOCK_WIDTH)))
        BLOCK_Y1 = int(np.ceil(dim_M / float(BLOCK_WIDTH)))
        BLOCK_Y2 = int(np.ceil(dim_R / float(BLOCK_WIDTH)))

        # Create the various empty arrays on host and type conversion for input
        h_input = h_input.astype(np.float32)
        h_tmp_a1 = np.zeros(shape=(dim_M, dim_C), dtype=np.float32)
        h_tmp_a2 = np.zeros(shape=(dim_M, dim_C), dtype=np.float32)

        h_cA = np.zeros(shape=(dim_R, dim_C), dtype=np.float32)
        h_cH = np.zeros(shape=(dim_R, dim_C), dtype=np.float32)
        h_cV = np.zeros(shape=(dim_R, dim_C), dtype=np.float32)
        h_cD = np.zeros(shape=(dim_R, dim_C), dtype=np.float32)

        # Transfer data to device
        d_input = gpuarray.to_gpu(h_input)
        d_tmp_a1 = gpuarray.to_gpu(h_tmp_a1)
        d_tmp_a2 = gpuarray.to_gpu(h_tmp_a2)

        d_cA = gpuarray.to_gpu(h_cA)
        d_cH = gpuarray.to_gpu(h_cH)
        d_cV = gpuarray.to_gpu(h_cV)
        d_cD = gpuarray.to_gpu(h_cD)

        d_filter_lo = gpuarray.to_gpu(h_filter_lo)
        d_filter_hi = gpuarray.to_gpu(h_fitler_hi)

        # Call kernel
        dwt_forward1_naive_kernel = self.dwt_forward1 % {
            'M': maskwidth,
            'H': dim_M,
            'W': dim_N
        }

        dwt_forward2_naive_kernel = self.dwt_forward2 % {
            'M': maskwidth,
            'H': dim_M,
            'W': dim_N
        }

        # Call kernel function
        prg_dwt_forward1_naive = compiler.SourceModule(dwt_forward1_naive_kernel)
        prg_dwt_forward2_naive = compiler.SourceModule(dwt_forward2_naive_kernel)
        dwt_forward1_naive = prg_dwt_forward1_naive.get_function("w_kernel_forward1")
        dwt_forward2_naive = prg_dwt_forward2_naive.get_function("w_kernel_forward2")
        tic = cuda.Event()
        toc = cuda.Event()

        # Execute the kernels and record time taken for both kernels to complete their tasks
        tic.record()
        dwt_forward1_naive(d_input, d_tmp_a1, d_tmp_a2, d_filter_lo, d_filter_hi,
                           block=(BLOCK_WIDTH, BLOCK_WIDTH, 1), grid=(BLOCK_X, BLOCK_Y1, 1))
        dwt_forward2_naive(d_tmp_a1, d_tmp_a2, d_cA, d_cH, d_cV, d_cD, d_filter_lo, d_filter_hi,
                           block=(BLOCK_WIDTH, BLOCK_WIDTH, 1), grid=(BLOCK_X, BLOCK_Y2, 1))
        toc.record()
        toc.synchronize()

        # Obtain the outputs
        kernel_time = tic.time_till(toc)*1e-3
        h_cA = d_cA.get()
        h_cH = d_cH.get()
        h_cV = d_cV.get()
        h_cD = d_cD.get()

        return h_cA, h_cH, h_cV, h_cD, kernel_time




