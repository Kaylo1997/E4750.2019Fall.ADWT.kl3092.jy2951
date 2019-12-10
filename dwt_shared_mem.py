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

class DWT_optimized_shared_mem:
    def __init__(self):
        # Grid size should be (ceil((N + maskwidth - 1)/2), M) for input image shape (M, N) to avoid wasting threads
        # and to make indexing work
        
        self.dwt_forward1_opt= """
        __global__ void w_kernel_forward1(float* input, float* tmp_a1, float* tmp_a2, float* filter_lo, float* filter_hi){
            // params:
            // float* input: input image of shape (M, N)
            // float* tmp_a1: subband 1 subject to second forward pass
            // float* tmp_a2: subband 2 subject to second forward pass
            // float* filter_lo: LPF coefficients for approximation of shape (hlen,)
            // float* filter_hi: HPF coefficients for detail of shape (hlen,)
            
            // Obtain the thread idx along the rows and columns
            int ty = threadIdx.y;
            int tx = threadIdx.x;
            
            // Obtain the dimension of the problem
            // size of the mask, width and height of input image
            // int maskwidth: length of the filter (default is 10 for CDF9/7)
            // int O_TILE_WIDTH: length of the output tile per block
            // int H: number of rows for input (height) (equals to M)
            // int W: number of columns for input (width) (equals to N)
            int H = %(H)s;
            int W = %(W)s;
            #define O_TILE_WIDTH %(T)s
            #define maskwidth %(M)s

            int Row = threadIdx.y + blockIdx.y*blockDim.y;
            int Col_o = threadIdx.x + blockIdx.x*O_TILE_WIDTH;
            int Col_i = threadIdx.x + blockIdx.x*blockDim.x - (maskwidth - 2) * (1 + blockIdx.x);

            
            // Define the shared memory variable
            __shared__ float ds_in [2 * (O_TILE_WIDTH - 1) + maskwidth][2 * (O_TILE_WIDTH - 1) + maskwidth];
            
            // Obtain half of the width
            int W_half = (W + maskwidth - 1)/2;
            
            if ((Col_i > -1) && (Col_i < W)){
                ds_in[ty][tx] = input[Row * W + Col_i];
            }
            else{
                ds_in[ty][tx] = 0.0f;
            }
            
            // Wait for all the threads to load data into shared memory
            __syncthreads();
            
            float res_tmp_a1 = 0, res_tmp_a2 = 0;
            if ((ty < 2 * (O_TILE_WIDTH - 1) + maskwidth) && (tx < O_TILE_WIDTH)){
                for (int j = 0; j < maskwidth; j++) {
                    int kerIdx = maskwidth - j - 1;

                    // 1D Convolution with zeropadding boundary constraints
                    // Convolution is performed along each row

                    // Perform the convolution with both filters
                    res_tmp_a1 += ds_in[ty][2 * tx + j] * filter_lo[kerIdx];
                    res_tmp_a2 += ds_in[ty][2 * tx + j] * filter_hi[kerIdx];
                }
                
                if ((Row < H) && (Col_o < W_half)){
                    tmp_a1[Row * W_half + Col_o] = res_tmp_a1;
                    tmp_a2[Row * W_half + Col_o] = res_tmp_a2;
                }
            }
        }
        """

        # Grid size should be (ceil((N + maskwidth - 1)/2), ceil((M + maskwidth - 1)/2)) for input image shape (M, N)
        # to avoid wasting threads
        self.dwt_forward2_opt = """
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

            // Obtain the thread idx along the rows and columns
            int ty = threadIdx.y;
            int tx = threadIdx.x;

            // Obtain the dimension of the problem
            // size of the mask, width and height of input image
            // int maskwidth: length of the filter (default is 10 for CDF9/7)
            // int O_TILE_WIDTH: length of the output tile per block
            // int H: number of rows for input (height) (equals to M)
            // int W: number of columns for input (width) (equals to N)
            int H = %(H)s;
            int W = %(W)s;
            #define O_TILE_WIDTH %(T)s
            #define maskwidth %(M)s
            
            // Obtain the rows and the columns
            int Row_o = threadIdx.y + blockIdx.y*O_TILE_WIDTH;
            int Row_i = threadIdx.y + blockIdx.y*blockDim.y - (maskwidth - 2) * (1 + blockIdx.y);
            int Col = threadIdx.x + blockIdx.x*blockDim.x;
            int H_half = (H + maskwidth - 1)/2;
            int W_half = (W + maskwidth - 1)/2;

            //note the allocated memory is non-square since our convolution is only being applied horizontally
            // Define the shared memory variable
            __shared__ float ds_tmp_a1 [2 * (O_TILE_WIDTH - 1) + maskwidth][2 * (O_TILE_WIDTH - 1) + maskwidth];
            __shared__ float ds_tmp_a2 [2 * (O_TILE_WIDTH - 1) + maskwidth][2 * (O_TILE_WIDTH - 1) + maskwidth];
            
            if ((Row_i > -1) && (Row_i < H)){
                ds_tmp_a1[ty][tx] = tmp_a1[Row_i * W_half + Col];
                ds_tmp_a2[ty][tx] = tmp_a2[Row_i * W_half + Col];
            }
            else{
                ds_tmp_a1[ty][tx] = 0.0f;
                ds_tmp_a2[ty][tx] = 0.0f;
            }
            
            // Wait for all the threads to load data into shared memory
            __syncthreads();
            
            // 1D Convolution with zeropadding boundary constraints
            // Convolution is performed along each row
            float res_a = 0, res_h = 0, res_v = 0, res_d = 0;
            if ((tx < 2 * (O_TILE_WIDTH - 1) + maskwidth) && (ty < O_TILE_WIDTH)){
                for (int i = 0; i < maskwidth; i++) {
                    int kerIdx = maskwidth - i - 1;

                    // 1D Convolution with zeropadding boundary constraints
                    // Convolution is performed along each row

                    // Perform the convolution with both filters
                    res_a += ds_tmp_a1[2 * ty + i][tx] * filter_lo[kerIdx];
                    res_h += ds_tmp_a1[2 * ty + i][tx] * filter_hi[kerIdx];
                    res_v += ds_tmp_a2[2 * ty + i][tx] * filter_lo[kerIdx];
                    res_d += ds_tmp_a2[2 * ty + i][tx] * filter_hi[kerIdx];
                }
                
                if ((Row_o < H_half) && (Col < W_half)){
                    c_a[Row_o * W_half + Col] = res_a;
                    c_h[Row_o * W_half + Col] = res_h;
                    c_v[Row_o * W_half + Col] = res_v;
                    c_d[Row_o * W_half + Col] = res_d;
                }
            }
        }
        """

    def dwt_gpu_optimized(self, h_input, filters, BLOCK_WIDTH):
        
        # Obtain the shape of the input matrix
        dim_M = h_input.shape[0]
        dim_N = h_input.shape[1]
        maskwidth = filters[0].shape[0]

        # Obtain the filters
        filters = filters.astype(np.float32)
        h_filter_lo = filters[0, :]
        h_fitler_hi = filters[1, :]

        # Compute the size of the output of the wavelet transform
        dim_R = int(np.ceil(((dim_M + maskwidth - 1)/2)))
        dim_C = int(np.ceil(((dim_N + maskwidth - 1)/2)))

        # Set the width of the output tiles
        O_TILE_WIDTH = (BLOCK_WIDTH - maskwidth)/2 + 1

        # Calculate the number of blocks
        # Note that final output has shape (M, N)
        BLOCK_X1 = int(np.ceil(dim_C / float(O_TILE_WIDTH)))
        BLOCK_Y1 = int(np.ceil(dim_M / float(BLOCK_WIDTH)))

        BLOCK_X2 = int(np.ceil(dim_C / float(BLOCK_WIDTH)))
        BLOCK_Y2 = int(np.ceil(dim_R / float(O_TILE_WIDTH)))

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
        dwt_forward1_optimized_kernel = self.dwt_forward1_opt % {
            'M': maskwidth,
            'T': O_TILE_WIDTH,
            'H': dim_M,
            'W': dim_N
        }

        dwt_forward2_optimized_kernel = self.dwt_forward2_opt % {
            'M': maskwidth,
            'T': O_TILE_WIDTH,
            'H': dim_M,
            'W': dim_N
        }

        # Call kernel function
        prg_dwt_forward1_optimized = compiler.SourceModule(dwt_forward1_optimized_kernel)
        prg_dwt_forward2_optimized = compiler.SourceModule(dwt_forward2_optimized_kernel)
        dwt_forward1_optimized = prg_dwt_forward1_optimized.get_function("w_kernel_forward1")
        dwt_forward2_optimized = prg_dwt_forward2_optimized.get_function("w_kernel_forward2")
        tic = cuda.Event()
        toc = cuda.Event()

        tic.record()
        dwt_forward1_optimized(d_input, d_tmp_a1, d_tmp_a2, d_filter_lo, d_filter_hi,
                           block=(BLOCK_WIDTH, BLOCK_WIDTH, 1), grid=(BLOCK_X1, BLOCK_Y1, 1))
        dwt_forward2_optimized(d_tmp_a1, d_tmp_a2, d_cA, d_cH, d_cV, d_cD, d_filter_lo, d_filter_hi,
                           block=(BLOCK_WIDTH, BLOCK_WIDTH, 1), grid=(BLOCK_X2, BLOCK_Y2, 1))
        toc.record()
        toc.synchronize()

        kernel_time = tic.time_till(toc)*1e-3
        h_cA = d_cA.get()
        h_cH = d_cH.get()
        h_cV = d_cV.get()
        h_cD = d_cD.get()

        return h_cA, h_cH, h_cV, h_cD, kernel_time
