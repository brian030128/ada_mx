#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

using e8m0_t = uint8_t;

template <int K_GLOBAL> 
__global__ 
void mxfp8_matmul_kernel(
    int M, int N,
    const __nv_fp8_e4m3* __restrict__ A,
    const __nv_fp8_e4m3* __restrict__ B,
    const e8m0_t* __restrict__ scale_a,
    const e8m0_t* __restrict__ scale_b,
    const half* __restrict__ C,
    half* __restrict__ D_out
) {

    

    
}




// Host function
torch::Tensor mxfp8_matmul(
    torch::Tensor A,        
    torch::Tensor B,
    torch::Tensor scale_a,
    torch::Tensor scale_b,
    torch::Tensor C
) {
    TORCH_CHECK(A.scalar_type() == torch::kFloat8_e4m3fn, "A must be FP8");
    TORCH_CHECK(B.scalar_type() == torch::kFloat8_e4m3fn, "B must be FP8");
    TORCH_CHECK(scale_a.scalar_type() == torch::kFloat8_e8m0fnu, "scale A must be FP8");
    TORCH_CHECK(scale_b.scalar_type() == torch::kFloat8_e8m0fnu, "scale B must be FP8");
    TORCH_CHECK(C.scalar_type() == torch::kFloat16, "C must be FP16");

    
    TORCH_CHECK(A.stride(1) == 1, "A must be row major")
    TORCH_CHECK(B.stride(0) == 1, "B must be column major")
    
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda() && scale_a.is_cuda() && scale_b.is_cuda(), "All input must be on CUDA");
    TORCH_CHECK(B.size(0) == A.size(1), "A and B must match size.");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && C.is_contiguous(), "All input must be contiguous.");
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto out = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));

    dim3 gridDim(CEIL_DIV(N, BLOCK_COL_SIZE), CEIL_DIV(M, BLOCK_ROW_SIZE));
    dim3 blockDim(TCORE_THREADS);

    
    switch(K) {
        case 1024:
            optimized_v8_kernel<1024><<<gridDim, blockDim>>>(
                M, N,
                reinterpret_cast<const __nv_fp8_e4m3*>(A.data_ptr()),
                reinterpret_cast<const __nv_fp8_e4m3*>(B.data_ptr()),
                reinterpret_cast<const e8m0_t*>(scale_a.data_ptr()),
                reinterpret_cast<const e8m0_t*>(scale_b.data_ptr()),
                reinterpret_cast<const half*>(C.data_ptr()),
                reinterpret_cast<half*>(out.data_ptr()));
            break;
        case 2048:
            optimized_v8_kernel<2048><<<gridDim, blockDim>>>(
                M, N,
                reinterpret_cast<const __nv_fp8_e4m3*>(A.data_ptr()),
                reinterpret_cast<const __nv_fp8_e4m3*>(B.data_ptr()),
                reinterpret_cast<const e8m0_t*>(scale_a.data_ptr()),
                reinterpret_cast<const e8m0_t*>(scale_b.data_ptr()),
                reinterpret_cast<const half*>(C.data_ptr()),
                reinterpret_cast<half*>(out.data_ptr()));
            break;
        default:
            TORCH_CHECK(false, "Unsupported K size. Add K=", K);
    }

    return D_out;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mxfp8_matmul", &mxfp8_matmul, "mxfp8_matmul");
}