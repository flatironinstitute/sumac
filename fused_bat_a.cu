#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_F32(x)  TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

constexpr int D  = 16;

__device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
  // Convert generic pointer to shared address space - I should rewrite this with modern cuda::memcpy_async at some point
  unsigned long long smem_u64;
  asm volatile("cvta.to.shared.u64 %0, %1;\n"
               : "=l"(smem_u64)
               : "l"(smem_ptr));

  unsigned smem_u32 = (unsigned)smem_u64;

  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
               :
               : "r"(smem_u32), "l"(gmem_ptr));
#else
  *reinterpret_cast<float4*>(smem_ptr) = *reinterpret_cast<const float4*>(gmem_ptr);
#endif
}


__device__ __forceinline__ void cp_async_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_all() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::);
#endif
}


template<int BK, int D>
__global__ void relu_bat_a_fused_kernel(
    const float* __restrict__ A,   
    const float* __restrict__ B,   
    float* __restrict__ Y,         
    int N, int M)
{
  const int BM  = (int)blockDim.x;
  const int tid = (int)threadIdx.x;
  const int m   = (int)(blockIdx.x * BM + tid);

  // Double-buffered shared A tiles
  __shared__ float4 As4[2][BK][D/4];

  const float4* __restrict__ A4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(A, 16));
  const float4* __restrict__ B4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(B, 16));
  float4* __restrict__ Y4 = reinterpret_cast<float4*>(__builtin_assume_aligned(Y, 16));

  float4 b0 = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 b1 = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 b2 = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 b3 = make_float4(0.f, 0.f, 0.f, 0.f);

  if (m < M) {
    const int row4 = m * (D / 4);   
    b0 = B4[row4 + 0];
    b1 = B4[row4 + 1];
    b2 = B4[row4 + 2];
    b3 = B4[row4 + 3];
  }

  float4 y0 = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 y1 = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 y2 = make_float4(0.f, 0.f, 0.f, 0.f);
  float4 y3 = make_float4(0.f, 0.f, 0.f, 0.f);

  auto issue_tile = [&](int stage, int n0) {
    const int tile_elems = BK * (D / 4);
    for (int idx = tid; idx < tile_elems; idx += BM) {
      const int k  = idx / (D / 4);          // 0..BK-1
      const int dv = idx - k * (D / 4);      // 0..3
      const int n  = n0 + k;

      float4* smem_ptr = &As4[stage][k][dv];

      if (n < N) {
        const float4* gmem_ptr = &A4[n * (D / 4) + dv];
        cp_async_16B((void*)smem_ptr, (const void*)gmem_ptr);
      } else {
        *smem_ptr = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
  };

  // prefetch tile 0 into stage 0
  int n0 = 0;
  if (n0 < N) {
    issue_tile(0, n0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
  }

  for (; n0 < N; n0 += BK) {

    const int cur_stage  = (n0 / BK) & 1;
    const int next_stage = cur_stage ^ 1;
    const int n1 = n0 + BK;

    // Prefetch next tile while computing current
    if (n1 < N) {
      issue_tile(next_stage, n1);
      cp_async_commit();
    }

    // Preload k=0
    float4 a0_0 = As4[cur_stage][0][0];
    float4 a0_1 = As4[cur_stage][0][1];
    float4 a0_2 = As4[cur_stage][0][2];
    float4 a0_3 = As4[cur_stage][0][3];

#pragma unroll
    for (int k = 0; k < BK - 1; ++k) {
      // Prefetch next k+1 early
      float4 a1_0 = As4[cur_stage][k + 1][0];
      float4 a1_1 = As4[cur_stage][k + 1][1];
      float4 a1_2 = As4[cur_stage][k + 1][2];
      float4 a1_3 = As4[cur_stage][k + 1][3];

      float acc = 0.f;
      float acc0 = 0.f;
      float acc1 = 0.f;

      acc0 = fmaf(b0.x, a0_0.x, acc0); 
      acc0 = fmaf(b0.y, a0_0.y, acc0);
      acc0 = fmaf(b0.z, a0_0.z, acc0); 
      acc0 = fmaf(b0.w, a0_0.w, acc0);

      acc1 = fmaf(b1.x, a0_1.x, acc1); 
      acc1 = fmaf(b1.y, a0_1.y, acc1);
      acc1 = fmaf(b1.z, a0_1.z, acc1); 
      acc1 = fmaf(b1.w, a0_1.w, acc1);

      acc0 = fmaf(b2.x, a0_2.x, acc0); 
      acc0 = fmaf(b2.y, a0_2.y, acc0);
      acc0 = fmaf(b2.z, a0_2.z, acc0); 
      acc0 = fmaf(b2.w, a0_2.w, acc0);

      acc1 = fmaf(b3.x, a0_3.x, acc1); 
      acc1 = fmaf(b3.y, a0_3.y, acc1);
      acc1 = fmaf(b3.z, a0_3.z, acc1); 
      acc1 = fmaf(b3.w, a0_3.w, acc1);

      acc = fmaxf(acc0 + acc1, 0.f);

      // y += Dot(ReLU(B A.T),A)
      y0.x = fmaf(acc, a0_0.x, y0.x); 
      y0.y = fmaf(acc, a0_0.y, y0.y);
      y0.z = fmaf(acc, a0_0.z, y0.z); 
      y0.w = fmaf(acc, a0_0.w, y0.w);

      y1.x = fmaf(acc, a0_1.x, y1.x); 
      y1.y = fmaf(acc, a0_1.y, y1.y);
      y1.z = fmaf(acc, a0_1.z, y1.z); 
      y1.w = fmaf(acc, a0_1.w, y1.w);

      y2.x = fmaf(acc, a0_2.x, y2.x); 
      y2.y = fmaf(acc, a0_2.y, y2.y);
      y2.z = fmaf(acc, a0_2.z, y2.z); 
      y2.w = fmaf(acc, a0_2.w, y2.w);

      y3.x = fmaf(acc, a0_3.x, y3.x); 
      y3.y = fmaf(acc, a0_3.y, y3.y);
      y3.z = fmaf(acc, a0_3.z, y3.z); 
      y3.w = fmaf(acc, a0_3.w, y3.w);

      // Advance pipeline
      a0_0 = a1_0; a0_1 = a1_1; a0_2 = a1_2; a0_3 = a1_3;
    }

    // Final iteration
    {
      float acc = 0.f;

      acc = fmaf(b0.x, a0_0.x, acc); acc = fmaf(b0.y, a0_0.y, acc);
      acc = fmaf(b0.z, a0_0.z, acc); acc = fmaf(b0.w, a0_0.w, acc);

      acc = fmaf(b1.x, a0_1.x, acc); acc = fmaf(b1.y, a0_1.y, acc);
      acc = fmaf(b1.z, a0_1.z, acc); acc = fmaf(b1.w, a0_1.w, acc);

      acc = fmaf(b2.x, a0_2.x, acc); acc = fmaf(b2.y, a0_2.y, acc);
      acc = fmaf(b2.z, a0_2.z, acc); acc = fmaf(b2.w, a0_2.w, acc);

      acc = fmaf(b3.x, a0_3.x, acc); acc = fmaf(b3.y, a0_3.y, acc);
      acc = fmaf(b3.z, a0_3.z, acc); acc = fmaf(b3.w, a0_3.w, acc);

      acc = max(acc, 0.f);

      y0.x = fmaf(acc, a0_0.x, y0.x); y0.y = fmaf(acc, a0_0.y, y0.y);
      y0.z = fmaf(acc, a0_0.z, y0.z); y0.w = fmaf(acc, a0_0.w, y0.w);

      y1.x = fmaf(acc, a0_1.x, y1.x); y1.y = fmaf(acc, a0_1.y, y1.y);
      y1.z = fmaf(acc, a0_1.z, y1.z); y1.w = fmaf(acc, a0_1.w, y1.w);

      y2.x = fmaf(acc, a0_2.x, y2.x); y2.y = fmaf(acc, a0_2.y, y2.y);
      y2.z = fmaf(acc, a0_2.z, y2.z); y2.w = fmaf(acc, a0_2.w, y2.w);

      y3.x = fmaf(acc, a0_3.x, y3.x); y3.y = fmaf(acc, a0_3.y, y3.y);
      y3.z = fmaf(acc, a0_3.z, y3.z); y3.w = fmaf(acc, a0_3.w, y3.w);
    }

    // Sync next tile before the next iteration reads it
    if (n1 < N) {
      cp_async_wait_all();
      __syncthreads();
    }
  }

  if (m < M) {
    const int row4 = m * (D / 4);
    Y4[row4 + 0] = y0;
    Y4[row4 + 1] = y1;
    Y4[row4 + 2] = y2;
    Y4[row4 + 3] = y3;
  }
}

torch::Tensor relu_bat_a_fused_cuda(torch::Tensor A, torch::Tensor B) {
  CHECK_CUDA(A); CHECK_CUDA(B);
  CHECK_F32(A);  CHECK_F32(B);
  CHECK_CONTIGUOUS(A);
  CHECK_CONTIGUOUS(B);

  TORCH_CHECK(A.dim() == 2 && A.size(1) == D, "A must be [N,16]");
  TORCH_CHECK(B.dim() == 2 && B.size(1) == D, "B must be [M,16]");

  const int N = (int)A.size(0);
  const int M = (int)B.size(0);

  auto Y = torch::empty({M, D}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

  const int BM = 256;
  TORCH_CHECK(BM % 32 == 0, "BM must be multiple of 32");
  const dim3 block(BM);
  const dim3 grid(ceil_div(M, BM));

  relu_bat_a_fused_kernel<64,16><<<grid, block, 0>>>(
      (const float*)A.data_ptr<float>(),
      (const float*)B.data_ptr<float>(),
      (float*)Y.data_ptr<float>(),
      N, M);

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
  return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relu_bat_a_fused_cuda", &relu_bat_a_fused_cuda, "relu_bat_a_fused_cuda (CUDA)");
}