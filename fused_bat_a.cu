#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_F32(x)  TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

constexpr int D  = 16;

__device__ __forceinline__ float dot_float4(const float4& A, const float4& B, const float c) {
  float r = fmaf(A.x, B.x, c);
  r = fmaf(A.y, B.y, r);
  r = fmaf(A.z, B.z, r);
  r = fmaf(A.w, B.w, r);
  return r;
}

__device__ __forceinline__ float4 axpy_float4(const float alpha, const float4& X, const float4& Y) {
  return make_float4(
    fmaf(alpha, X.x, Y.x),
    fmaf(alpha, X.y, Y.y),
    fmaf(alpha, X.z, Y.z),
    fmaf(alpha, X.w, Y.w)
  );
}

template<int BK, int D>
__device__ __forceinline__ void issue_tile(
    cuda::pipeline<cuda::thread_scope_thread>& pipe,
    float4 As4[2][BK][D/4],
    const float4* __restrict__ A4,
    int tid, int BM,
    int stage, int n0, int N)
{
  constexpr int VEC = D / 4;
  const int tile_elems = BK * VEC;

  for (int idx = tid; idx < tile_elems; idx += BM) {
    const int k  = idx / VEC;
    const int dv = idx - k * VEC;
    const int n  = n0 + k;

    float4* smem_ptr = &As4[stage][k][dv];

    if (n < N) {
      const float4* gmem_ptr = &A4[n * VEC + dv];
      cuda::memcpy_async(smem_ptr, gmem_ptr, cuda::aligned_size_t<16>(sizeof(float4)), pipe);
    } else {
      *smem_ptr = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
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

  __shared__ alignas(16) float4 As4[2][BK][D/4];

  const float4* __restrict__ A4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(A, 16));
  const float4* __restrict__ B4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(B, 16));
  float4* __restrict__ Y4       = reinterpret_cast<float4*>(__builtin_assume_aligned(Y, 16));

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

  auto pipe = cuda::make_pipeline();

  pipe.producer_acquire();
  issue_tile<BK, D>(pipe, As4, A4, tid, BM, 0, 0, N);
  pipe.producer_commit();
  
  if (BK < N) {
    pipe.producer_acquire();
    issue_tile<BK, D>(pipe, As4, A4, tid, BM, 1, BK, N);
    pipe.producer_commit();
  }


  for (int n0 = 0; n0 < N; n0 += BK) {
    const int tile = n0/BK;
    const int cur_stage  = tile & 1;
    const int n_prefetch = n0 + 2 * BK;
    const int prefetch_stage = cur_stage;

    pipe.consumer_wait();
    __syncthreads();

    float4 a0_0 = As4[cur_stage][0][0];
    float4 a0_1 = As4[cur_stage][0][1];
    float4 a0_2 = As4[cur_stage][0][2];
    float4 a0_3 = As4[cur_stage][0][3];

#pragma unroll
    for (int k = 0; k < BK - 1; ++k) {
      float4 a1_0 = As4[cur_stage][k + 1][0];
      float4 a1_1 = As4[cur_stage][k + 1][1];
      float4 a1_2 = As4[cur_stage][k + 1][2];
      float4 a1_3 = As4[cur_stage][k + 1][3];

      float acc0 = 0.f, acc1 = 0.f;

      acc0 = dot_float4(b0, a0_0, acc0);
      acc1 = dot_float4(b1, a0_1, acc1);
      acc0 = dot_float4(b2, a0_2, acc0);
      acc1 = dot_float4(b3, a0_3, acc1);

      const float acc = fmaxf(acc0 + acc1, 0.f);

      y0 = axpy_float4(acc, a0_0, y0);      
      y1 = axpy_float4(acc, a0_1, y1);
      y2 = axpy_float4(acc, a0_2, y2);
      y3 = axpy_float4(acc, a0_3, y3);

      a0_0 = a1_0; a0_1 = a1_1; a0_2 = a1_2; a0_3 = a1_3;
    }

    {
      float acc0 = 0.f, acc1 = 0.f;

      acc0 = dot_float4(b0, a0_0, acc0);
      acc1 = dot_float4(b1, a0_1, acc1);
      acc0 = dot_float4(b2, a0_2, acc0);
      acc1 = dot_float4(b3, a0_3, acc1);

      const float acc = fmaxf(acc0 + acc1, 0.f);

      y0 = axpy_float4(acc, a0_0, y0);
      y1 = axpy_float4(acc, a0_1, y1);
      y2 = axpy_float4(acc, a0_2, y2);
      y3 = axpy_float4(acc, a0_3, y3);
    }
    __syncthreads();
    pipe.consumer_release();

    if (n_prefetch < N) {
      pipe.producer_acquire();
      issue_tile<BK, D>(pipe, As4, A4, tid, BM, prefetch_stage, n_prefetch, N);
      pipe.producer_commit();
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