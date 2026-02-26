#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <utility>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_F32(x)  TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

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


template<int I>
struct Lane4 { float4 v; };

template<int... Is>
struct Vec4RegsImpl : Lane4<Is>... {
  template<int I>
  __device__ __forceinline__ float4& get() {
    return static_cast<Lane4<I>&>(*this).v;
  }
  template<int I>
  __device__ __forceinline__ const float4& get() const {
    return static_cast<const Lane4<I>&>(*this).v;
  }
};

template<class Seq> struct Vec4RegsFromSeq;
template<int... Is>
struct Vec4RegsFromSeq<std::integer_sequence<int, Is...>> { using type = Vec4RegsImpl<Is...>; };

template<int V>
using Vec4Regs = typename Vec4RegsFromSeq<std::make_integer_sequence<int, V>>::type;

template<int... Is, class F>
__device__ __forceinline__ void static_for_impl(std::integer_sequence<int, Is...>, F&& f) {
  (f(std::integral_constant<int, Is>{}), ...);
}
template<int N, class F>
__device__ __forceinline__ void static_for(F&& f) {
  static_for_impl(std::make_integer_sequence<int, N>{}, (F&&)f);
}

template<int BK, int V, int STAGES>
__device__ __forceinline__ void issue_tile(
    cuda::pipeline<cuda::thread_scope_thread>& pipe,
    float4 As4[STAGES][BK][V],
    const float4* __restrict__ A4,
    int tid, int BM,
    int stage, int n0, int N,
    int D4_full)
{
  const int full_elems = BK * D4_full;
  for (int idx = tid; idx < full_elems; idx += BM) {
    const int k  = idx / D4_full;
    const int dv = idx - k * D4_full;
    const int n  = n0 + k;

    float4* smem_ptr = &As4[stage][k][dv];
    if (n < N) {
      const float4* gmem_ptr = &A4[n * D4_full + dv];
      cuda::memcpy_async(
          smem_ptr, gmem_ptr,
          cuda::aligned_size_t<16>(sizeof(float4)),
          pipe);
    } else {
      *smem_ptr = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
}



template<int BK, int V, int STAGES>
__global__ void relu_bat_a_fused_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ Y,       
    int N, int M, int D)
{

  static_assert(STAGES >= 2, "STAGES must be >= 2");

  const int BM  = (int)blockDim.x;
  const int tid = (int)threadIdx.x;
  const int m   = (int)(blockIdx.x * BM + tid);
  

  __shared__ alignas(16) float4 As4[STAGES][BK][V];

  const float4* __restrict__ A4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(A, 16));
  const float4* __restrict__ B4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(B, 16));
  float4* __restrict__ Y4       = reinterpret_cast<float4*>(__builtin_assume_aligned(Y, 16));

  Vec4Regs<V> b, y, a0, a1;
  static_for<V>([&](auto I) {
    b.template get<I>() = make_float4(0.f,0.f,0.f,0.f);
    y.template get<I>() = make_float4(0.f,0.f,0.f,0.f);
  });

  if (m < M) {
    static_for<V>([&](auto I) { b.template get<I>() = B4[m * (D/4) + (int)I]; });
  }

  auto pipe = cuda::make_pipeline();

  const int warm = (N + BK - 1) / BK;              
  const int prefetch_tiles = (warm < STAGES) ? warm : STAGES;

  for (int t = 0; t < STAGES; ++t) {
    const int n0 = t * BK;
    if (t < prefetch_tiles) {
      pipe.producer_acquire();
      issue_tile<BK, V, STAGES>(pipe, As4, A4, tid, BM, t, n0, N, D / 4);
      pipe.producer_commit();
    }
  }
  
  for (int tile = 0, n0 = 0; n0 < N; ++tile, n0 += BK) {
    
    const int cur_stage = tile % STAGES;
    const int n_prefetch = n0 + STAGES * BK;

    pipe.consumer_wait();
    __syncthreads();

    static_for<V>([&](auto I) { 
      a0.template get<I>() = As4[cur_stage][0][(int)I]; 
    });
#pragma unroll
    for (int k = 0; k < BK; ++k) {

      if (k+1 < BK) {
        static_for<V>([&](auto I) { 
          a1.template get<I>() = As4[cur_stage][k+1][(int)I];
        });
      }

      float acc0 = 0.f; float acc1 = 0.f;
      { 
        constexpr int V0 = (V + 1) / 2;   
        constexpr int V1 = V - V0;      
        static_for<V0>([&](auto I) {
          acc0 = dot_float4(b.template get<I>(), a0.template get<I>(), acc0);
        });
        static_for<V1>([&](auto I) {
          acc1 = dot_float4(b.template get<I+V0>(), a0.template get<I+V0>(), acc1);
        });
      }
      const float acc = fmaxf(acc0 + acc1, 0.f);

      static_for<V>([&](auto I) {
        y.template get<I>() = axpy_float4(acc, a0.template get<I>(), y.template get<I>());
      });
      if (k+1 < BK) {
        a0 = a1;
      }
    }

    __syncthreads();
    pipe.consumer_release();

    if (n_prefetch < N) {
      pipe.producer_acquire();
      issue_tile<BK, V, STAGES>(pipe, As4, A4, tid, BM, cur_stage, n_prefetch, N, D/4);
      pipe.producer_commit();
    }
  }

  if (m < M) {
    static_for<V>([&](auto I) {
      Y4[m * (D/4) + (int)I] = y.template get<I>();
    });
  }
}

// 4 Float4 Y's
// 4 Float4 B's
// 2x4 Float4 A's -> 4*4+4*4+2*4*4 = 64 registers minimum


template<int BK, int V, int STAGES>
inline void launch_relu_bat_a_fused(
    dim3 grid, dim3 block,
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& Y,
    int N, int M, int D)
{
  relu_bat_a_fused_kernel<BK, V, STAGES><<<grid, block, 0>>>(
      (const float*)A.data_ptr<float>(),
      (const float*)B.data_ptr<float>(),
      (float*)Y.data_ptr<float>(),
      N, M, D);
}

template<int V, int STAGES>
inline void dispatch_bk(
    int BK,
    dim3 grid, dim3 block,
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& Y,
    int N, int M, int D)
{
  switch (BK) {
    case 16:  return launch_relu_bat_a_fused<16,  V, STAGES>(grid, block, A, B, Y, N, M, D);
    case 32:  return launch_relu_bat_a_fused<32,  V, STAGES>(grid, block, A, B, Y, N, M, D);
    case 64:  return launch_relu_bat_a_fused<64,  V, STAGES>(grid, block, A, B, Y, N, M, D);
    case 128: return launch_relu_bat_a_fused<128, V, STAGES>(grid, block, A, B, Y, N, M, D);
    default:
      TORCH_CHECK(false, "Unsupported BK=", BK, " (expected 16/32/64/128)");
  }
}

template<int V>
inline void dispatch_stages(
    int num_stages,
    int BK,
    dim3 grid, dim3 block,
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& Y,
    int N, int M, int D)
{
  switch (num_stages) {
    case 2: return dispatch_bk<V, 2>(BK, grid, block, A, B, Y, N, M, D);
    case 3: return dispatch_bk<V, 3>(BK, grid, block, A, B, Y, N, M, D);
    default:
      TORCH_CHECK(false, "Unsupported num_stages=", num_stages, " (expected 2 or 3)");
  }
}

inline void dispatch_v(
    int V_runtime,
    int num_stages,
    int BK,
    dim3 grid, dim3 block,
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& Y,
    int N, int M, int D)
{
  switch (V_runtime) {
    case 1:  return dispatch_stages<1>(num_stages, BK, grid, block, A, B, Y, N, M, D);
    case 2:  return dispatch_stages<2>(num_stages, BK, grid, block, A, B, Y, N, M, D);
    case 3:  return dispatch_stages<3>(num_stages, BK, grid, block, A, B, Y, N, M, D);
    case 4:  return dispatch_stages<4>(num_stages, BK, grid, block, A, B, Y, N, M, D);
    case 5:  return dispatch_stages<5>(num_stages, BK, grid, block, A, B, Y, N, M, D);
    case 6:  return dispatch_stages<6>(num_stages, BK, grid, block, A, B, Y, N, M, D);
    case 7:  return dispatch_stages<7>(num_stages, BK, grid, block, A, B, Y, N, M, D);
    case 8:  return dispatch_stages<8>(num_stages, BK, grid, block, A, B, Y, N, M, D);
    case 16:  return dispatch_stages<16>(num_stages, BK, grid, block, A, B, Y, N, M, D);
    default:
      TORCH_CHECK(false, "Unsupported V=", V_runtime);
  }
} 
  

torch::Tensor relu_bat_a_fused_cuda(torch::Tensor A, torch::Tensor B, int64_t BM, int64_t BK, int64_t num_stages) {
  CHECK_CUDA(A); CHECK_CUDA(B);
  CHECK_F32(A);  CHECK_F32(B);
  CHECK_CONTIGUOUS(A);
  CHECK_CONTIGUOUS(B);

  TORCH_CHECK(A.dim() == 2, "A must be 2D Tensor");
  TORCH_CHECK(B.dim() == 2, "B must be 2D Tensor");

  const int N = (int)A.size(0);
  const int M = (int)B.size(0);
  const int D = (int)A.size(1);
  const int V = D / 4;
  auto Y = torch::empty({M, D}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

  TORCH_CHECK(BM % 32 == 0, "BM must be multiple of 32");
  const dim3 block(BM);
  const dim3 grid(ceil_div(M, BM));

  dispatch_v(V, num_stages, BK, grid, block, A, B, Y, N, M, D);

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
  return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relu_bat_a_fused_cuda", &relu_bat_a_fused_cuda, "relu_bat_a_fused_cuda (CUDA)");
}