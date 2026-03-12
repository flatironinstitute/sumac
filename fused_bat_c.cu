#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cooperative_groups.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <utility>

namespace cg = cooperative_groups;

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_F32(x)  TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

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

template<int... Is, class F>
__device__ __forceinline__ void static_for_impl(std::integer_sequence<int, Is...>, F&& f) {
  (f(std::integral_constant<int, Is>{}), ...);
}
template<int N, class F>
__device__ __forceinline__ void static_for(F&& f) {
  static_for_impl(std::make_integer_sequence<int, N>{}, (F&&)f);
}


template<int BK, int V, int STAGES>
__device__ __forceinline__ void issue_tile_float4(
    cg::thread_block cta,
    cuda::pipeline<cuda::thread_scope_block>& pipe,
    float4 As4[STAGES][BK][V],
    const float4* __restrict__ A4,
    int stage, int n0, int N,
    int D4_full)
{
  constexpr int tile_bytes  = BK * V * sizeof(float4);

  float4* smem_ptr = &As4[stage][0][0];

  if (n0 + BK <= N) {
    const float4* gmem_ptr = &A4[n0 * D4_full];

    cuda::memcpy_async(
        cta,
        smem_ptr,
        gmem_ptr,
        cuda::aligned_size_t<16>(tile_bytes),
        pipe);
  }
  else {
    const int full_elems = BK * D4_full;
    for (int idx = threadIdx.x; idx < full_elems; idx += blockDim.x) {
      const int k  = idx / D4_full;
      const int dv = idx - k * D4_full;
      const int n  = n0 + k;

      float4* s = &As4[stage][k][dv];
      if (n < N)
        *s = A4[n * D4_full + dv];
      else
        *s = make_float4(0.f,0.f,0.f,0.f);
    }
  }
}

template<int BK, int V>
__device__ __forceinline__ void issue_tile_float4_sync(
    float4 As4[BK][V],
    const float4* __restrict__ A4,
    int tid, int BM,
    int n0, int N,
    int D4_full)
{
  const int full_elems = BK * D4_full;
  for (int idx = tid; idx < full_elems; idx += BM) {
    const int k = idx / D4_full;
    const int dv = idx - k * D4_full;
    const int n = n0 + k;

    float4* smem_ptr = &As4[k][dv];
    if (n < N) {
      *smem_ptr = __ldg(&A4[n * D4_full + dv]);
    } else {
      *smem_ptr = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }

}


template<int BK, int V, int STAGES, int MS>
__global__ void relu_bat_c_fused_kernel_float4(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    const float* __restrict__ C,
    float* __restrict__ Y,       
    int N, int M, int D)
{
  
  static_assert(STAGES >= 2, "STAGES must be >= 2");
  cg::thread_block cta = cg::this_thread_block();

  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> shared_state;

  auto pipe = cuda::make_pipeline(cta, &shared_state);

  const int BM  = blockDim.x;
  const int tid = threadIdx.x;
  const int m   = blockIdx.x * MS * BM + tid;
  

  __shared__ alignas(16) float4 As4[STAGES][BK][V];
  __shared__ alignas(16) float4 Cs4[STAGES][BK][V];

  const float4* __restrict__ A4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(A, 16));
  const float4* __restrict__ B4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(B, 16));
  const float4* __restrict__ C4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(C, 16));
  float4* __restrict__ Y4       = reinterpret_cast<float4*>(__builtin_assume_aligned(Y, 16));

  float4 b[MS][V] = {0.f};
  float4 y[MS][V] = {0.f};

  float4 a0[V] = {0.f};
  float4 a1[V] = {0.f};
  float4 c0[V] = {0.f};
  float4 c1[V] = {0.f};

  static_for<MS>([&](auto J) {
    if (m + J*BM < M) {
      static_for<V>([&](auto I) {
        b[J][I] = B4[(m + J*BM) * (D/4) + I];
      });
    }
  });
  

  const int warm = (N + BK - 1) / BK;              
  const int prefetch_tiles = (warm < STAGES) ? warm : STAGES;

  for (int t = 0; t < STAGES; ++t) {
    const int n0 = t * BK;
    if (t < prefetch_tiles) {
      pipe.producer_acquire();
      issue_tile_float4<BK, V, STAGES>(cta, pipe, As4, A4, t, n0, N, D / 4);
      issue_tile_float4<BK, V, STAGES>(cta, pipe, Cs4, C4, t, n0, N, D / 4);
      pipe.producer_commit();
    }
  }
  
  for (int tile = 0, n0 = 0; n0 < N; ++tile, n0 += BK) {
    
    const int cur_stage = tile % STAGES;
    const int n_prefetch = n0 + STAGES * BK;

    pipe.consumer_wait();
    __syncthreads();

    static_for<V>([&](auto I) { 
      a0[I] = As4[cur_stage][0][I]; 
      c0[I] = Cs4[cur_stage][0][I];
    });
#pragma unroll
    for (int k = 0; k < BK; ++k) {

      if (k+1 < BK) {
        static_for<V>([&](auto I) { 
          a1[I] = As4[cur_stage][k+1][I];
          c1[I] = Cs4[cur_stage][k+1][I];
        });
      }

      static_for<MS>([&](auto J) {
        {
          float acc = 0.f;
          static_for<V>([&](auto I) {
            acc = dot_float4(b[J][I], a0[I], acc);
          });
          acc = fmaxf(acc, 0.f);
          static_for<V>([&](auto I) {
            y[J][I] = axpy_float4(acc, c0[I], y[J][I]);
          });
        } 
      });

      if (k+1 < BK) {
        static_for<V>([&](auto I) {
          a0[I] = a1[I];
          c0[I] = c1[I];
        });
      }
    }

    __syncthreads();
    pipe.consumer_release();

    if (n_prefetch < N) {
      pipe.producer_acquire();
      issue_tile_float4<BK, V, STAGES>(cta, pipe, As4, A4, cur_stage, n_prefetch, N, D/4);
      issue_tile_float4<BK, V, STAGES>(cta, pipe, Cs4, C4, cur_stage, n_prefetch, N, D/4);
      pipe.producer_commit();
    }
  }

  static_for<MS>([&](auto J) {
    if (m + J*BM < M) {
      static_for<V>([&](auto I) {
        Y4[(m + J*BM) * (D/4) + I] = y[J][I];
      });
    }
  });
  
}

template<int BK, int V, int MS>
__global__ void relu_bat_c_fused_kernel_float4_sync(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    const float* __restrict__ C,
    float* __restrict__ Y,       
    int N, int M, int D)
{

  const int BM  = blockDim.x;
  const int tid = threadIdx.x;
  const int m   = blockIdx.x * MS * BM + tid;
  

  __shared__ alignas(16) float4 As4[BK][V];
  __shared__ alignas(16) float4 Cs4[BK][V];

  const float4* __restrict__ A4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(A, 16));
  const float4* __restrict__ B4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(B, 16));
  const float4* __restrict__ C4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(C, 16));

  float4* __restrict__ Y4       = reinterpret_cast<float4*>(__builtin_assume_aligned(Y, 16));

  float4 b[MS][V] = {0.f};
  float4 y[MS][V] = {0.f};

  float4 a0[V] = {0.f};
  float4 a1[V] = {0.f};
  float4 c0[V] = {0.f};
  float4 c1[V] = {0.f};

  static_for<MS>([&](auto J) {
    if (m + J*BM < M) {
      static_for<V>([&](auto I) {
        b[J][I] = B4[(m + J*BM) * (D/4) + I];
      });
    }
  });
  
  for (int n0 = 0; n0 < N; n0 += BK) {

    issue_tile_float4_sync<BK,V>(As4, A4, tid, BM, n0, N, D / 4);
    issue_tile_float4_sync<BK,V>(Cs4, C4, tid, BM, n0, N, D / 4);
    __syncthreads();

    static_for<V>([&](auto I) { 
      a0[I] = As4[0][I]; 
      c0[I] = Cs4[0][I];
    });
#pragma unroll
    for (int k = 0; k < BK; ++k) {

      if (k+1 < BK) {
        static_for<V>([&](auto I) { 
          a1[I] = As4[k+1][I];
          c1[I] = Cs4[k+1][I];
        });
      }

      static_for<MS>([&](auto J) {
        {
          float acc = 0.f;
          static_for<V>([&](auto I) {
            acc = dot_float4(b[J][I], a0[I], acc);
          });
          acc = fmaxf(acc, 0.f);
          static_for<V>([&](auto I) {
            y[J][I] = axpy_float4(acc, c0[I], y[J][I]);
          });
        } 
      });

      if (k+1 < BK) {
        static_for<V>([&](auto I) {
          a0[I] = a1[I];
          c0[I] = c1[I];
        });
      }
    }

    __syncthreads();
  }

  static_for<MS>([&](auto J) {
    if (m + J*BM < M) {
      static_for<V>([&](auto I) {
        Y4[(m + J*BM) * (D/4) + I] = y[J][I];
      });
    }
  });
  
}


template<int... Xs, class F>
inline bool dispatch_from_values_impl(int value, std::integer_sequence<int, Xs...>, F&& f) {
  bool matched = false;
  ((value == Xs ? (f(std::integral_constant<int, Xs>{}), matched = true) : false) || ...);
  return matched;
}

template<int... Xs, class F>
inline void dispatch_from_values(int value, F&& f, const char* name) {
  const bool ok = dispatch_from_values_impl(value, std::integer_sequence<int, Xs...>{}, std::forward<F>(f));
  TORCH_CHECK(ok, "Unsupported ", name, "=", value);
}


template<int BK, int V, int STAGES, int MS>
inline void launch_relu_bat_c_fused(
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    at::Tensor& Y,
    int N, int M, int D)
{

  if constexpr (STAGES==1) {
    relu_bat_c_fused_kernel_float4_sync<BK, V, MS><<<grid, block, 0, stream>>>(
      (const float*)A.data_ptr<float>(),
      (const float*)B.data_ptr<float>(),
      (const float*)C.data_ptr<float>(),
      (float*)Y.data_ptr<float>(),
      N, M, D);
  } else {
    relu_bat_c_fused_kernel_float4<BK, V, STAGES, MS><<<grid, block, 0, stream>>>(
      (const float*)A.data_ptr<float>(),
      (const float*)B.data_ptr<float>(),
      (const float*)C.data_ptr<float>(),
      (float*)Y.data_ptr<float>(),
      N, M, D);
  }
  
}

inline void dispatch(
    int V_runtime,
    int num_stages,
    int BK,
    int MS_runtime,
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    at::Tensor& Y,
    int N, int M, int D)
{
  dispatch_from_values<4>(V_runtime, [&](auto Vc) {
    constexpr int V = decltype(Vc)::value;

    dispatch_from_values<1,2,3>(num_stages, [&](auto Sc) {
      constexpr int STAGES = decltype(Sc)::value;

      dispatch_from_values<16,32,64,128>(BK, [&](auto BKc) {
        constexpr int BK_ = decltype(BKc)::value;

        dispatch_from_values<1,2,4>(MS_runtime, [&](auto MSc) {
          constexpr int MS = decltype(MSc)::value;

          launch_relu_bat_c_fused<BK_, V, STAGES, MS>(
              grid, block, stream, A, B, C, Y, N, M, D);
        }, "MS");

      }, "BK");

    }, "num_stages");

  }, "V");
}

torch::Tensor relu_bat_c_fused_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    int64_t BM,
    int64_t BK,
    int64_t num_stages,
    int64_t num_ms = 1)
{
  CHECK_CUDA(A); CHECK_CUDA(B);
  CHECK_CUDA(C);
  CHECK_F32(A);  CHECK_F32(B);
  CHECK_F32(C);
  CHECK_CONTIGUOUS(A);
  CHECK_CONTIGUOUS(B);
  CHECK_CONTIGUOUS(C);

  TORCH_CHECK(A.dim() == 2, "A must be 2D Tensor");
  TORCH_CHECK(B.dim() == 2, "B must be 2D Tensor");
  TORCH_CHECK(C.dim() == 2, "C must be 2D Tensor");

  c10::cuda::CUDAGuard device_guard(A.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(A.device().index()).stream();

  const int N = A.size(0);
  const int M = B.size(0);
  const int D = A.size(1);
  const int V = D / 4;

  auto Y = torch::empty({M, D}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

  TORCH_CHECK(BM % 32 == 0, "BM must be multiple of 32");
  TORCH_CHECK(num_ms >= 1, "num_ms must be >= 1");

  const dim3 block(BM);
  const dim3 grid(ceil_div(M, num_ms * BM));

  dispatch(V, num_stages, BK, num_ms, grid, block, stream, A, B, C, Y, N, M, D);
  

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
  return Y;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relu_bat_c_fused_cuda", &relu_bat_c_fused_cuda, "relu_bat_c_fused_cuda (CUDA)");
}