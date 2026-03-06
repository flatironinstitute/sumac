#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <utility>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_F32(x)  TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ float4 load4_scalar(const float* p) {
  return make_float4(p[0], p[1], p[2], p[3]);
}
__device__ __forceinline__ float2 load2_scalar(const float* p) {
  return make_float2(p[0], p[1]);
}

template<int V, int STAGES, int BK>
struct SharedTileR1 {
  alignas(16) float4 As4[STAGES][BK][V];
  float  As1[STAGES][BK];
  __device__ float2* as2() { return nullptr; }
  __device__ float*  as1() { return &As1[0][0]; }
};

template<int V, int STAGES, int BK>
struct SharedTileR2 {
  alignas(16) float4 As4[STAGES][BK][V];
  alignas(8) float2 As2[STAGES][BK];
  __device__ float2* as2() { return &As2[0][0]; }
  __device__ float*  as1() { return nullptr; }
};

template<int V, int STAGES, int BK>
struct SharedTileR3 {
  alignas(16) float4 As4[STAGES][BK][V];
  alignas(8) float2 As2[STAGES][BK];
  float  As1[STAGES][BK];
  __device__ float2* as2() { return &As2[0][0]; }
  __device__ float*  as1() { return &As1[0][0]; }
};

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

__device__ __forceinline__ float dot_float2(const float2& A, const float2& B, const float c) {
  float r = fmaf(A.x, B.x, c);
  r = fmaf(A.y, B.y, r);
  return r;
}

__device__ __forceinline__ float2 axpy_float2(const float alpha, const float2& X, const float2& Y) {
  return make_float2(
    fmaf(alpha, X.x, Y.x),
    fmaf(alpha, X.y, Y.y)
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

template<int BK, int V, int STAGES, int R>
__device__ __forceinline__ void issue_tile_mixed(
    cuda::pipeline<cuda::thread_scope_thread>& pipe,
    float4 As4[STAGES][BK][V],
    float2* __restrict__ As2_flat, 
    float*  __restrict__ As1_flat, 
    const float* __restrict__ A,
    int tid, int BM,
    int stage, int n0, int N)
{
  constexpr bool HAS2 = (R >= 2);
  constexpr bool HAS1 = (R == 1 || R == 3);

  const int full_elems = BK * V;
  for (int idx = tid; idx < full_elems; idx += BM) {
    const int k  = idx / V;
    const int dv = idx - k * V;
    const int n  = n0 + k;

    float4* smem_ptr = &As4[stage][k][dv];

    if (n < N) {
      const float* row = A + n * (4*V + R); // D
      const float* p   = row + 4*dv;

      if (((uintptr_t)p & 0xF) == 0) {
        cuda::memcpy_async(
            smem_ptr,
            reinterpret_cast<const float4*>(p),
            cuda::aligned_size_t<16>(sizeof(float4)),
            pipe);
      } else {
        *smem_ptr = load4_scalar(p);
      }
    } else {
      *smem_ptr = make_float4(0.f,0.f,0.f,0.f);
    }
  }

  for (int k = tid; k < BK; k += BM) {
    const int n = n0 + k;
    if (n >= N) {
      if constexpr (HAS2) As2_flat[stage * BK + k] = make_float2(0.f, 0.f);
      if constexpr (HAS1) As1_flat[stage * BK + k] = 0.f;
      continue;
    }

    const float* row = A + n * (4*V + R);
    const float* tail = row + 4*V;

    if constexpr (HAS2) {
      if (((uintptr_t)tail & 0x7) == 0) {
        cuda::memcpy_async(
            &As2_flat[stage * BK + k],
            reinterpret_cast<const float2*>(tail),
            cuda::aligned_size_t<8>(sizeof(float2)),
            pipe);
      } else {
        As2_flat[stage * BK + k] = load2_scalar(tail);
      }
    }

    if constexpr (HAS1) {
      const int off = (HAS2 ? 2 : 0);
      As1_flat[stage * BK + k] = tail[off];
    }
  }
}


template<int BK, int V, int STAGES, int MS>
__global__ void relu_bat_a_fused_kernel_float4(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ Y,       
    int N, int M, int D)
{

  static_assert(STAGES >= 2, "STAGES must be >= 2");

  const int BM  = (int)blockDim.x;
  const int tid = (int)threadIdx.x;
  const int m   = (int)(blockIdx.x * MS * BM + tid);
  

  __shared__ alignas(16) float4 As4[STAGES][BK][V];

  const float4* __restrict__ A4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(A, 16));
  const float4* __restrict__ B4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(B, 16));
  float4* __restrict__ Y4       = reinterpret_cast<float4*>(__builtin_assume_aligned(Y, 16));

  float4 b[MS][V] = {0.f};
  float4 y[MS][V] = {0.f};

  float4 a0[V] = {0.f};
  float4 a1[V] = {0.f};

  static_for<MS>([&](auto J) {
    if (m + J*BM < M) {
      static_for<V>([&](auto I) {
        b[J][I] = B4[(m + J*BM) * (D/4) + (int)I];
      });
    }
  });
  
  auto pipe = cuda::make_pipeline();

  const int warm = (N + BK - 1) / BK;              
  const int prefetch_tiles = (warm < STAGES) ? warm : STAGES;

  for (int t = 0; t < STAGES; ++t) {
    const int n0 = t * BK;
    if (t < prefetch_tiles) {
      pipe.producer_acquire();
      issue_tile_float4<BK, V, STAGES>(pipe, As4, A4, tid, BM, t, n0, N, D / 4);
      pipe.producer_commit();
    }
  }
  
  for (int tile = 0, n0 = 0; n0 < N; ++tile, n0 += BK) {
    
    const int cur_stage = tile % STAGES;
    const int n_prefetch = n0 + STAGES * BK;

    pipe.consumer_wait();
    __syncthreads();

    static_for<V>([&](auto I) { 
      a0[I] = As4[cur_stage][0][(int)I]; 
    });
#pragma unroll
    for (int k = 0; k < BK; ++k) {

      if (k+1 < BK) {
        static_for<V>([&](auto I) { 
          a1[I] = As4[cur_stage][k+1][(int)I];
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
            y[J][I] = axpy_float4(acc, a0[I], y[J][I]);
          });
        } 
      });

      if (k+1 < BK) {
        static_for<V>([&](auto I) {
          a0[I] = a1[I];
        });
      }
    }

    __syncthreads();
    pipe.consumer_release();

    if (n_prefetch < N) {
      pipe.producer_acquire();
      issue_tile_float4<BK, V, STAGES>(pipe, As4, A4, tid, BM, cur_stage, n_prefetch, N, D/4);
      pipe.producer_commit();
    }
  }

  static_for<MS>([&](auto J) {
    if (m + J*BM < M) {
      static_for<V>([&](auto I) {
        Y4[(m + J*BM) * (D/4) + (int)I] = y[J][I];
      });
    }
  });
  
}

template<int BK, int V, int STAGES, int R, int MS>
__global__ void relu_bat_a_fused_kernel_mixed(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ Y,
    int N, int M)
{
  static_assert(STAGES >= 2);
  static_assert(R >= 0 && R <= 3);

  constexpr int D = 4*V + R;
  constexpr bool HAS2 = (R >= 2);
  constexpr bool HAS1 = (R == 1 || R == 3);

  const int BM  = (int)blockDim.x;
  const int tid = (int)threadIdx.x;
  const int m   = (int)(blockIdx.x * MS * BM + tid);

  using Smem =
  typename std::conditional<(R == 1),
    SharedTileR1<V, STAGES, BK>,
    typename std::conditional<(R == 2),
      SharedTileR2<V, STAGES, BK>,
      SharedTileR3<V, STAGES, BK>
    >::type
  >::type;

  __shared__ Smem smem;

  
  float4 b4[MS][V] = {0.f};
  float4 y4[MS][V] = {0.f};
  float4 a04[V] = {0.f};
  float4 a14[V] = {0.f};
  float2 b2[MS] = {0.f};
  float2 y2[MS] = {0.f};
  float2 a02, a12;
  float  b1[MS] = {0.f}, y1[MS] = {0.f};
  float a01, a11;

  static_for<MS>([&](auto J) {
    if (m + J*BM < M) {
      const float* Brow = B + (m + J*BM) * D;
      static_for<V>([&](auto I) {
        b4[J][I] = load4_scalar(Brow + 4*I);
      });
      if constexpr (HAS2) b2[J] = load2_scalar(Brow + 4*V);
      if constexpr (HAS1) b1[J] = Brow[4*V + (HAS2 ? 2 : 0)];
    }
  });

  auto pipe = cuda::make_pipeline();

  const int warm = (N + BK - 1) / BK;
  const int prefetch_tiles = (warm < STAGES) ? warm : STAGES;

  for (int t = 0; t < STAGES; ++t) {
    const int n0 = t * BK;
    if (t < prefetch_tiles) {
      pipe.producer_acquire();
      issue_tile_mixed<BK,V,STAGES,R>(
          pipe, smem.As4, smem.as2(), smem.as1(),
          A, tid, BM, t, n0, N);
      pipe.producer_commit();
    }
  }

  for (int tile = 0, n0 = 0; n0 < N; ++tile, n0 += BK) {
    const int cur_stage = tile % STAGES;
    const int n_prefetch = n0 + STAGES * BK;

    pipe.consumer_wait();
    __syncthreads();

    static_for<V>([&](auto I) { a04[I] = smem.As4[cur_stage][0][I]; });
    if constexpr (HAS2) a02 = smem.as2()[cur_stage * BK + 0];
    if constexpr (HAS1) a01 = smem.as1()[cur_stage * BK + 0];

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      if (k + 1 < BK) {
        static_for<V>([&](auto I) { a14[I] = smem.As4[cur_stage][k+1][I]; });
        if constexpr (HAS2) a12 = smem.as2()[cur_stage * BK + (k+1)];
        if constexpr (HAS1) a11 = smem.as1()[cur_stage * BK + (k+1)];
      }

      static_for<MS>([&](auto J) {
        float acc = 0.f;
        static_for<V>([&](auto I) { 
          acc = dot_float4(b4[J][I], a04[I], acc); 
        });
        if constexpr (HAS2) acc = dot_float2(b2[J], a02, acc);
        if constexpr (HAS1) acc = fmaf(b1[J], a01, acc);
        acc = fmaxf(acc, 0.f);

        static_for<V>([&](auto I) { 
          y4[J][I] = axpy_float4(acc, a04[I], y4[J][I]); 
        });
        if constexpr (HAS2) y2[J] = axpy_float2(acc, a02, y2[J]);
        if constexpr (HAS1) y1[J] = fmaf(acc, a01, y1[J]);

      });

      if (k + 1 < BK) {
        static_for<V>([&](auto I) {
          a04[I] = a14[I];
        });
        if constexpr (HAS2) a02 = a12;
        if constexpr (HAS1) a01 = a11;
      }
    }

    __syncthreads();
    pipe.consumer_release();

    if (n_prefetch < N) {
      pipe.producer_acquire();
      issue_tile_mixed<BK,V,STAGES,R>(
          pipe, smem.As4, smem.as2(), smem.as1(),
          A, tid, BM, cur_stage, n_prefetch, N);
      pipe.producer_commit();
    }
  }

  static_for<MS>([&](auto J) {
    if (m + J*BM < M) {
      float* Yrow = Y + (m + J*BM) * D;

      static_for<V>([&](auto I) {
        const float4 v = y4[J][I];
        Yrow[4*I + 0] = v.x; Yrow[4*I + 1] = v.y; Yrow[4*I + 2] = v.z; Yrow[4*I + 3] = v.w;
      });
      if constexpr (HAS2) { Yrow[4*V + 0] = y2[J].x; Yrow[4*V + 1] = y2[J].y; }
      if constexpr (HAS1) { Yrow[4*V + (HAS2 ? 2 : 0)] = y1[J]; }
    }
  });

  // if (m < M) {
  //   float* Yrow = Y + m * D;

  //   static_for<V>([&](auto I) {
  //     const float4 v = y4[I];
  //     Yrow[4*I + 0] = v.x; Yrow[4*I + 1] = v.y; Yrow[4*I + 2] = v.z; Yrow[4*I + 3] = v.w;
  //   });
  //   if constexpr (HAS2) { Yrow[4*V + 0] = y2.x; Yrow[4*V + 1] = y2.y; }
  //   if constexpr (HAS1) { Yrow[4*V + (HAS2 ? 2 : 0)] = y1; }
  // }
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
inline void launch_relu_bat_a_fused(
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& Y,
    int N, int M, int D)
{
  relu_bat_a_fused_kernel_float4<BK, V, STAGES, MS><<<grid, block, 0, stream>>>(
      (const float*)A.data_ptr<float>(),
      (const float*)B.data_ptr<float>(),
      (float*)Y.data_ptr<float>(),
      N, M, D);
}

template<int BK, int V, int STAGES, int R, int MS>
inline void launch_relu_bat_a_fused_r(
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& Y,
    int N, int M)
{
  relu_bat_a_fused_kernel_mixed<BK, V, STAGES, R, MS><<<grid, block, 0, stream>>>(
      (const float*)A.data_ptr<float>(),
      (const float*)B.data_ptr<float>(),
      (float*)Y.data_ptr<float>(),
      N, M);
}

inline void dispatch_main(
    int V_runtime,
    int num_stages,
    int BK,
    int MS_runtime,
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& Y,
    int N, int M, int D)
{
  dispatch_from_values<1,2,3,4>(V_runtime, [&](auto Vc) {
    constexpr int V = decltype(Vc)::value;

    dispatch_from_values<2,3,4>(num_stages, [&](auto Sc) {
      constexpr int STAGES = decltype(Sc)::value;

      dispatch_from_values<16,32,64,128>(BK, [&](auto BKc) {
        constexpr int BK_ = decltype(BKc)::value;

        dispatch_from_values<1,2,4>(MS_runtime, [&](auto MSc) {
          constexpr int MS = decltype(MSc)::value;

          launch_relu_bat_a_fused<BK_, V, STAGES, MS>(
              grid, block, stream, A, B, Y, N, M, D);
        }, "MS");

      }, "BK");

    }, "num_stages");

  }, "V");
}

inline void dispatch_main_r(
    int V_runtime,
    int R_runtime,
    int num_stages,
    int BK,
    int MS_runtime,
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& Y,
    int N, int M)
{
  dispatch_from_values<1,2,3>(R_runtime, [&](auto Rc) {
    constexpr int R = decltype(Rc)::value;

    dispatch_from_values<1,2,3,4>(V_runtime, [&](auto Vc) {
      constexpr int V = decltype(Vc)::value;

      dispatch_from_values<2,3,4>(num_stages, [&](auto Sc) {
        constexpr int STAGES = decltype(Sc)::value;

        dispatch_from_values<16,32,64,128>(BK, [&](auto BKc) {
          constexpr int BK_ = decltype(BKc)::value;

          dispatch_from_values<1,2,4>(MS_runtime, [&](auto MSc) {
            constexpr int MS = decltype(MSc)::value;

            launch_relu_bat_a_fused_r<BK_, V, STAGES, R, MS>(
                grid, block, stream, A, B, Y, N, M);
          }, "MS");

        }, "BK");

      }, "num_stages");

    }, "V");

  }, "R");
}


torch::Tensor relu_bat_a_fused_cuda(
    torch::Tensor A,
    torch::Tensor B,
    int64_t BM,
    int64_t BK,
    int64_t num_stages,
    int64_t num_ms = 1)
{
  CHECK_CUDA(A); CHECK_CUDA(B);
  CHECK_F32(A);  CHECK_F32(B);
  CHECK_CONTIGUOUS(A);
  CHECK_CONTIGUOUS(B);

  TORCH_CHECK(A.dim() == 2, "A must be 2D Tensor");
  TORCH_CHECK(B.dim() == 2, "B must be 2D Tensor");

  c10::cuda::CUDAGuard device_guard(A.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(A.device().index()).stream();

  const int N = (int)A.size(0);
  const int M = (int)B.size(0);
  const int D = (int)A.size(1);
  const int V = D / 4;
  const int R = D % 4;

  auto Y = torch::empty({M, D}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

  TORCH_CHECK(BM % 32 == 0, "BM must be multiple of 32");
  TORCH_CHECK(num_ms >= 1, "num_ms must be >= 1");

  const dim3 block(BM);
  const dim3 grid(ceil_div(M, (int)num_ms * (int)BM));

  if (R == 0) {
    dispatch_main(V, (int)num_stages, (int)BK, (int)num_ms,
                  grid, block, stream, A, B, Y, N, M, D);
  } else {
    dispatch_main_r(V, R, (int)num_stages, (int)BK, (int)num_ms,
                    grid, block, stream, A, B, Y, N, M);
  }

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
  return Y;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relu_bat_a_fused_cuda", &relu_bat_a_fused_cuda, "relu_bat_a_fused_cuda (CUDA)");
}