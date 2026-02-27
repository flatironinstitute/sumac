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
using Vec4Registers = typename Vec4RegsFromSeq<std::make_integer_sequence<int, V>>::type;

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


template<int BK, int V, int STAGES>
__global__ void relu_bat_a_fused_kernel_float4(
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

  Vec4Registers<V> b, y, a0, a1;
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
      issue_tile_float4<BK, V, STAGES>(pipe, As4, A4, tid, BM, cur_stage, n_prefetch, N, D/4);
      pipe.producer_commit();
    }
  }

  if (m < M) {
    static_for<V>([&](auto I) {
      Y4[m * (D/4) + (int)I] = y.template get<I>();
    });
  }
}

template<int BK, int V, int STAGES, int R>
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
  const int m   = (int)(blockIdx.x * BM + tid);

  using Smem =
  typename std::conditional<(R == 1),
    SharedTileR1<V, STAGES, BK>,
    typename std::conditional<(R == 2),
      SharedTileR2<V, STAGES, BK>,
      SharedTileR3<V, STAGES, BK>
    >::type
  >::type;

  __shared__ Smem smem;

  
  Vec4Registers<V> b4, y4, a04, a14;
  float2 b2 = make_float2(0.f,0.f), y2 = make_float2(0.f,0.f), a02, a12;
  float  b1 = 0.f, y1 = 0.f, a01, a11;

  static_for<V>([&](auto I) {
    b4.template get<I>() = make_float4(0.f,0.f,0.f,0.f);
    y4.template get<I>() = make_float4(0.f,0.f,0.f,0.f);
  });

  
  if (m < M) {
    const float* Brow = B + m * D;

    static_for<V>([&](auto I) {
      constexpr int i = (int)I;
      b4.template get<I>() = load4_scalar(Brow + 4*i);
    });

    if constexpr (HAS2) b2 = load2_scalar(Brow + 4*V);
    if constexpr (HAS1) b1 = Brow[4*V + (HAS2 ? 2 : 0)];
  }

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

    static_for<V>([&](auto I) { a04.template get<I>() = smem.As4[cur_stage][0][(int)I]; });
    if constexpr (HAS2) a02 = smem.as2()[cur_stage * BK + 0];
    if constexpr (HAS1) a01 = smem.as1()[cur_stage * BK + 0];

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      if (k + 1 < BK) {
        static_for<V>([&](auto I) { a14.template get<I>() = smem.As4[cur_stage][k+1][(int)I]; });
        if constexpr (HAS2) a12 = smem.as2()[cur_stage * BK + (k+1)];
        if constexpr (HAS1) a11 = smem.as1()[cur_stage * BK + (k+1)];
      }

      float acc = 0.f;
      static_for<V>([&](auto I) { acc = dot_float4(b4.template get<I>(), a04.template get<I>(), acc); });
      if constexpr (HAS2) acc = dot_float2(b2, a02, acc);
      if constexpr (HAS1) acc = fmaf(b1, a01, acc);
      acc = fmaxf(acc, 0.f);

      static_for<V>([&](auto I) { y4.template get<I>() = axpy_float4(acc, a04.template get<I>(), y4.template get<I>()); });
      if constexpr (HAS2) y2 = axpy_float2(acc, a02, y2);
      if constexpr (HAS1) y1 = fmaf(acc, a01, y1);

      if (k + 1 < BK) {
        a04 = a14;
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

  if (m < M) {
    float* Yrow = Y + m * D;

    static_for<V>([&](auto I) {
      constexpr int i = (int)I;
      const float4 v = y4.template get<I>();
      Yrow[4*i + 0] = v.x; Yrow[4*i + 1] = v.y; Yrow[4*i + 2] = v.z; Yrow[4*i + 3] = v.w;
    });
    if constexpr (HAS2) { Yrow[4*V + 0] = y2.x; Yrow[4*V + 1] = y2.y; }
    if constexpr (HAS1) { Yrow[4*V + (HAS2 ? 2 : 0)] = y1; }
  }
}


template<int BK, int V, int STAGES>
inline void launch_relu_bat_a_fused(
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& Y,
    int N, int M, int D)
{
  relu_bat_a_fused_kernel_float4<BK, V, STAGES> <<<grid, block, 0, stream>>>(
      (const float*)A.data_ptr<float>(),
      (const float*)B.data_ptr<float>(),
      (float*)Y.data_ptr<float>(),
      N, M, D);
}

template<int BK, int V, int STAGES, int R>
inline void launch_relu_bat_a_fused_r(
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& Y,
    int N, int M)
{
  relu_bat_a_fused_kernel_mixed<BK, V, STAGES, R> <<<grid, block, 0, stream>>>(
      (const float*)A.data_ptr<float>(),
      (const float*)B.data_ptr<float>(),
      (float*)Y.data_ptr<float>(),
      N, M);
}

template<int V, int STAGES>
inline void dispatch_bk(
    int BK,
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& Y,
    int N, int M, int D)
{
  switch (BK) {
    case 16:  return launch_relu_bat_a_fused<16,  V, STAGES>(grid, block, stream, A, B, Y, N, M, D);
    case 32:  return launch_relu_bat_a_fused<32,  V, STAGES>(grid, block, stream, A, B, Y, N, M, D);
    case 64:  return launch_relu_bat_a_fused<64,  V, STAGES>(grid, block, stream, A, B, Y, N, M, D);
    case 128: return launch_relu_bat_a_fused<128, V, STAGES>(grid, block, stream, A, B, Y, N, M, D);
    default:
      TORCH_CHECK(false, "Unsupported BK=", BK, " (expected 16/32/64/128)");
  }
}

template<int V>
inline void dispatch_stages(
    int num_stages,
    int BK,
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& Y,
    int N, int M, int D)
{
  switch (num_stages) {
    case 2: return dispatch_bk<V, 2>(BK, grid, block, stream, A, B, Y, N, M, D);
    case 3: return dispatch_bk<V, 3>(BK, grid, block, stream, A, B, Y, N, M, D);
    default:
      TORCH_CHECK(false, "Unsupported num_stages=", num_stages, " (expected 2 or 3)");
  }
}

template<int V, int STAGES, int R>
inline void dispatch_bk_r(
    int BK,
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& Y,
    int N, int M)
{
  switch (BK) {
    case 16:  return launch_relu_bat_a_fused_r<16,  V, STAGES, R>(grid, block, stream, A, B, Y, N, M);
    case 32:  return launch_relu_bat_a_fused_r<32,  V, STAGES, R>(grid, block, stream, A, B, Y, N, M);
    case 64:  return launch_relu_bat_a_fused_r<64,  V, STAGES, R>(grid, block, stream, A, B, Y, N, M);
    case 128: return launch_relu_bat_a_fused_r<128, V, STAGES, R>(grid, block, stream, A, B, Y, N, M);
    default:
      TORCH_CHECK(false, "Unsupported BK=", BK, " (expected 16/32/64/128)");
  }
}

template<int V, int R>
inline void dispatch_stages_r(
    int num_stages,
    int BK,
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& Y,
    int N, int M)
{
  switch (num_stages) {
    case 2: return dispatch_bk_r<V, 2, R>(BK, grid, block, stream, A, B, Y, N, M);
    case 3: return dispatch_bk_r<V, 3, R>(BK, grid, block, stream, A, B, Y, N, M);
    default:
      TORCH_CHECK(false, "Unsupported num_stages=", num_stages, " (expected 2 or 3)");
  }
}



inline void dispatch_v(
    int V_runtime,
    int num_stages,
    int BK,
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& Y,
    int N, int M, int D)
{
  switch (V_runtime) {
    case 1:  return dispatch_stages<1>(num_stages, BK, grid, block, stream, A, B, Y, N, M, D);
    case 2:  return dispatch_stages<2>(num_stages, BK, grid, block, stream, A, B, Y, N, M, D);
    case 3:  return dispatch_stages<3>(num_stages, BK, grid, block, stream, A, B, Y, N, M, D);
    case 4:  return dispatch_stages<4>(num_stages, BK, grid, block, stream, A, B, Y, N, M, D);
    // case 5:  return dispatch_stages<5>(num_stages, BK, grid, block, stream, A, B, Y, N, M, D);
    // case 6:  return dispatch_stages<6>(num_stages, BK, grid, block, stream, A, B, Y, N, M, D);
    // case 7:  return dispatch_stages<7>(num_stages, BK, grid, block, stream, A, B, Y, N, M, D);
    // case 8:  return dispatch_stages<8>(num_stages, BK, grid, block, stream, A, B, Y, N, M, D);
    // case 16:  return dispatch_stages<16>(num_stages, BK, grid, block, stream, A, B, Y, N, M, D);
    default:
      TORCH_CHECK(false, "Unsupported V=", V_runtime);
  }
} 
  
template<int R>
inline void dispatch_v_r_fixedR(
    int V_runtime,
    int num_stages,
    int BK,
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& Y,
    int N, int M)
{
  switch (V_runtime) {
    case 1:  return dispatch_stages_r<1,  R>(num_stages, BK, grid, block, stream, A, B, Y, N, M);
    case 2:  return dispatch_stages_r<2,  R>(num_stages, BK, grid, block, stream, A, B, Y, N, M);
    case 3:  return dispatch_stages_r<3,  R>(num_stages, BK, grid, block, stream, A, B, Y, N, M);
    case 4:  return dispatch_stages_r<4,  R>(num_stages, BK, grid, block, stream, A, B, Y, N, M);
    // case 5:  return dispatch_stages_r<5,  R>(num_stages, BK, grid, block, A, B, Y, N, M);
    // case 6:  return dispatch_stages_r<6,  R>(num_stages, BK, grid, block, A, B, Y, N, M);
    // case 7:  return dispatch_stages_r<7,  R>(num_stages, BK, grid, block, A, B, Y, N, M);
    // case 8:  return dispatch_stages_r<8,  R>(num_stages, BK, grid, block, A, B, Y, N, M);
    // case 16: return dispatch_stages_r<16, R>(num_stages, BK, grid, block, A, B, Y, N, M);
    default:
      TORCH_CHECK(false, "Unsupported V=", V_runtime);
  }
}

inline void dispatch_vr(
    int V_runtime,
    int R_runtime,
    int num_stages,
    int BK,
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& Y,
    int N, int M)
{
  TORCH_CHECK(R_runtime != 0, "dispatch_vr called with R==0");
  TORCH_CHECK(0 <= R_runtime && R_runtime <= 3, "R must be in {1,2,3}, got ", R_runtime);

  switch (R_runtime) {
    case 1: return dispatch_v_r_fixedR<1>(V_runtime, num_stages, BK, grid, block, stream, A, B, Y, N, M);
    case 2: return dispatch_v_r_fixedR<2>(V_runtime, num_stages, BK, grid, block, stream, A, B, Y, N, M);
    case 3: return dispatch_v_r_fixedR<3>(V_runtime, num_stages, BK, grid, block, stream, A, B, Y, N, M);
    default:
      TORCH_CHECK(false, "Unreachable R=", R_runtime);
  }
}

torch::Tensor relu_bat_a_fused_cuda(torch::Tensor A, torch::Tensor B, int64_t BM, int64_t BK, int64_t num_stages) {
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
  const dim3 block(BM);
  const dim3 grid(ceil_div(M, BM));
  if (R == 0) {
    dispatch_v(V, num_stages, BK, grid, block, stream, A, B, Y, N, M, D);
  } else {
    dispatch_vr(V, R, num_stages, BK, grid, block, stream, A, B, Y, N, M);
  }
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
  return Y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relu_bat_a_fused_cuda", &relu_bat_a_fused_cuda, "relu_bat_a_fused_cuda (CUDA)");
}