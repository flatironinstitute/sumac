#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <cooperative_groups.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <utility>

#pragma nv_diag_suppress static_var_with_dynamic_init

namespace cg = cooperative_groups;

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_F32(x)  TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

__device__ __forceinline__ float4 load4_scalar(const float* p) {
  return make_float4(p[0], p[1], p[2], p[3]);
}

__device__ __forceinline__ float2 load2_scalar(const float* p) {
  return make_float2(p[0], p[1]);
}


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


// template<int BK, int V, int STAGES>
// __device__ __forceinline__ void copy_tiles_async_float4(
//     cg::thread_block cta,
//     cuda::pipeline<cuda::thread_scope_block>& pipe,
//     float4 As4[STAGES][BK][V],
//     const float4* __restrict__ A4,
//     float4 Cs4[STAGES][BK][V],
//     const float4* __restrict__ C4,
//     int stage, int n0, int N)
// {
//   constexpr int tile_bytes  = BK * V * sizeof(float4);

//   float4* smem_ptr_A = &As4[stage][0][0];
//   float4* smem_ptr_C = &Cs4[stage][0][0];

//   if (n0 + BK <= N) {
//     const float4* gmem_ptr_A = &A4[n0 * V];
//     const float4* gmem_ptr_C = &C4[n0 * V];
//     cuda::memcpy_async(
//         cta,
//         smem_ptr_A,
//         gmem_ptr_A,
//         cuda::aligned_size_t<16>(tile_bytes),
//         pipe);
//     cuda::memcpy_async(
//         cta,
//         smem_ptr_C,
//         gmem_ptr_C,
//         cuda::aligned_size_t<16>(tile_bytes),
//         pipe);
    
//   }
//     else {
//       const int full_elems = BK * V;
//       for (int idx = threadIdx.x; idx < full_elems; idx += blockDim.x) {
//         const int k  = idx / V;
//         const int dv = idx - k * V;
//         const int n  = n0 + k;

//         float4* s_A = &As4[stage][k][dv];
//         float4* s_C = &Cs4[stage][k][dv];
//         if (n < N) {
//           *s_A = A4[n * V + dv];
//           *s_C = C4[n * V + dv];
//         } else {
//           *s_A = make_float4(0.f, 0.f, 0.f, 0.f);
//           *s_C = make_float4(0.f, 0.f, 0.f, 0.f);
//         }
//     }
//   }
// }


template<int BK, int V, int R>
__device__ __forceinline__ void copy_tiles_mixed_sync(
    float4 As4[BK][V],
    float2* __restrict__ As2_flat, 
    float*  __restrict__ As1_flat, 
    const float* __restrict__ A,
    float4 Cs4[BK][V],
    float2* __restrict__ Cs2_flat,
    float*  __restrict__ Cs1_flat,
    const float* __restrict__ C, 
    int tid, int BM, int n0, int N)
{
  constexpr bool HAS2 = (R >= 2);
  constexpr bool HAS1 = (R == 1 || R == 3);

  const int full_elems = BK * V;
  for (int idx = tid; idx < full_elems; idx += BM) {
    const int k  = idx / V;
    const int dv = idx - k * V;
    const int n  = n0 + k;

    float4* smem_ptr_A = &As4[k][dv];
    float4* smem_ptr_C = &Cs4[k][dv];

    if (n < N) {
      const float* row_A = A + n * (4*V + R); 
      const float* p_A   = row_A + 4*dv;
      const float* row_C = C + n * (4*V + R); 
      const float* p_C   = row_C + 4*dv;

      if (((uintptr_t)p_A & 0xF) == 0) {
        *smem_ptr_A = __ldg(reinterpret_cast<const float4*>(p_A));
      } else {
        *smem_ptr_A = load4_scalar(p_A);
      }
      if (((uintptr_t)p_C & 0xF) == 0) {
        *smem_ptr_C = __ldg(reinterpret_cast<const float4*>(p_C));
      } else {
        *smem_ptr_C = load4_scalar(p_C);   
      }
    } else {
      *smem_ptr_A = make_float4(0.f,0.f,0.f,0.f);
      *smem_ptr_C = make_float4(0.f,0.f,0.f,0.f);
    }
  }

  for (int k = tid; k < BK; k += BM) {
    const int n = n0 + k;
    if (n >= N) {
      if constexpr (HAS2) {
        As2_flat[k] = make_float2(0.f, 0.f);
        Cs2_flat[k] = make_float2(0.f, 0.f);
      }
      if constexpr (HAS1) {
        As1_flat[k] = 0.f;
        Cs1_flat[k] = 0.f;
      }
        continue;
    }

    const float* row_A = A + n * (4*V + R);
    const float* tail_A = row_A + 4*V;
    const float* row_C = C + n * (4*V + R);
    const float* tail_C = row_C + 4*V;

    if constexpr (HAS2) {
      if (((uintptr_t)tail_A & 0x7) == 0) {
        As2_flat[k] = __ldg(reinterpret_cast<const float2*>(tail_A));
      } else {
        As2_flat[k] = load2_scalar(tail_A);
      }
      if (((uintptr_t)tail_C & 0x7) == 0) {
        Cs2_flat[k] = __ldg(reinterpret_cast<const float2*>(tail_C));
      } else {
        Cs2_flat[k] = load2_scalar(tail_C);
      }
    }

    if constexpr (HAS1) {
      const int off = (HAS2 ? 2 : 0);
      As1_flat[k] = tail_A[off];
      Cs1_flat[k] = tail_C[off];
    }
  }
}


// template<int BK, int V, int STAGES, int MS>
// __global__ void relu_bat_c_fused_kernel_float4(
//     const float* __restrict__ A, 
//     const float* __restrict__ B, 
//     const float* __restrict__ C,
//     float* __restrict__ Y,       
//     int N, int M, int D)
// {
  
//   static_assert(STAGES >= 2, "STAGES must be >= 2");
//   cg::thread_block cta = cg::this_thread_block();

//   __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> shared_state;

//   auto pipe = cuda::make_pipeline(cta, &shared_state);

//   const int BM  = blockDim.x;
//   const int tid = threadIdx.x;
//   const int m   = blockIdx.x * MS * BM + tid;
  

//   __shared__ alignas(16) float4 As4[STAGES][BK][V];
//   __shared__ alignas(16) float4 Cs4[STAGES][BK][V];

//   const float4* __restrict__ A4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(A, 16));
//   const float4* __restrict__ B4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(B, 16));
//   const float4* __restrict__ C4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(C, 16));
//   float4* __restrict__ Y4       = reinterpret_cast<float4*>(__builtin_assume_aligned(Y, 16));

//   float4 b[MS][V] = {0.f};
//   float4 y[MS][V] = {0.f};
//   float4 a[V] = {0.f};
//   float score[MS] = {0.f};

//   #pragma unroll
//   for (int J = 0; J < MS; ++J) {
//     if (m + J*BM < M) {
//       #pragma unroll
//       for (int I = 0; I < V; ++I) {
//         b[J][I] = B4[(m + J*BM) * (D/4) + I];
//       }
//     }
//   }
  

//   const int warm = (N + BK - 1) / BK;              
//   const int prefetch_tiles = (warm < STAGES) ? warm : STAGES;

//   for (int t = 0; t < STAGES; ++t) {
//     const int n0 = t * BK;
//     if (t < prefetch_tiles) {
//       pipe.producer_acquire();
//       copy_tiles_async_float4<BK, V, STAGES>(cta, pipe, As4, A4, Cs4, C4, t, n0, N);
//       pipe.producer_commit();
//     }
//   }
  
//   for (int tile = 0, n0 = 0; n0 < N; ++tile, n0 += BK) {
    
//     const int cur_stage = tile % STAGES;
//     const int n_prefetch = n0 + STAGES * BK;

//     pipe.consumer_wait();
//     __syncthreads();


//     #pragma unroll
//     for (int kk = 0; kk < BK; ++kk) {

//       #pragma unroll
//       for (int I = 0; I < V; ++I) {
//         a[I] = As4[cur_stage][kk][I];
//       }

//       #pragma unroll
//       for (int J = 0; J < MS; ++J) {
//           score[J] = 0.f;
//       }

//       #pragma unroll
//       for (int I = 0; I < V; ++I) {
//         #pragma unroll
//         for (int J = 0; J < MS; ++J) {
//           const float4 bj = b[J][I];
//           score[J] = dot_float4(bj, a[I], score[J]);
//         }
//       }

//       #pragma unroll
//       for (int I = 0; I < V; ++I) {
//         a[I] = Cs4[cur_stage][kk][I];
//       }

//       #pragma unroll
//       for (int J = 0; J < MS; ++J) {
//         const float alpha = fmaxf(score[J], 0.f);

//         #pragma unroll
//         for (int I = 0; I < V; ++I) {
//           y[J][I] = axpy_float4(alpha, a[I], y[J][I]);
//         }
//       }
//     }

//     __syncthreads();
//     pipe.consumer_release();

//     if (n_prefetch < N) {
//       pipe.producer_acquire();
//       copy_tiles_async_float4<BK, V, STAGES>(cta, pipe, As4, A4, Cs4, C4, cur_stage, n_prefetch, N);
//       pipe.producer_commit();
//     }
//   }

//   #pragma unroll
//   for (int J = 0; J < MS; ++J) {
//     if (m + J*BM < M) {
//       #pragma unroll
//       for (int I = 0; I < V; ++I) {
//         Y4[(m + J*BM) * (D/4) + I] = y[J][I];
//       }
//     }
//   }
  
// }

template<int BK, int V, int MS>
__global__ void relu_bat_c_fused_kernel_float4_sync(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    float* __restrict__ Y,
    int N, int M, int D)
{
  

  const int BM  = (int)blockDim.x;
  const int tid = (int)threadIdx.x;
  const int m0  = (int)(blockIdx.x * MS * BM + tid);

  __shared__ alignas(16) float4 As4[BK][V];
  __shared__ alignas(16) float4 Cs4[BK][V];

  const float4* __restrict__ A4 = reinterpret_cast<const float4*>(A);
  const float4* __restrict__ B4 = reinterpret_cast<const float4*>(B);
  const float4* __restrict__ C4 = reinterpret_cast<const float4*>(C);
  float4* __restrict__ Y4       = reinterpret_cast<float4*>(Y);

  float4 b[MS][V] = {0.f};
  float4 y[MS][V] = {0.f};

  float4 a[V] = {0.f};

  float score[MS] = {0.f};
  #pragma unroll
  for (int j = 0; j < MS; ++j) {
    #pragma unroll
    for (int i = 0; i < V; ++i) {

      if (m0 + j * BM < M) {
       b[j][i] = B4[(m0 + j * BM) * V + i];
      }
    }
  }

  for (int n0 = 0; n0 < N; n0 += BK) {

    for (int idx = tid; idx < BK*V; idx += BM) {
      const int k  = idx / V;
      const int dv = idx - k * V;
      const int n  = n0 + k;

      const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
      As4[k][dv] = (n < N) ? __ldg(&A4[n * V + dv]) : z;
      Cs4[k][dv] = (n < N) ? __ldg(&C4[n * V + dv]) : z;
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      #pragma unroll
      for (int i = 0; i < V; ++i) {
        a[i] = As4[kk][i];
      }
      

      #pragma unroll
      for (int j = 0; j < MS; ++j) {
        score[j] = 0.f;
      }

      #pragma unroll
      for (int j = 0; j < MS; ++j) {
        #pragma unroll  
        for (int i = 0; i < V; ++i) {
          const float4 bj = b[j][i];
          const float4 ar = a[i];
          score[j] = dot_float4(bj,ar,score[j]);
          }
        }
      
      #pragma unroll
      for (int i = 0; i < V; ++i) {
          a[i] = Cs4[kk][i];
      }
      
      #pragma unroll
      for (int j = 0; j < MS; ++j) {
        const float alpha = fmaxf(score[j], 0.f);

        #pragma unroll
        for (int i = 0; i < V; ++i) {
          const float4 ar = a[i];
          y[j][i] = axpy_float4(alpha, ar, y[j][i]);
        }
      }
    }



    

    __syncthreads();
  }

  #pragma unroll
  for (int j = 0; j < MS; ++j) {
    const int row = m0 + j * BM;
    if (row < M) {
      #pragma unroll
      for (int i = 0; i < V; ++i) {
        Y4[row * V + i] = y[j][i];
      }
    }
  }
}


template<int BK, int V, int R, int MS>
__global__ void relu_bat_c_fused_kernel_mixed_sync(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    const float* __restrict__ C,
    float* __restrict__ Y,       
    int N, int M, int D)
{
  
  static_assert(R >= 0 && R <=3);

  constexpr bool HAS2 = (R >= 2);
  constexpr bool HAS1 = (R == 1 || R == 3);

  const int BM  = blockDim.x;
  const int tid = threadIdx.x;
  const int m   = blockIdx.x * MS * BM + tid;
  

  __shared__ alignas(16) float4 As4[BK][V];
  __shared__ alignas(16) float4 Cs4[BK][V];
  __shared__ float2 As2[BK];
  __shared__ float As[BK];
  __shared__ float2 Cs2[BK];
  __shared__ float Cs[BK];
  

  const float4* __restrict__ A4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(A, 16));
  const float4* __restrict__ B4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(B, 16));
  const float4* __restrict__ C4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(C, 16));
  float4* __restrict__ Y4       = reinterpret_cast<float4*>(__builtin_assume_aligned(Y, 16));

  float4 b[MS][V] = {0.f};
  float2 b2[MS]       = {0.f};
  float b1[MS]        = {0.f};
  float4 y[MS][V] = {0.f};
  float2 y2[MS]   = {0.f};
  float  y1[MS]   = {0.f};
  float4 a[V]     = {0.f};
  float score[MS] = {0.f};
  float2 a2       = {0.f};
  float a1        = {0.f};

  #pragma unroll
  for (int J = 0; J < MS; ++J) {
    const float* Brow = B + (m + J*BM) * D;
    if (m + J*BM < M) {
      #pragma unroll
      for (int I = 0; I < V; ++I) {
        b[J][I] = load4_scalar(Brow + 4*I);
      }
      if constexpr (HAS2) b2[J] = load2_scalar(Brow + 4 * V);
      if constexpr (HAS1) b1[J] = Brow[4*V + (HAS2 ? 2 : 0)];
    }
  }
  
  
  for (int n0 = 0; n0 < N; n0 += BK) {
    
    copy_tiles_mixed_sync<BK,V,R>(As4, As2, As, A, Cs4, Cs2, Cs, C, tid, BM, n0, N);
    __syncthreads();


    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {

      #pragma unroll
      for (int I = 0; I < V; ++I) {
        a[I] = As4[kk][I];
      }
      if constexpr (HAS2) a2 = As2[kk];
      if constexpr (HAS1) a1 = As[kk];

      #pragma unroll
      for (int J = 0; J < MS; ++J) {
          score[J] = 0.f;
      }

      #pragma unroll
      for (int J = 0; J < MS; ++J) {
        #pragma unroll
        for (int I = 0; I < V; ++I) {
          const float4 bj = b[J][I];
          score[J] = dot_float4(bj, a[I], score[J]);
        }
        if constexpr (HAS2) score[J] = dot_float2(b2[J], a2, score[J]);
        if constexpr (HAS1) score[J] = fmaf(b1[J], a1, score[J]);
      }

      #pragma unroll
      for (int I = 0; I < V; ++I) {
        a[I] = Cs4[kk][I];
      }
      if constexpr (HAS2) a2 = Cs2[kk];
      if constexpr (HAS1) a1 = Cs[kk];

      #pragma unroll
      for (int J = 0; J < MS; ++J) {
        const float alpha = fmaxf(score[J], 0.f);

        #pragma unroll
        for (int I = 0; I < V; ++I) {
          y[J][I] = axpy_float4(alpha, a[I], y[J][I]);
        }
        if constexpr (HAS2) y2[J] = axpy_float2(alpha, a2, y2[J]);
        if constexpr (HAS1) y1[J] = fmaf(alpha, a1, y1[J]);
      }
    }

    __syncthreads();
    
  }
  #pragma unroll
  for (int J = 0; J < MS; ++J) {
    if (m + J*BM < M) {
      float* Yrow = Y + (m + J*BM) * D;

      #pragma unroll
      for (int I = 0; I < V; ++I) {
        const float4 v = y[J][I];
        Yrow[4*I + 0] = v.x; Yrow[4*I + 1] = v.y; Yrow[4*I + 2] = v.z; Yrow[4*I + 3] = v.w;
      }
      if constexpr (HAS2) { Yrow[4*V + 0] = y2[J].x; Yrow[4*V + 1] = y2[J].y; }
      if constexpr (HAS1) { Yrow[4*V + (HAS2 ? 2 : 0)] = y1[J]; }
    }
  }
  
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
  TORCH_CHECK(ok, "Unsupported ", name, "=", value, ". Add ", name, "=", value, " to the dispatch set in fused_bat_c.cu and recompile the extension.");
}

template<int BK, int V, int MS>
inline void launch_relu_bat_c_fused(
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    at::Tensor& Y,
    int N, int M, int D)
{
  relu_bat_c_fused_kernel_float4_sync<BK, V, MS><<<grid, block, 0, stream>>>(
      (const float*)A.data_ptr<float>(),
      (const float*)B.data_ptr<float>(),
      (const float*)C.data_ptr<float>(),
      (float*)Y.data_ptr<float>(),
      N, M, D);
}

template<int BK, int V, int R, int MS>
inline void launch_relu_bat_c_fused_mixed(
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    at::Tensor& Y,
    int N, int M, int D)
{
  relu_bat_c_fused_kernel_mixed_sync<BK, V, R, MS><<<grid, block, 0, stream>>>(
      (const float*)A.data_ptr<float>(),
      (const float*)B.data_ptr<float>(),
      (const float*)C.data_ptr<float>(),
      (float*)Y.data_ptr<float>(),
      N, M, D);
}

inline void dispatch(
    int V_runtime,
    int BK,
    int MS_runtime,
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    at::Tensor& Y,
    int N, int M, int D)
{
  dispatch_from_values<1,2,3,4>(V_runtime, [&](auto Vc) {
    constexpr int V = decltype(Vc)::value;

    dispatch_from_values<16,32,64>(BK, [&](auto BKc) {
      constexpr int BK_ = decltype(BKc)::value;

      dispatch_from_values<2,4,6>(MS_runtime, [&](auto MSc) {
        constexpr int MS = decltype(MSc)::value;

            launch_relu_bat_c_fused<BK_, V, MS>(
                grid, block, stream, A, B, C, Y, N, M, D);

      }, "MS");

    }, "BK");

  }, "V");
}


inline void dispatch_R(
    int V_runtime,
    int R_runtime,
    int BK,
    int MS_runtime,
    dim3 grid, dim3 block, cudaStream_t stream,
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& C,
    at::Tensor& Y,
    int N, int M, int D)
{
  dispatch_from_values<1,2,3,4>(V_runtime, [&](auto Vc) {
    constexpr int V = decltype(Vc)::value;

    dispatch_from_values<1,2,3>(R_runtime, [&](auto Rc) {
      constexpr int R = decltype(Rc)::value;

          dispatch_from_values<16,32,64>(BK, [&](auto BKc) {
            constexpr int BK_ = decltype(BKc)::value;

            dispatch_from_values<2,4,6>(MS_runtime, [&](auto MSc) {
              constexpr int MS = decltype(MSc)::value;

                  launch_relu_bat_c_fused_mixed<BK_, V, R, MS>(
                      grid, block, stream, A, B, C, Y, N, M, D);

            }, "MS");

          }, "BK");

      }, "R");

  }, "V");
}


torch::Tensor relu_bat_c_fused_cuda(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    int64_t BM,
    int64_t BK,
    int64_t num_ms)
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
  const int R = D % 4;

  auto Y = torch::empty({M, D}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

  TORCH_CHECK(BM % 32 == 0, "BM must be multiple of 32");
  TORCH_CHECK(num_ms >= 1, "num_ms must be >= 1");

  const dim3 block(BM);
  const dim3 grid(ceil_div(M, num_ms * BM));

  if (R == 0) {
    dispatch(V, BK, num_ms, grid, block, stream, A, B, C, Y, N, M, D);
  } else {
    dispatch_R(V, R, BK, num_ms, grid, block, stream, A, B, C, Y, N, M, D);
  }

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
  return Y;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("relu_bat_c_fused_cuda", &relu_bat_c_fused_cuda, "relu_bat_c_fused_cuda (CUDA)");
}