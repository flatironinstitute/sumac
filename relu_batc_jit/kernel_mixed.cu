#include <cuda/std/cstdint>

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

// KERNEL_START

extern "C" __global__ void relu_bat_c_fused_kernel_mixed_sync(
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
  float2 b2[MS]   = {0.f};
  float b1[MS]    = {0.f};
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
    
    copy_tiles_mixed_sync(As4, As2, As, A, Cs4, Cs2, Cs, C, tid, BM, n0, N);
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