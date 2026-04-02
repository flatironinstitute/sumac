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

__device__ __forceinline__ float dot_float2(const float2& A, const float2& B, const float c) {
  float r = fmaf(A.x, B.x, c);
  r = fmaf(A.y, B.y, r);
  return r;
}

__device__ __forceinline__ void copy_tile_mixed_sync(
    float4 Bs4[BK][V],
    float2* __restrict__ Bs2_flat,
    float*  __restrict__ Bs1_flat,
    const float* __restrict__ B,
    int tid, int BM, int n0, int N)
{
  constexpr bool HAS2 = (R >= 2);
  constexpr bool HAS1 = (R == 1 || R == 3);

  const int full_elems = BK * V;
  for (int idx = tid; idx < full_elems; idx += BM) {
    const int k  = idx / V;
    const int dv = idx - k * V;
    const int n  = n0 + k;

    float4* smem_ptr = &Bs4[k][dv];

    if (n < N) {
      const float* row = B + n * (4 * V + R);
      const float* p   = row + 4 * dv;

      if (((uintptr_t)p & 0xF) == 0) {
        *smem_ptr = __ldg(reinterpret_cast<const float4*>(p));
      } else {
        *smem_ptr = load4_scalar(p);
      }
    } else {
      *smem_ptr = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }

  for (int k = tid; k < BK; k += BM) {
    const int n = n0 + k;

    if (n >= N) {
      if constexpr (HAS2) Bs2_flat[k] = make_float2(0.f, 0.f);
      if constexpr (HAS1) Bs1_flat[k] = 0.f;
      continue;
    }

    const float* row  = B + n * (4 * V + R);
    const float* tail = row + 4 * V;

    if constexpr (HAS2) {
      if (((uintptr_t)tail & 0x7) == 0) {
        Bs2_flat[k] = __ldg(reinterpret_cast<const float2*>(tail));
      } else {
        Bs2_flat[k] = load2_scalar(tail);
      }
    }

    if constexpr (HAS1) {
      const int off = (HAS2 ? 2 : 0);
      Bs1_flat[k] = tail[off];
    }
  }
}

// KERNEL_START

extern "C" __global__ void relu_abt_reduce_kernel_mixed_sync(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ out_sum,
    float* __restrict__ out_sum2,
    int M, int N, int D)
{
  static_assert(R >= 0 && R <= 3);

  constexpr bool HAS2 = (R >= 2);
  constexpr bool HAS1 = (R == 1 || R == 3);

  const int BM  = (int)blockDim.x;
  const int tid = (int)threadIdx.x;

  const int a0 = (int)(blockIdx.x * MS * BM + tid);
  const int b0 = (int)(blockIdx.y * BK);

  __shared__ alignas(16) float4 Bs4[BK][V];
  __shared__ float2 Bs2[BK];
  __shared__ float  Bs1[BK];

  float4 a4[MS][V];
  float2 a2[MS];
  float  a1[MS];
  float  score[MS];

  float local_sum  = 0.f;
  float local_sum2 = 0.f;

  #pragma unroll
  for (int j = 0; j < MS; ++j) {
    const int row = a0 + j * BM;

    #pragma unroll
    for (int i = 0; i < V; ++i) {
      if (row < M) {
        const float* p = A + row * (4 * V + R) + 4 * i;
        a4[j][i] = load4_scalar(p);
      } else {
        a4[j][i] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }

    if constexpr (HAS2) {
      if (row < M) {
        const float* tail = A + row * (4 * V + R) + 4 * V;
        a2[j] = load2_scalar(tail);
      } else {
        a2[j] = make_float2(0.f, 0.f);
      }
    }

    if constexpr (HAS1) {
      if (row < M) {
        const float* tail = A + row * (4 * V + R) + 4 * V;
        a1[j] = tail[HAS2 ? 2 : 0];
      } else {
        a1[j] = 0.f;
      }
    }
  }

  copy_tile_mixed_sync(Bs4, Bs2, Bs1, B, tid, BM, b0, N);
  __syncthreads();

  #pragma unroll
  for (int kk = 0; kk < BK; ++kk) {
    const int brow = b0 + kk;
    if (brow >= N) break;

    #pragma unroll
    for (int j = 0; j < MS; ++j) {
      score[j] = 0.f;
    }

    #pragma unroll
    for (int i = 0; i < V; ++i) {
      const float4 b = Bs4[kk][i];

      #pragma unroll
      for (int j = 0; j < MS; ++j) {
        score[j] = dot_float4(a4[j][i], b, score[j]);
      }
    }

    if constexpr (HAS2) {
      const float2 b2 = Bs2[kk];
      #pragma unroll
      for (int j = 0; j < MS; ++j) {
        score[j] = dot_float2(a2[j], b2, score[j]);
      }
    }

    if constexpr (HAS1) {
      const float b1 = Bs1[kk];
      #pragma unroll
      for (int j = 0; j < MS; ++j) {
        score[j] = fmaf(a1[j], b1, score[j]);
      }
    }

    #pragma unroll
    for (int j = 0; j < MS; ++j) {
      const int arow = a0 + j * BM;
      if (arow < M) {
        const float x = fmaxf(score[j], 0.f);
        local_sum  += x;
        local_sum2 += x * x;
      }
    }
  }

  // block reduction
  extern __shared__ float redbuf[];
  float* s_sum  = redbuf;
  float* s_sum2 = redbuf + BM;

  s_sum[tid]  = local_sum;
  s_sum2[tid] = local_sum2;
  __syncthreads();

  for (int stride = BM >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_sum[tid]  += s_sum[tid + stride];
      s_sum2[tid] += s_sum2[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(out_sum,  s_sum[0]);
    atomicAdd(out_sum2, s_sum2[0]);
  }
}