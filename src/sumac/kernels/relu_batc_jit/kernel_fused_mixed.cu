static constexpr int VEC = (V > 0) ? V : 1;

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
    float4 As4[BK][VEC],
    float2* __restrict__ As2,
    float* __restrict__ As1,
    const float* __restrict__ A,
    float4 Cs4[BK][VEC],
    float2* __restrict__ Cs2,
    float* __restrict__ Cs1,
    const float* __restrict__ C,
    int tid, int BM, int n0, int N, int D)
{
  constexpr bool HAS2 = (R >= 2);
  constexpr bool HAS1 = (R == 1 || R == 3);

  if constexpr (V > 0) {
    for (int idx = tid; idx < BK * V; idx += BM) {
      const int k = idx / V;
      const int dv = idx - k * V;
      const int n = n0 + k;

      if (n < N) {
        const float* p_A = A + n * D + 4 * dv;
        const float* p_C = C + n * D + 4 * dv;

        if (((unsigned long long)p_A & 0xF) == 0) {
          As4[k][dv] = __ldg(reinterpret_cast<const float4*>(p_A));
        } else {
          As4[k][dv] = load4_scalar(p_A);
        }
        if (((unsigned long long)p_C & 0xF) == 0) {
          Cs4[k][dv] = __ldg(reinterpret_cast<const float4*>(p_C));
        } else {
          Cs4[k][dv] = load4_scalar(p_C);
        }
      } else {
        As4[k][dv] = make_float4(0.f, 0.f, 0.f, 0.f);
        Cs4[k][dv] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
  }

  for (int k = tid; k < BK; k += BM) {
    const int n = n0 + k;
    if (n >= N) {
      if constexpr (HAS2) {
        As2[k] = make_float2(0.f, 0.f);
        Cs2[k] = make_float2(0.f, 0.f);
      }
      if constexpr (HAS1) {
        As1[k] = 0.f;
        Cs1[k] = 0.f;
      }
      continue;
    }

    const float* tail_A = A + n * D + 4 * V;
    const float* tail_C = C + n * D + 4 * V;

    if constexpr (HAS2) {
      if (((unsigned long long)tail_A & 0x7) == 0) {
        As2[k] = __ldg(reinterpret_cast<const float2*>(tail_A));
      } else {
        As2[k] = load2_scalar(tail_A);
      }
      if (((unsigned long long)tail_C & 0x7) == 0) {
        Cs2[k] = __ldg(reinterpret_cast<const float2*>(tail_C));
      } else {
        Cs2[k] = load2_scalar(tail_C);
      }
    }
    if constexpr (HAS1) {
      const int off = (HAS2 ? 2 : 0);
      As1[k] = tail_A[off];
      Cs1[k] = tail_C[off];
    }
  }
}

// KERNEL_START

extern "C" __global__ void relu_bat_c_sparse_fused_kernel_mixed_sync(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const long long* __restrict__ row_ptr,
    const long long* __restrict__ S_nnz_i,
    const float* __restrict__ S_nnz_val,
    float* __restrict__ Y,
    int N, int M, int D)
{
  static_assert(R >= 1 && R <= 3);

  constexpr bool HAS2 = (R >= 2);
  constexpr bool HAS1 = (R == 1 || R == 3);

  const int BM = (int)blockDim.x;
  const int tid = (int)threadIdx.x;
  const int m0 = (int)(blockIdx.x * MS * BM + tid);

  __shared__ alignas(16) float4 As4[BK][VEC];
  __shared__ alignas(16) float4 Cs4[BK][VEC];
  __shared__ float2 As2[BK];
  __shared__ float As1[BK];
  __shared__ float2 Cs2[BK];
  __shared__ float Cs1[BK];

  float4 b[MS][VEC] = {0.f};
  float2 b2[MS] = {0.f};
  float b1[MS] = {0.f};
  float4 y[MS][VEC] = {0.f};
  float2 y2[MS] = {0.f};
  float y1[MS] = {0.f};
  float4 a[VEC] = {0.f};
  float2 a2 = {0.f};
  float a1 = 0.f;
  float score[MS] = {0.f};
  int S_nnz_pos[MS] = {0};
  int S_nnz_end[MS] = {0};
  int next_nnz_i[MS] = {0};
  float next_nnz_val[MS] = {0.f};

  #pragma unroll
  for (int J = 0; J < MS; ++J) {
    const int row = m0 + J * BM;
    const float* Brow = B + row * D;

    if (row < M) {
      S_nnz_pos[J] = (int)row_ptr[row];
      S_nnz_end[J] = (int)row_ptr[row + 1];
      if (S_nnz_pos[J] < S_nnz_end[J]) {
        next_nnz_i[J] = (int)S_nnz_i[S_nnz_pos[J]];
        next_nnz_val[J] = S_nnz_val[S_nnz_pos[J]];
      } else {
        next_nnz_i[J] = N;
      }

      #pragma unroll
      for (int I = 0; I < V; ++I) {
        b[J][I] = load4_scalar(Brow + 4 * I);
      }
      if constexpr (HAS2) {
        b2[J] = load2_scalar(Brow + 4 * V);
      }
      if constexpr (HAS1) {
        b1[J] = Brow[4 * V + (HAS2 ? 2 : 0)];
      }
    } else {
      next_nnz_i[J] = N;
    }
  }

  for (int n0 = 0; n0 < N; n0 += BK) {
    copy_tiles_mixed_sync(As4, As2, As1, A, Cs4, Cs2, Cs1, C, tid, BM, n0, N, D);
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      const int n = n0 + kk;

      #pragma unroll
      for (int I = 0; I < V; ++I) {
        a[I] = As4[kk][I];
      }
      if constexpr (HAS2) {
        a2 = As2[kk];
      }
      if constexpr (HAS1) {
        a1 = As1[kk];
      }

      #pragma unroll
      for (int J = 0; J < MS; ++J) {
        score[J] = 0.f;
      }

      #pragma unroll
      for (int J = 0; J < MS; ++J) {
        #pragma unroll
        for (int I = 0; I < V; ++I) {
          score[J] = dot_float4(b[J][I], a[I], score[J]);
        }
        if constexpr (HAS2) {
          score[J] = dot_float2(b2[J], a2, score[J]);
        }
        if constexpr (HAS1) {
          score[J] = fmaf(b1[J], a1, score[J]);
        }
      }

      #pragma unroll
      for (int I = 0; I < V; ++I) {
        a[I] = Cs4[kk][I];
      }
      if constexpr (HAS2) {
        a2 = Cs2[kk];
      }
      if constexpr (HAS1) {
        a1 = Cs1[kk];
      }

      #pragma unroll
      for (int J = 0; J < MS; ++J) {
        const int row = m0 + J * BM;
        float alpha = fmaxf(score[J], 0.f);

        if (row < M && n < N && n == next_nnz_i[J]) {
          alpha = score[J] - next_nnz_val[J];
          ++S_nnz_pos[J];

          if (S_nnz_pos[J] < S_nnz_end[J]) {
            next_nnz_i[J] = (int)S_nnz_i[S_nnz_pos[J]];
            next_nnz_val[J] = S_nnz_val[S_nnz_pos[J]];
          } else {
            next_nnz_i[J] = N;
          }
        }

        #pragma unroll
        for (int I = 0; I < V; ++I) {
          y[J][I] = axpy_float4(alpha, a[I], y[J][I]);
        }
        if constexpr (HAS2) {
          y2[J] = axpy_float2(alpha, a2, y2[J]);
        }
        if constexpr (HAS1) {
          y1[J] = fmaf(alpha, a1, y1[J]);
        }
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (int J = 0; J < MS; ++J) {
    const int row = m0 + J * BM;
    if (row < M) {
      float* Yrow = Y + row * D;

      #pragma unroll
      for (int I = 0; I < V; ++I) {
        const float4 v = y[J][I];
        Yrow[4 * I + 0] = v.x;
        Yrow[4 * I + 1] = v.y;
        Yrow[4 * I + 2] = v.z;
        Yrow[4 * I + 3] = v.w;
      }
      if constexpr (HAS2) {
        Yrow[4 * V + 0] = y2[J].x;
        Yrow[4 * V + 1] = y2[J].y;
      }
      if constexpr (HAS1) {
        Yrow[4 * V + (HAS2 ? 2 : 0)] = y1[J];
      }
    }
  }
}
