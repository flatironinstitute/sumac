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

// KERNEL_START

extern "C" __global__ void relu_bat_c_sparse_fused_kernel_float4_sync(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    const long long* __restrict__ row_ptr,
    const long long* __restrict__ S_nnz_i,
    const float* __restrict__ S_nnz_val,
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
  int S_nnz_pos[MS] = {0};
  int S_nnz_end[MS] = {0};
  int next_nnz_i[MS] = {0};
  float next_nnz_val[MS] = {0.f};

  #pragma unroll
  for (int J = 0; J < MS; ++J) {
    const int row = m0 + J * BM;
    if (row < M) {
      S_nnz_pos[J] = (int)row_ptr[row];
      S_nnz_end[J] = (int)row_ptr[row + 1];
      if (S_nnz_pos[J] < S_nnz_end[J]) {
        next_nnz_i[J] = (int)S_nnz_i[S_nnz_pos[J]];
        next_nnz_val[J] = S_nnz_val[S_nnz_pos[J]];
      } else {
        next_nnz_i[J] = N;
      }
    } else {
      next_nnz_i[J] = N;
    }

    #pragma unroll
    for (int I = 0; I < V; ++I) {
      if (row < M) {
        b[J][I] = B4[row * V + I];
      }
    }
  }

  for (int n0 = 0; n0 < N; n0 += BK) {
    for (int idx = tid; idx < BK * V; idx += BM) {
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
      const int n = n0 + kk;

      #pragma unroll
      for (int I = 0; I < V; ++I) {
        a[I] = As4[kk][I];
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
      }

      #pragma unroll
      for (int I = 0; I < V; ++I) {
        a[I] = Cs4[kk][I];
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
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (int J = 0; J < MS; ++J) {
    const int row = m0 + J * BM;
    if (row < M) {
      #pragma unroll
      for (int I = 0; I < V; ++I) {
        Y4[row * V + I] = y[J][I];
      }
    }
  }
}
