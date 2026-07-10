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

extern "C" __global__ void relu_bat_c_fused_kernel_float4_sync(
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

    #pragma unroll 16
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
