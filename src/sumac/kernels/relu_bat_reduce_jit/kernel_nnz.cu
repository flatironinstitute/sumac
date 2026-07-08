__device__ __forceinline__ float dot_float4(const float4& A, const float4& B, const float c) {
  float r = fmaf(A.x, B.x, c);
  r = fmaf(A.y, B.y, r);
  r = fmaf(A.z, B.z, r);
  r = fmaf(A.w, B.w, r);
  return r;
}

// KERNEL_START

extern "C" __global__ void relu_bat_reduce_kernel_float4_sync(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    double* __restrict__ out_sum, 
    double* __restrict__ out_sum2,
    unsigned long long* __restrict__ out_nnz,
    int M, int N, int D)
{
  const int BM  = (int)blockDim.x;
  const int tid = (int)threadIdx.x;

  const int a0 = (int)(blockIdx.x * MS * BM + tid);
  const int b0 = (int)(blockIdx.y * BK);

  const float4* __restrict__ A4 = reinterpret_cast<const float4*>(A);
  const float4* __restrict__ B4 = reinterpret_cast<const float4*>(B);

  __shared__ alignas(16) float4 Bs4[BK][V];

  float4 a[MS][V] = {0.f};
  float score[MS] = {0.f};

  float local_sum  = {0.f};
  float local_sum2 = {0.f};
  unsigned local_nnz = {0};

  #pragma unroll
  for (int j = 0; j < MS; ++j) {
    const int row = a0 + j * BM;
    #pragma unroll
    for (int i = 0; i < V; ++i) {
      if (row < M) {
        a[j][i] = A4[row * V + i];
      } else {
        a[j][i] = make_float4(0.f, 0.f, 0.f, 0.f);
      }
    }
  }

  for (int idx = tid; idx < BK * V; idx += BM) {
    const int k  = idx / V;
    const int dv = idx - k * V;
    const int row = b0 + k;

    if (row < N) {
      Bs4[k][dv] = __ldg(&B4[row * V + dv]);
    } else {
      Bs4[k][dv] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
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
        score[j] = dot_float4(a[j][i], b, score[j]);
      }
    }

    #pragma unroll
    for (int j = 0; j < MS; ++j) {
      const int arow = a0 + j * BM;
      if (arow < M) {
        const float x  = fmaxf(score[j], 0.f);
        local_sum  += x;
        local_sum2 += x * x;
        local_nnz  += static_cast<unsigned>(x > 0.f);
      }
    }
  }

  extern __shared__ __align__(16) unsigned char redbuf[];

  float*    s_sum  = reinterpret_cast<float*>(redbuf);
  float*    s_sum2 = s_sum + BM;
  unsigned* s_nnz  = reinterpret_cast<unsigned*>(s_sum2 + BM);

  s_sum[tid]  = local_sum;
  s_sum2[tid] = local_sum2;
  s_nnz[tid]  = local_nnz;
  __syncthreads();

  for (int stride = BM >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_sum[tid]  += s_sum[tid + stride];
      s_sum2[tid] += s_sum2[tid + stride];
      s_nnz[tid]  += s_nnz[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(out_sum,  (double)s_sum[0]);
    atomicAdd(out_sum2, (double)s_sum2[0]);
    atomicAdd(out_nnz,  (unsigned long long)s_nnz[0]);
  }
}