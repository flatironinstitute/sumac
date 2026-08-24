#ifndef REDUCE_MFMA_FP32_KERNEL_NAME
#define REDUCE_MFMA_FP32_KERNEL_NAME relu_bat_reduce_fp32_mfma
#endif

static constexpr int MFMA_M = 16;
static constexpr int MFMA_N = 16;
static constexpr int MFMA_K = 4;
static constexpr int WAVE_SIZE = 64;

static constexpr int WAVE_M_ROWS = M_TILES * MFMA_M;
static constexpr int WAVES_PER_BLOCK = BM / WAVE_M_ROWS;
static constexpr int THREADS_PER_BLOCK = WAVES_PER_BLOCK * WAVE_SIZE;

static constexpr int D_K_TILES = D_f / MFMA_K;
static constexpr int PANEL_N_TILES = BN / MFMA_N;
static constexpr int PANEL_WORDS = BN * D_f;
static constexpr int REDUCTION_WORDS = 2 * THREADS_PER_BLOCK;
static constexpr int SMEM_WORDS = PANEL_WORDS + REDUCTION_WORDS;

static_assert(BM > 0 && BN > 0 && M_TILES > 0 && D_f > 0,
              "MFMA tile dimensions must be positive");
static_assert((BN % MFMA_N) == 0, "BN must be divisible by 16");
static_assert((D_f % MFMA_K) == 0, "D_f must be divisible by 4");
static_assert((BM % WAVE_M_ROWS) == 0,
              "BM must be divisible by M_TILES * 16");
static_assert(THREADS_PER_BLOCK > 0 && THREADS_PER_BLOCK <= 1024,
              "MFMA workgroup size must be in [64, 1024]");

using mfma_f32x4 = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ mfma_f32x4 mfma_zero_f32x4()
{
    return mfma_f32x4{0.0f, 0.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ mfma_f32x4 mfma_m16n16k4_f32(
    float a,
    float b,
    mfma_f32x4 accum)
{
    return __builtin_amdgcn_mfma_f32_16x16x4f32(
        a,
        b,
        accum,
        0, // cbsz
        0, // abid
        0  // blgp
    );
}

// KERNEL_START

extern "C" __global__
__launch_bounds__(THREADS_PER_BLOCK, 1)
void REDUCE_MFMA_FP32_KERNEL_NAME(
    const float* __restrict__ A,
    const float* __restrict__ B,
    double* __restrict__ out_sum,
    double* __restrict__ out_sum2,
    int M,
    int N,
    int D)
{
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (WAVE_SIZE - 1);
    const int wave = tid / WAVE_SIZE;
    const int lane_x = lane & (MFMA_N - 1);
    const int lane_q = lane / MFMA_N;

    const int block_m0 = static_cast<int>(blockIdx.x) * BM;
    const int block_n0 = static_cast<int>(blockIdx.y) * BN;
    const int wave_m0 = block_m0 + wave * WAVE_M_ROWS;

    extern __shared__ float dynamic_smem[];
    float* __restrict__ Bs = dynamic_smem;
    float* __restrict__ red_sum = Bs + PANEL_WORDS;
    float* __restrict__ red_sum2 = red_sum + THREADS_PER_BLOCK;

    float a_regs[M_TILES][D_K_TILES];
    #pragma unroll
    for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
        const int m = wave_m0 + m_tile * MFMA_M + lane_x;

        #pragma unroll
        for (int d_tile = 0; d_tile < D_K_TILES; ++d_tile) {
            const int d = d_tile * MFMA_K + lane_q;
            a_regs[m_tile][d_tile] =
                (m < M && d < D)
                    ? A[static_cast<long long>(m) * D + d]
                    : 0.0f;
        }
    }

    for (int idx = tid; idx < PANEL_WORDS; idx += THREADS_PER_BLOCK) {
        const int panel_n = idx / D_f;
        const int d = idx - panel_n * D_f;
        const int n = block_n0 + panel_n;

        const float b =
            (n < N && d < D)
                ? B[static_cast<long long>(n) * D + d]
                : 0.0f;
        Bs[d * BN + panel_n] = b;
    }
    __syncthreads();

    float local_sum = 0.0f;
    float local_sum2 = 0.0f;

    #pragma unroll
    for (int n_tile = 0; n_tile < PANEL_N_TILES; ++n_tile) {
        mfma_f32x4 score[M_TILES];
        #pragma unroll
        for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
            score[m_tile] = mfma_zero_f32x4();
        }

        #pragma unroll
        for (int d_tile = 0; d_tile < D_K_TILES; ++d_tile) {
            const int panel_n = n_tile * MFMA_N + lane_x;
            const int d = d_tile * MFMA_K + lane_q;
            const float b = Bs[d * BN + panel_n];

            #pragma unroll
            for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
                score[m_tile] = mfma_m16n16k4_f32(
                    b,
                    a_regs[m_tile][d_tile],
                    score[m_tile]);
            }
        }

        #pragma unroll
        for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
            const int m = wave_m0 + m_tile * MFMA_M + lane_x;

            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int n =
                    block_n0 + n_tile * MFMA_N + MFMA_K * lane_q + r;
                if (m < M && n < N) {
                    const float x = fmaxf(score[m_tile][r], 0.0f);
                    local_sum += x;
                    local_sum2 += x * x;
                }
            }
        }
    }

    red_sum[tid] = local_sum;
    red_sum2[tid] = local_sum2;
    __syncthreads();

    for (int active = THREADS_PER_BLOCK; active > 1;
         active = (active + 1) >> 1) {
        const int half = (active + 1) >> 1;
        const int paired = active - half;
        if (tid < paired) {
            red_sum[tid] += red_sum[tid + half];
            red_sum2[tid] += red_sum2[tid + half];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_sum, static_cast<double>(red_sum[0]));
        atomicAdd(out_sum2, static_cast<double>(red_sum2[0]));
    }
}
