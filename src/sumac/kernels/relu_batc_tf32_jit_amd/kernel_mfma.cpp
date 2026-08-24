#ifndef MFMA_TF32_KERNEL_NAME
#define MFMA_TF32_KERNEL_NAME relu_bat_c_tf32_mfma
#endif

static constexpr int MFMA_M = 16;
static constexpr int MFMA_N = 16;
static constexpr int MFMA_K = 8;
static constexpr int WAVE_SIZE = 64;

static constexpr int MFMA_INPUTS_PER_LANE = 2;
static constexpr int MFMA_OUTPUTS_PER_LANE = 4;
static constexpr int LANE_GROUPS = WAVE_SIZE / MFMA_N;
static constexpr int SCORE_PAIRS =
    MFMA_OUTPUTS_PER_LANE / MFMA_INPUTS_PER_LANE;

static constexpr int WAVE_M_ROWS = M_TILES * MFMA_M;
static constexpr int WAVES_PER_BLOCK = BM / WAVE_M_ROWS;
static constexpr int THREADS_PER_BLOCK = WAVES_PER_BLOCK * WAVE_SIZE;

static constexpr int D_K_TILES = D_f / MFMA_K;
static constexpr int D_N_TILES = D_f / MFMA_N;
static constexpr int PANEL_N_TILES = BN / MFMA_N;
static constexpr int PANEL_ELEMS = BN * D_f;

static_assert(BM > 0 && BN > 0 && M_TILES > 0 && D_f > 0,
              "MFMA tile dimensions must be positive");
static_assert((WAVE_SIZE % MFMA_N) == 0,
              "MFMA lane groups must divide the wave");
static_assert(MFMA_INPUTS_PER_LANE * LANE_GROUPS == MFMA_K,
              "XF32 input fragment must cover K=8");
static_assert(MFMA_OUTPUTS_PER_LANE * LANE_GROUPS == MFMA_M,
              "MFMA accumulator fragment must cover 16 rows");
static_assert((MFMA_OUTPUTS_PER_LANE % MFMA_INPUTS_PER_LANE) == 0,
              "score fragment must split into two-value XF32 operands");
static_assert((BN % MFMA_N) == 0, "BN must be divisible by 16");
static_assert((D_f % MFMA_N) == 0, "D_f must be divisible by 16");
static_assert((D_f % MFMA_K) == 0, "D_f must be divisible by 8");
static_assert((BM % WAVE_M_ROWS) == 0,
              "BM must be divisible by M_TILES * 16");
static_assert(THREADS_PER_BLOCK > 0 && THREADS_PER_BLOCK <= 1024,
              "MFMA workgroup size must be in [64, 1024]");


using mfma_f32x2 = float __attribute__((ext_vector_type(2)));
using mfma_f32x4 = float __attribute__((ext_vector_type(4)));

__device__ __forceinline__ mfma_f32x4 mfma_zero_f32x4()
{
    return mfma_f32x4{0.0f, 0.0f, 0.0f, 0.0f};
}

__device__ __forceinline__ mfma_f32x4 mfma_m16n16k8_xf32(
    mfma_f32x2 a,
    mfma_f32x2 b,
    mfma_f32x4 accum)
{
    return __builtin_amdgcn_mfma_f32_16x16x8_xf32(
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
void MFMA_TF32_KERNEL_NAME(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    float* __restrict__ Y,
    int N,
    int M,
    int D)
{
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & (WAVE_SIZE - 1);
    const int wave = tid / WAVE_SIZE;
    const int lane_x = lane & (MFMA_N - 1);
    const int lane_q = lane / MFMA_N;

    const int block_m0 = static_cast<int>(blockIdx.x) * BM;
    const int wave_m0 = block_m0 + wave * WAVE_M_ROWS;

    extern __shared__ float dynamic_smem[];
    float* __restrict__ As = dynamic_smem;
    float* __restrict__ Cs = As + PANEL_ELEMS;

    mfma_f32x2 b_regs[M_TILES][D_K_TILES];

    #pragma unroll
    for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
        const int m = wave_m0 + m_tile * MFMA_M + lane_x;

        #pragma unroll
        for (int d_tile = 0; d_tile < D_K_TILES; ++d_tile) {
            const int d0 =
                d_tile * MFMA_K + MFMA_INPUTS_PER_LANE * lane_q;
            mfma_f32x2 b = mfma_f32x2{0.0f, 0.0f};
            if (m < M) {
                const long long row_offset = static_cast<long long>(m) * D;
                if (d0 < D) {
                    b[0] = B[row_offset + d0];
                }
                if (d0 + 1 < D) {
                    b[1] = B[row_offset + d0 + 1];
                }
            }
            b_regs[m_tile][d_tile] = b;
        }
    }

    mfma_f32x4 y[M_TILES][D_N_TILES];
    #pragma unroll
    for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
        #pragma unroll
        for (int d_tile = 0; d_tile < D_N_TILES; ++d_tile) {
            y[m_tile][d_tile] = mfma_zero_f32x4();
        }
    }

    for (int panel_n0 = 0; panel_n0 < N; panel_n0 += BN) {
        for (int idx = tid; idx < PANEL_ELEMS; idx += THREADS_PER_BLOCK) {
            const int panel_n = idx / D_f;
            const int d = idx - panel_n * D_f;
            const int n = panel_n0 + panel_n;

            float a = 0.0f;
            float c = 0.0f;
            if (n < N && d < D) {
                const long long offset = static_cast<long long>(n) * D + d;
                a = A[offset];
                c = C[offset];
            }
            As[d * BN + panel_n] = a;
            Cs[idx] = c;
        }
        __syncthreads();

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
                const int d0 =
                    d_tile * MFMA_K + MFMA_INPUTS_PER_LANE * lane_q;
                const mfma_f32x2 a = mfma_f32x2{
                    As[(d0 + 0) * BN + panel_n],
                    As[(d0 + 1) * BN + panel_n],
                };

                #pragma unroll
                for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
                    score[m_tile] = mfma_m16n16k8_xf32(
                        a,
                        b_regs[m_tile][d_tile],
                        score[m_tile]);
                }
            }

            #pragma unroll
            for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
                #pragma unroll
                for (int r = 0; r < MFMA_OUTPUTS_PER_LANE; ++r) {
                    score[m_tile][r] = fmaxf(score[m_tile][r], 0.0f);
                }
            }

            #pragma unroll
            for (int d_tile = 0; d_tile < D_N_TILES; ++d_tile) {
                const int d = d_tile * MFMA_N + lane_x;

                #pragma unroll
                for (int pair = 0; pair < SCORE_PAIRS; ++pair) {
                    const int panel_n0_pair =
                        n_tile * MFMA_N
                        + MFMA_OUTPUTS_PER_LANE * lane_q
                        + MFMA_INPUTS_PER_LANE * pair;
                    const mfma_f32x2 c = mfma_f32x2{
                        Cs[(panel_n0_pair + 0) * D_f + d],
                        Cs[(panel_n0_pair + 1) * D_f + d],
                    };

                    #pragma unroll
                    for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
                        const mfma_f32x2 s = mfma_f32x2{
                            score[m_tile][MFMA_INPUTS_PER_LANE * pair + 0],
                            score[m_tile][MFMA_INPUTS_PER_LANE * pair + 1],
                        };
                        y[m_tile][d_tile] = mfma_m16n16k8_xf32(
                            s,
                            c,
                            y[m_tile][d_tile]);
                    }
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
        #pragma unroll
        for (int d_tile = 0; d_tile < D_N_TILES; ++d_tile) {
            const int d = d_tile * MFMA_N + lane_x;

            #pragma unroll
            for (int r = 0; r < MFMA_OUTPUTS_PER_LANE; ++r) {
                const int m =
                    wave_m0
                    + m_tile * MFMA_M
                    + MFMA_OUTPUTS_PER_LANE * lane_q
                    + r;
                if (m < M && d < D) {
                    Y[static_cast<long long>(m) * D + d] =
                        y[m_tile][d_tile][r];
                }
            }
        }
    }
}
