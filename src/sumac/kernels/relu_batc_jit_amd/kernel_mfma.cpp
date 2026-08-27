#ifndef MFMA_FP32_KERNEL_NAME
#define MFMA_FP32_KERNEL_NAME relu_bat_c_fp32_mfma
#endif

#ifndef MFMA_FP32_PACK_KERNEL_NAME
#define MFMA_FP32_PACK_KERNEL_NAME relu_bat_c_fp32_mfma_pack
#endif

#ifndef MFMA_FP32_STAGES
#define MFMA_FP32_STAGES 2
#endif

// FP32 MFMA implementation of Y = ReLU(B A.T) C.
//
// A and C are first packed into the operand layouts consumed by
// v_mfma_f32_16x16x4_f32. We then use an async copy pipeline to
// overlap the loads to LDS with MFMA instructions.

static constexpr int MFMA_M = 16;
static constexpr int MFMA_N = 16;
static constexpr int MFMA_K = 4;
static constexpr int MFMA_OUTPUTS_PER_LANE = 4;
static constexpr int WAVE_SIZE = 64;
static constexpr int PIPELINE_STAGES = MFMA_FP32_STAGES;

static constexpr int WAVE_M_ROWS = M_TILES * MFMA_M;
static constexpr int WAVES_PER_BLOCK = BM / WAVE_M_ROWS;
static constexpr int THREADS_PER_BLOCK = WAVES_PER_BLOCK * WAVE_SIZE;

static constexpr int D_K_TILES = D_f / MFMA_K;
static constexpr int D_N_TILES = D_f / MFMA_N;
static constexpr int PANEL_N_TILES = BN / MFMA_N;
static constexpr int PANEL_ELEMS = BN * D_f;
static constexpr int PANEL_WAVE_CHUNKS = PANEL_ELEMS / WAVE_SIZE;
static constexpr int COPY_CHUNKS_PER_WAVE =
    PANEL_WAVE_CHUNKS / WAVES_PER_BLOCK;
static constexpr int COPY_EXTRA_WAVES =
    PANEL_WAVE_CHUNKS % WAVES_PER_BLOCK;
static constexpr int VMCNT_MAX = 63;

static_assert(BM > 0 && BN > 0 && M_TILES > 0 && D_f > 0,
              "MFMA tile dimensions must be positive");
static_assert((BN % MFMA_N) == 0, "BN must be divisible by 16");
static_assert((D_f % MFMA_N) == 0, "D_f must be divisible by 16");
static_assert((BM % WAVE_M_ROWS) == 0,
              "BM must be divisible by M_TILES * 16");
static_assert(THREADS_PER_BLOCK > 0 && THREADS_PER_BLOCK <= 1024,
              "MFMA workgroup size must be in [64, 1024]");
static_assert(PIPELINE_STAGES >= 1 && PIPELINE_STAGES <= 3,
              "MFMA direct-copy pipeline supports 1..3 stages");
static_assert((PANEL_ELEMS % WAVE_SIZE) == 0,
              "packed panels must contain whole wave-sized copy chunks");
static_assert(PANEL_N_TILES * D_K_TILES * WAVE_SIZE == PANEL_ELEMS,
              "packed A fragments must cover one panel exactly");
static_assert(
    PANEL_N_TILES * D_N_TILES * MFMA_OUTPUTS_PER_LANE * WAVE_SIZE
        == PANEL_ELEMS,
    "packed C fragments must cover one panel exactly");

using mfma_f32x4 = float __attribute__((ext_vector_type(4)));
using mfma_i32x4 = int __attribute__((ext_vector_type(4)));

union mfma_buffer_resource {
    mfma_i32x4 content;
    struct {
        const float* address;
        unsigned int range;
        unsigned int config;
    } fields;
};

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
        0, // cbsz: no A-lane broadcast
        0, // abid
        0  // blgp: no B-lane permutation
    );
}

__device__ __forceinline__ void store_global_f32x4(
    float* destination,
    mfma_f32x4 value)
{
    __builtin_memcpy(destination, &value, sizeof(value));
}

__device__ __forceinline__ mfma_i32x4 make_buffer_resource(
    const float* wave_uniform_base)
{
    mfma_buffer_resource resource;
    resource.content = mfma_i32x4{0, 0, 0, 0};
    resource.fields.address = wave_uniform_base;
    resource.fields.range = PANEL_ELEMS * sizeof(float);
    // gfx9 buffer-resource format used by ROCm's Composable Kernel.
    resource.fields.config = 0x00020000u;
    return resource.content;
}

__device__ __forceinline__ void direct_copy_wave_dword(
    mfma_i32x4 source_resource,
    unsigned int source_byte_offset,
    float* wave_lds_base)
{
    const unsigned int lds_address = static_cast<unsigned int>(
        reinterpret_cast<unsigned long long>(wave_lds_base));
    const unsigned int wave_lds_address =
        __builtin_amdgcn_readfirstlane(lds_address);

    // CDNA buffer_loads load directly into LDS, without staging through registers,
    // similar to PTX's cp.async. 
    asm volatile(
        "s_mov_b32 m0, %0\n\t"
        "s_nop 0\n\t"
        "buffer_load_dword %1, %2, 0 offen lds\n\t"
        :
        : "s"(wave_lds_address),
          "v"(source_byte_offset),
          "s"(source_resource)
        : "memory");
}

template <int KeepOperations>
__device__ __forceinline__ void wait_direct_copy_operations()
{
    static_assert(KeepOperations >= 0 && KeepOperations <= VMCNT_MAX,
                  "invalid VM_CNT wait threshold");
    asm volatile(
        "s_waitcnt vmcnt(%0)\n\t"
        :
        : "n"(KeepOperations)
        : "memory");
}

template <int YoungerPanels>
__device__ __forceinline__ void wait_direct_copy_groups(int wave)
{
    static_assert(YoungerPanels >= 0 && YoungerPanels <= 2,
                  "pipeline can retain at most two younger panels");

    constexpr int keep_low_unclamped =
        YoungerPanels * 2 * COPY_CHUNKS_PER_WAVE;
    constexpr int keep_high_unclamped =
        YoungerPanels * 2 * (COPY_CHUNKS_PER_WAVE + 1);
    constexpr int keep_low =
        keep_low_unclamped < VMCNT_MAX ? keep_low_unclamped : VMCNT_MAX;
    constexpr int keep_high =
        keep_high_unclamped < VMCNT_MAX ? keep_high_unclamped : VMCNT_MAX;
    if constexpr (COPY_EXTRA_WAVES == 0) {
        wait_direct_copy_operations<keep_low>();
    } else if (wave < COPY_EXTRA_WAVES) {
        wait_direct_copy_operations<keep_high>();
    } else {
        wait_direct_copy_operations<keep_low>();
    }
}

__device__ __forceinline__ float* panel_stage(
    float* pipeline_base,
    int stage)
{
    return pipeline_base + stage * PANEL_ELEMS;
}

__device__ __forceinline__ void issue_panel_copy(
    int tile,
    int stage,
    int wave,
    int lane,
    const float* __restrict__ A_packed_global,
    const float* __restrict__ C_packed_global,
    float* __restrict__ A_pipeline,
    float* __restrict__ C_pipeline)
{
    const float* tile_a = A_packed_global +
        static_cast<long long>(tile) * PANEL_ELEMS;
    const float* tile_c = C_packed_global +
        static_cast<long long>(tile) * PANEL_ELEMS;
    const mfma_i32x4 a_resource = make_buffer_resource(tile_a);
    const mfma_i32x4 c_resource = make_buffer_resource(tile_c);

    float* stage_a = panel_stage(A_pipeline, stage);
    float* stage_c = panel_stage(C_pipeline, stage);

    #pragma unroll 1
    for (int chunk = wave;
         chunk < PANEL_WAVE_CHUNKS;
         chunk += WAVES_PER_BLOCK) {
        const int element = chunk * WAVE_SIZE + lane;
        const unsigned int source_byte_offset =
            static_cast<unsigned int>(element * sizeof(float));
        direct_copy_wave_dword(
            a_resource,
            source_byte_offset,
            stage_a + chunk * WAVE_SIZE);
        direct_copy_wave_dword(
            c_resource,
            source_byte_offset,
            stage_c + chunk * WAVE_SIZE);
    }
}

// KERNEL_START

extern "C" __global__
void MFMA_FP32_PACK_KERNEL_NAME(
    const float* __restrict__ A,
    const float* __restrict__ C,
    float* __restrict__ A_packed,
    float* __restrict__ C_packed,
    int N,
    int D)
{
    const int logical_idx =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
        + static_cast<int>(threadIdx.x);
    const int tile = static_cast<int>(blockIdx.y);
    if (logical_idx >= PANEL_ELEMS) {
        return;
    }

    const long long tile_offset = static_cast<long long>(tile) * PANEL_ELEMS;
    const int panel_n0 = tile * BN;
    const int panel_n = logical_idx / D_f;
    const int d = logical_idx - panel_n * D_f;
    const int n = panel_n0 + panel_n;
    float a = 0.0f;
    float c = 0.0f;
    if (n < N && d < D) {
        const long long source_offset = static_cast<long long>(n) * D + d;
        a = A[source_offset];
        c = C[source_offset];
    }


    const int a_n_tile = panel_n / MFMA_N;
    const int a_lane_x = panel_n % MFMA_N;
    const int a_d_tile = d / MFMA_K;
    const int a_lane_q = d % MFMA_K;
    const int a_lane = a_lane_q * MFMA_N + a_lane_x;
    const int a_fragment = a_n_tile * D_K_TILES + a_d_tile;
    const int a_packed_idx = a_fragment * WAVE_SIZE + a_lane;
    A_packed[tile_offset + a_packed_idx] = a;

    const int c_n_tile = panel_n / MFMA_N;
    const int c_n_in_tile = panel_n % MFMA_N;
    const int c_lane_q = c_n_in_tile / MFMA_K;
    const int c_r = c_n_in_tile % MFMA_K;
    const int c_d_tile = d / MFMA_N;
    const int c_lane_x = d % MFMA_N;
    const int c_lane = c_lane_q * MFMA_N + c_lane_x;
    const int c_fragment =
        (c_n_tile * D_N_TILES + c_d_tile)
        * MFMA_OUTPUTS_PER_LANE
        + c_r;
    const int c_packed_idx = c_fragment * WAVE_SIZE + c_lane;
    C_packed[tile_offset + c_packed_idx] = c;
}

extern "C" __global__
__launch_bounds__(THREADS_PER_BLOCK, 1)
void MFMA_FP32_KERNEL_NAME(
    const float* __restrict__ A_packed_global,
    const float* __restrict__ C_packed_global,
    const float* __restrict__ B,
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
    float* __restrict__ A_pipeline = dynamic_smem;
    float* __restrict__ C_pipeline =
        A_pipeline + PIPELINE_STAGES * PANEL_ELEMS;

    float b_regs[M_TILES][D_K_TILES];

    #pragma unroll
    for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
        const int m = wave_m0 + m_tile * MFMA_M + lane_x;

        #pragma unroll
        for (int d_tile = 0; d_tile < D_K_TILES; ++d_tile) {
            const int d = d_tile * MFMA_K + lane_q;
            b_regs[m_tile][d_tile] =
                (m < M && d < D)
                    ? B[static_cast<long long>(m) * D + d]
                    : 0.0f;
        }
    }

    // Phase two accumulates Y.T so each lane owns four adjacent D values.
    mfma_f32x4 y_transposed[M_TILES][D_N_TILES];
    #pragma unroll
    for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
        #pragma unroll
        for (int d_tile = 0; d_tile < D_N_TILES; ++d_tile) {
            y_transposed[m_tile][d_tile] = mfma_zero_f32x4();
        }
    }

    //need to manually place a wait here or LLVM will collapse the pipeline with added vmcnt(0) before the matmul loop
    //Adding a wait here ensures that B is fully resident in registers before we start the async copy pipeline for A/C
    //so LLVM has no reason to add waits.
    __builtin_amdgcn_s_waitcnt(0x0f70);

    const int num_panels = (N + BN - 1) / BN;

    #pragma unroll
    for (int panel = 0; panel < PIPELINE_STAGES; ++panel) {
        if (panel < num_panels) {
            issue_panel_copy(
                panel,
                panel,
                wave,
                lane,
                A_packed_global,
                C_packed_global,
                A_pipeline,
                C_pipeline);
        }
    }
    if constexpr (PIPELINE_STAGES == 3) {
        if (num_panels >= 3) {
            wait_direct_copy_groups<2>(wave);
        } else if (num_panels >= 2) {
            wait_direct_copy_groups<1>(wave);
        } else {
            wait_direct_copy_groups<0>(wave);
        }
    } else if constexpr (PIPELINE_STAGES == 2) {
        if (num_panels >= 2) {
            wait_direct_copy_groups<1>(wave);
        } else {
            wait_direct_copy_groups<0>(wave);
        }
    } else {
        wait_direct_copy_groups<0>(wave);
    }
    __syncthreads();

    for (int panel = 0; panel < num_panels; ++panel) {
        const int stage = panel % PIPELINE_STAGES;

        const float* __restrict__ As = panel_stage(A_pipeline, stage);
        const float* __restrict__ Cs = panel_stage(C_pipeline, stage);

        #pragma unroll
        for (int n_tile = 0; n_tile < PANEL_N_TILES; ++n_tile) {
            mfma_f32x4 score[M_TILES];
            #pragma unroll
            for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
                score[m_tile] = mfma_zero_f32x4();
            }

            #pragma unroll
            for (int d_tile = 0; d_tile < D_K_TILES; ++d_tile) {
                const int fragment = n_tile * D_K_TILES + d_tile;
                const float a = As[fragment * WAVE_SIZE + lane];

                #pragma unroll
                for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
                    score[m_tile] = mfma_m16n16k4_f32(
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
                #pragma unroll
                for (int r = 0; r < MFMA_OUTPUTS_PER_LANE; ++r) {
                    const int fragment =
                        (n_tile * D_N_TILES + d_tile)
                        * MFMA_OUTPUTS_PER_LANE
                        + r;
                    const float c = Cs[fragment * WAVE_SIZE + lane];

                    #pragma unroll
                    for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
                        y_transposed[m_tile][d_tile] = mfma_m16n16k4_f32(
                            c,
                            score[m_tile][r],
                            y_transposed[m_tile][d_tile]);
                    }
                }
            }
        }

        if (panel + 1 < num_panels) {
            if constexpr (PIPELINE_STAGES == 1) {
                __syncthreads();
                issue_panel_copy(
                    panel + 1,
                    0,
                    wave,
                    lane,
                    A_packed_global,
                    C_packed_global,
                    A_pipeline,
                    C_pipeline);
                wait_direct_copy_groups<0>(wave);
                __syncthreads();
            } else {
                if constexpr (PIPELINE_STAGES == 3) {
                    if (panel + 2 < num_panels) {
                        wait_direct_copy_groups<1>(wave);
                    } else {
                        wait_direct_copy_groups<0>(wave);
                    }
                } else {
                    wait_direct_copy_groups<0>(wave);
                }
                __syncthreads();

                const int refill_panel = panel + PIPELINE_STAGES;
                if (refill_panel < num_panels) {
                    issue_panel_copy(
                        refill_panel,
                        stage,
                        wave,
                        lane,
                        A_packed_global,
                        C_packed_global,
                        A_pipeline,
                        C_pipeline);
                }
            }
        }
    }

    #pragma unroll
    for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
        #pragma unroll
        for (int d_tile = 0; d_tile < D_N_TILES; ++d_tile) {
            const int m = wave_m0 + m_tile * MFMA_M + lane_x;
            const int d0 =
                d_tile * MFMA_N + MFMA_OUTPUTS_PER_LANE * lane_q;

            if (m < M && d0 < D) {
                float* destination =
                    Y + static_cast<long long>(m) * D + d0;
                if (d0 + MFMA_OUTPUTS_PER_LANE <= D) {
                    store_global_f32x4(
                        destination,
                        y_transposed[m_tile][d_tile]);
                } else {
                    #pragma unroll
                    for (int r = 0; r < MFMA_OUTPUTS_PER_LANE; ++r) {
                        if (d0 + r < D) {
                            destination[r] =
                                y_transposed[m_tile][d_tile][r];
                        }
                    }
                }
            }
        }
    }
}
