#include <cuda/std/cstdint>

#ifndef MMA_SYNC_TF32_KERNEL_NAME
#define MMA_SYNC_TF32_KERNEL_NAME relu_bat_c_tf32_mma_sync
#endif

#ifndef MMA_SYNC_TF32_PACK_KERNEL_NAME
#define MMA_SYNC_TF32_PACK_KERNEL_NAME relu_bat_c_tf32_mma_sync_pack
#endif

#ifndef MMA_SYNC_TF32_STAGES
#define MMA_SYNC_TF32_STAGES 2
#endif

// SM80+ mma.sync variant of Y=ReLU(B A.T)C kernel.
//
//mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 performs D = A @ B + D, where A and B are TF32 and D is FP32.
//
// A and C are prepacked into the MMA B-operand layout used by
// mma.sync.m16n8k8. Each packed pair contains the two 32-bit TF32 words
// that one compute lane later loads with ld.shared.v2.u32.

static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 8;

static constexpr int WARP_NTHREADS = 32;
static constexpr int WARP_M_ROWS = M_TILES * MMA_M;
static constexpr int WARPS_PER_BLOCK = BM / WARP_M_ROWS;
static constexpr int THREADS_PER_BLOCK =
    WARPS_PER_BLOCK * WARP_NTHREADS;

static constexpr int K_TILES = D_f / MMA_K;
static constexpr int N_TILES = BN / MMA_N;
static constexpr int MMA_B_OPERAND_WORDS_PER_LANE = 2;
static constexpr int MMA_B_OPERAND_FRAGMENT_ELEMS =
    WARP_NTHREADS * MMA_B_OPERAND_WORDS_PER_LANE;
static constexpr int PACKED_PANEL_ELEMS =
    N_TILES * K_TILES * MMA_B_OPERAND_FRAGMENT_ELEMS;
static constexpr int PACKED_PANEL_PAIRS =
    PACKED_PANEL_ELEMS / MMA_B_OPERAND_WORDS_PER_LANE;
static constexpr int PACKED_BUFFER_ELEMS = MMA_SYNC_TF32_STAGES * PACKED_PANEL_ELEMS;
static constexpr int PACKED_BUFFER_BYTES =
    PACKED_BUFFER_ELEMS * sizeof(cuda::std::uint32_t);

static_assert(MMA_SYNC_TF32_STAGES >= 1 && MMA_SYNC_TF32_STAGES <= 3,
              "cp.async staging supports 1..3 stages");
static_assert(
    (PACKED_BUFFER_BYTES % 128) == 0,
    "packed buffers must stay aligned");

using uint32_t = cuda::std::uint32_t;

__device__ __forceinline__ unsigned smem_u32(const void* p)
{
    unsigned long long u64;
    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(u64) : "l"(p));
    return static_cast<unsigned>(u64);
}

__device__ __forceinline__ unsigned long long gmem_u64(const void* p)
{
    unsigned long long u64;
    asm volatile("cvta.to.global.u64 %0, %1;" : "=l"(u64) : "l"(p));
    return u64;
}

__device__ __forceinline__ unsigned char* align_smem_128(unsigned char* p)
{
    const unsigned long long addr = reinterpret_cast<unsigned long long>(p);
    const unsigned long long aligned = (addr + 127ull) & ~127ull;
    return reinterpret_cast<unsigned char*>(aligned);
}

__device__ __forceinline__ uint32_t* packed_panel(uint32_t* base, int buf)
{
    return base + buf * PACKED_PANEL_ELEMS;
}

__device__ __forceinline__ void cp_async_shared_global_16(
    void* smem_ptr,
    const void* gmem_ptr)
{
    const unsigned dst_s = smem_u32(smem_ptr);
    const unsigned long long src_g = gmem_u64(gmem_ptr);

    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"
        :
        : "r"(dst_s), "l"(src_g)
        : "memory");
}

__device__ __forceinline__ void cp_async_commit_group()
{
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

__device__ __forceinline__ void cp_async_wait_group(int keep_groups)
{
    switch (keep_groups) {
        case 0:
            asm volatile("cp.async.wait_group 0;\n" ::: "memory");
            break;
        case 1:
            asm volatile("cp.async.wait_group 1;\n" ::: "memory");
            break;
        case 2:
            asm volatile("cp.async.wait_group 2;\n" ::: "memory");
            break;
        default:
            asm volatile("cp.async.wait_group 3;\n" ::: "memory");
            break;
    }
}

__device__ __forceinline__ int perm_phys_to_logical_k8(int k_phys)
{
    return ((k_phys & 1) << 2) | (k_phys >> 1);
}

__device__ __forceinline__ uint32_t f32_to_tf32_bits(float x)
{
    uint32_t y;
    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void acc_frag_to_a_regs_relu_tf32(
    const float acc[4],
    uint32_t a[4])
{                                                 //Operand A-matrix has the following thread-fragment mapping, each thread holding 4 A values {a0,...,a3}
    a[0] = f32_to_tf32_bits(fmaxf(acc[0], 0.0f)); //row\col     0     1      2      3      4      5     6     7  
    a[1] = f32_to_tf32_bits(fmaxf(acc[2], 0.0f)); //      0  T0:a0  T1:a0  T2:a0  T3:a0 | T0:a2  T1:a2
    a[2] = f32_to_tf32_bits(fmaxf(acc[1], 0.0f)); //      1  T4:a0  T5:a0  T6:a0        |
    a[3] = f32_to_tf32_bits(fmaxf(acc[3], 0.0f)); //     ...                            |
}                                                 //      8  T0:a1  T1:a1  T2:a1  T3:a1 | T0:a3
                                                  //     ...
                                                  //      15 ...                                             T31:a3


                                                  //Accumulator matrix D has the following thread-fragment mapping, each thread holding 4 D values {d0,...,d3}
                                                  //row\col    0           1          2       3      4      5     6     7
                                                  //      0   T0:d0      T0:d1       T1:d0   T1:d1  T2:d0 
                                                  //      1   T4:d0       ...
                                                  //     ...
                                                  //      8   T0:d2      T0:d3       T1:d2   T1:d3
                                                  //     ...
                                                  //      15  ...                                                     T31:d3


__device__ __forceinline__ void mma_m16n8k8_tf32(
    float d[4],
    const uint32_t a[4],
    const uint32_t b[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );
}

__device__ __forceinline__ void lds_u32x2(uint32_t out[2], const void* ptr)
{
    const uint32_t smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(ptr));

    asm volatile(
        "ld.shared.v2.u32 {%0, %1}, [%2];\n"
        : "=r"(out[0]), "=r"(out[1])
        : "r"(smem_addr));
}

__device__ __forceinline__ int packed_fragment_word_offset(
    int n_tile,
    int inner_tile,
    int lane)
{
    return (((n_tile * K_TILES + inner_tile) * WARP_NTHREADS + lane) *
            MMA_B_OPERAND_WORDS_PER_LANE);
}

__device__ __forceinline__ void load_b_operand_from_Asmem(
    uint32_t b[2],
    const uint32_t* __restrict__ Apacked_smem,
    int n_tile,
    int k_tile)
{
    const int lane = threadIdx.x & 31;
    lds_u32x2(b, &Apacked_smem[packed_fragment_word_offset(n_tile, k_tile, lane)]);
}

__device__ __forceinline__ void load_b_operand_from_Csmem(
    uint32_t b[2],
    const uint32_t* __restrict__ Cpacked_smem,
    int n_tile,
    int k_tile)
{
    const int lane = threadIdx.x & 31;
    lds_u32x2(b, &Cpacked_smem[packed_fragment_word_offset(n_tile, k_tile, lane)]);
}

__device__ __forceinline__ void pack_panel_pair_tf32(
    const float* __restrict__ A,
    const float* __restrict__ C,
    uint32_t* __restrict__ A_packed,
    uint32_t* __restrict__ C_packed,
    int n0,
    int N,
    int D,
    int packed_pair_idx)
{
    const int lane = packed_pair_idx & (WARP_NTHREADS - 1);
    const int tile = packed_pair_idx >> 5;
    const int inner_tile = tile % K_TILES;
    const int n_tile = tile / K_TILES;

    const int nn_phys = lane >> 2;
    const int kk_low = lane & 3;

    const int a_n_local = n_tile * MMA_N + perm_phys_to_logical_k8(nn_phys);
    const int a_n = n0 + a_n_local;

    const int c_col = inner_tile * MMA_N + nn_phys;

    uint2 a_pair = {0u, 0u};
    uint2 c_pair = {0u, 0u};

    const int a_d0 = inner_tile * MMA_K + kk_low;
    const int a_d1 = a_d0 + 4;
    if (a_n < N && a_d0 < D) {
        a_pair.x = f32_to_tf32_bits(A[static_cast<long long>(a_n) * D + a_d0]);
    }
    if (a_n < N && a_d1 < D) {
        a_pair.y = f32_to_tf32_bits(A[static_cast<long long>(a_n) * D + a_d1]);
    }

    const int c_n0 = n0 + n_tile * MMA_N + kk_low;
    if (c_n0 < N && c_col < D) {
        c_pair.x = f32_to_tf32_bits(C[static_cast<long long>(c_n0) * D + c_col]);
    }
    const int c_n1 = c_n0 + 4;
    if (c_n1 < N && c_col < D) {
        c_pair.y = f32_to_tf32_bits(C[static_cast<long long>(c_n1) * D + c_col]);
    }

    reinterpret_cast<uint2*>(A_packed)[packed_pair_idx] = a_pair;
    reinterpret_cast<uint2*>(C_packed)[packed_pair_idx] = c_pair;
}

__device__ __forceinline__ void issue_panel_copy(
    int buf,
    int panel,
    int tid,
    const uint32_t* __restrict__ A_packed_global,
    const uint32_t* __restrict__ C_packed_global,
    uint32_t* __restrict__ Apacked_smem,
    uint32_t* __restrict__ Cpacked_smem)
{
    const int bytes = PACKED_PANEL_ELEMS * sizeof(uint32_t);
    const int chunks = bytes / 16;

    const unsigned char* panel_a = reinterpret_cast<const unsigned char*>(
        A_packed_global + panel * PACKED_PANEL_ELEMS);
    const unsigned char* panel_c = reinterpret_cast<const unsigned char*>(
        C_packed_global + panel * PACKED_PANEL_ELEMS);
    unsigned char* smem_a = reinterpret_cast<unsigned char*>(packed_panel(Apacked_smem, buf));
    unsigned char* smem_c = reinterpret_cast<unsigned char*>(packed_panel(Cpacked_smem, buf));

    #pragma unroll 1
    for (int chunk = tid; chunk < chunks; chunk += blockDim.x) {
        const int off = chunk * 16;
        cp_async_shared_global_16(smem_a + off, panel_a + off);
        cp_async_shared_global_16(smem_c + off, panel_c + off);
    }

    cp_async_commit_group();
}

__device__ __forceinline__ void compute_panel(
    uint32_t* __restrict__ Apacked_smem,
    uint32_t* __restrict__ Cpacked_smem,
    uint32_t b_regs[M_TILES][K_TILES][4],
    float y_regs[M_TILES][K_TILES][4],
    int panel)
{
    const int buf = panel % MMA_SYNC_TF32_STAGES;

    float s[M_TILES][N_TILES][4] = {0.f};

    #pragma unroll
    for (int k_tile = 0; k_tile < K_TILES; ++k_tile) {
        uint32_t A_mma_fragment[N_TILES][2];

        #pragma unroll
        for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
            load_b_operand_from_Asmem(
                A_mma_fragment[n_tile],
                packed_panel(Apacked_smem, buf),
                n_tile,
                k_tile);
        }

        #pragma unroll
        for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {

            #pragma unroll
            for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
                mma_m16n8k8_tf32(s[m_tile][n_tile], b_regs[m_tile][k_tile], A_mma_fragment[n_tile]);
            }
        }
    }

    #pragma unroll
    for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
        uint32_t S_mma_fragment[M_TILES][4];

        #pragma unroll
        for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
            acc_frag_to_a_regs_relu_tf32( //MMA accumulator fragments have a different register layout
                s[m_tile][n_tile],        //than MMA A-operand fragments, this helper converts them, applies relu and converts to TF32.
                S_mma_fragment[m_tile]);
        }

        #pragma unroll
        for (int k_tile = 0; k_tile < K_TILES; ++k_tile) {
            uint32_t C_mma_fragment[2];

            load_b_operand_from_Csmem(
                C_mma_fragment,
                packed_panel(Cpacked_smem, buf),
                n_tile,
                k_tile);

            #pragma unroll
            for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
                mma_m16n8k8_tf32(y_regs[m_tile][k_tile], S_mma_fragment[m_tile], C_mma_fragment);
            }
        }
    }
}

// KERNEL_START

extern "C" __global__
void MMA_SYNC_TF32_PACK_KERNEL_NAME(
    const float* __restrict__ A,
    const float* __restrict__ C,
    uint32_t* __restrict__ A_packed,
    uint32_t* __restrict__ C_packed,
    int N,
    int D)
{
    const int packed_pair_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int packed_pair_stride = blockDim.x * gridDim.x;
    const int num_panels = (N + BN - 1) / BN;
    const int panel = blockIdx.y;

    if (panel >= num_panels) {
        return;
    }

    const int n0 = panel * BN;
    for (int packed_pair_idx = packed_pair_global_idx;
         packed_pair_idx < PACKED_PANEL_PAIRS;
         packed_pair_idx += packed_pair_stride) {
        pack_panel_pair_tf32(
            A,
            C,
            A_packed + panel * PACKED_PANEL_ELEMS,
            C_packed + panel * PACKED_PANEL_ELEMS,
            n0,
            N,
            D,
            packed_pair_idx);
    }
}

extern "C" __global__
__launch_bounds__(THREADS_PER_BLOCK, 1)
void MMA_SYNC_TF32_KERNEL_NAME(
    const uint32_t* __restrict__ A_packed_global,
    const uint32_t* __restrict__ C_packed_global,
    const float* __restrict__ B,
    float* __restrict__ Y,
    int N,
    int M,
    int D)
{

    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int block_m0 = blockIdx.x * BM;
    const int warp_m0  = block_m0 + warp * WARP_M_ROWS;

    extern __shared__ unsigned char dynamic_smem[];
    unsigned char* packed_smem = align_smem_128(dynamic_smem);
    uint32_t* Apacked_smem = reinterpret_cast<uint32_t*>(packed_smem);
    uint32_t* Cpacked_smem = reinterpret_cast<uint32_t*>(packed_smem + PACKED_BUFFER_BYTES);

    uint32_t b_regs[M_TILES][K_TILES][4];
    const int group = lane >> 2;
    const int tid4  = lane & 3;

    #pragma unroll
    for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
        #pragma unroll
        for (int k_tile = 0; k_tile < K_TILES; ++k_tile) {
            const int r_base = warp * WARP_M_ROWS + m_tile * MMA_M;

            #pragma unroll
            for (int fragment_elem = 0; fragment_elem < 4; ++fragment_elem) {
                const int row = group + ((fragment_elem & 1) << 3);
                const int col = tid4  + ((fragment_elem >> 1) << 2);

                const int m = block_m0 + r_base + row;
                const int d = k_tile * MMA_K + col;

                float v = 0.0f;
                if (m < M && d < D) {
                    v = B[static_cast<long long>(m) * D + d];
                }

                b_regs[m_tile][k_tile][fragment_elem] = f32_to_tf32_bits(v);
            }
        }
    }

    const int num_panels = (N + BN - 1) / BN;
    if (num_panels == 0) {
        return;
    }

    const int initial_panels =
        num_panels < MMA_SYNC_TF32_STAGES ? num_panels : MMA_SYNC_TF32_STAGES;

    //We want to overlap MMAs with copying A/C from global memory to shared memory via cp.async.
    //Example for 2 stages:
    //We issue copies of A/C Panel 0 and Panel 1 ahead of the compute loop, setting up the pipeline.
    //Entering the pipeline loop, we ensure the Panel 0 copy is complete (cp.async.wait_group 1), then compute on Panel 0, issue copies of Panel 2, 
    //wait on Panel 1, compute on Panel 1 while Panel 2 is still in-flight etc.
    #pragma unroll
    for (int panel = 0; panel < MMA_SYNC_TF32_STAGES; ++panel) {
        if (panel < initial_panels) {
            issue_panel_copy(
                panel,
                panel,
                tid,
                A_packed_global,
                C_packed_global,
                Apacked_smem,
                Cpacked_smem);
        }
    }
    __syncthreads();

    float y[M_TILES][K_TILES][4] = {0.f};

    for (int panel = 0; panel < num_panels; ++panel) {
        const int newer_panels_left = num_panels - panel - 1;
        const int newer_groups =
            newer_panels_left < (MMA_SYNC_TF32_STAGES - 1) ?
                newer_panels_left : (MMA_SYNC_TF32_STAGES - 1);

        //Wait until only num_stages - 1 panels are still in-flight in the pipeline. (Or fewer once we approach the loop end)
        cp_async_wait_group(newer_groups);
        __syncthreads();

        //Issue the MMAs on the current panel. 
        //The helper performs S = B @ A_panel.T, then accumulates y += relu(S) @ C_panel using m16n8k8 mma.sync TF32 tensor core instructions
        compute_panel(Apacked_smem, Cpacked_smem, b_regs, y, panel);

        __syncthreads();
        //Place next A/C copies into the pipeline.
        const int next_panel = panel + MMA_SYNC_TF32_STAGES;
        if (next_panel < num_panels) {
            issue_panel_copy(
                next_panel % MMA_SYNC_TF32_STAGES,
                next_panel,
                tid,
                A_packed_global,
                C_packed_global,
                Apacked_smem,
                Cpacked_smem);
        }
    }

    #pragma unroll
    for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
        #pragma unroll
        for (int k_tile = 0; k_tile < K_TILES; ++k_tile) {
            const int row0 = warp_m0 + m_tile * MMA_M + (lane >> 2);
            const int row1 = row0 + 8;
            const int col  = k_tile * MMA_N + 2 * (lane & 3);

            if (row0 < M) {
                reinterpret_cast<float2*>(&Y[static_cast<long long>(row0) * D_f + col])[0] =
                    make_float2(y[m_tile][k_tile][0], y[m_tile][k_tile][1]);
            }
            if (row1 < M) {
                reinterpret_cast<float2*>(&Y[static_cast<long long>(row1) * D_f + col])[0] =
                    make_float2(y[m_tile][k_tile][2], y[m_tile][k_tile][3]);
            }
        }
    }
}
