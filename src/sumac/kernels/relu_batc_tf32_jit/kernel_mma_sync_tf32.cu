#include <cuda/std/cstdint>

#ifndef MMA_SYNC_TF32_KERNEL_NAME
#define MMA_SYNC_TF32_KERNEL_NAME relu_bat_c_tf32_mma_sync
#endif

#ifndef MMA_SYNC_TF32_PACK_KERNEL_NAME
#define MMA_SYNC_TF32_PACK_KERNEL_NAME relu_bat_c_tf32_mma_sync_pack
#endif

#ifndef MMA_SYNC_TF32_PADDED_D
#define MMA_SYNC_TF32_PADDED_D 0
#endif

// SM80+ mma.sync variant of Y=ReLU(B A.T)C kernel.
//
// mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 performs D = A @ B + D, where A and B are TF32 and D is FP32.
//
// A and C are prepacked and converted to tf32 in the B-operand layout used by
// mma.sync.m16n8k8 with a short packing kernel to avoid having to convert them repeatedly 
// in each threadblock of the compute kernel.
//
// B is loaded from global memory into registers at the start of the compute kernel.
// To coalesce the loads from global memory, but also provide the register layout required by mma.sync,
// we stage it through shared memory in a swizzled pattern, then use ldmatrix to load into registers without bank conflicts.

static constexpr int MMA_M = 16;
static constexpr int MMA_N = 8;
static constexpr int MMA_K = 8;

static constexpr int WARP_M_ROWS = M_TILES * MMA_M;
static constexpr int THREADS_PER_BLOCK = (BM / WARP_M_ROWS) * 32;

static constexpr int K_TILES = D_f / MMA_K;
static constexpr int N_TILES = BN / MMA_N;
static constexpr int MMA_B_OPERAND_FRAGMENT_ELEMS = MMA_N * MMA_K;
static constexpr int LDMATRIX_WORDS = 32;

static constexpr int PACKED_TILE_ELEMS = BN * D_f;

static_assert(MMA_SYNC_TF32_STAGES >= 1 && MMA_SYNC_TF32_STAGES <= 3, "cp.async staging supports 1..3 stages");

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

__device__ __forceinline__ uint32_t* packed_tile(uint32_t* base, int stage)
{
    return base + stage * PACKED_TILE_ELEMS;
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

__device__ __forceinline__ uint32_t load_global_u32_cs(
    const void* gmem_ptr)
{
    const unsigned long long src_g = gmem_u64(gmem_ptr);
    uint32_t value;

    asm volatile(
        "ld.global.cs.u32 %0, [%1];\n"
        : "=r"(value)
        : "l"(src_g)
        : "memory");

    return value;
}

__device__ __forceinline__ uint4 load_global_u32x4_cs(
    const void* gmem_ptr)
{
    const unsigned long long src_g = gmem_u64(gmem_ptr);
    uint4 value;

    asm volatile(
        "ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
        : "l"(src_g)
        : "memory");

    return value;
}

__device__ __forceinline__ void store_shared_u32x4(
    void* smem_ptr,
    const uint4& value)
{
    const unsigned dst_s = smem_u32(smem_ptr);

    asm volatile(
        "st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
        :
        : "r"(dst_s),
          "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
        : "memory");
}

__device__ __forceinline__ void cp_async_commit_group()
{
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int KeepGroups>
__device__ __forceinline__ void cp_async_wait_group()
{
    asm volatile(
        "cp.async.wait_group %0;\n"
        :
        : "n"(KeepGroups)
        : "memory");
}

__device__ __forceinline__ void cp_async_wait_group(int keep_groups)
{
    switch (keep_groups) {
        case 0:
            cp_async_wait_group<0>();
            break;
        case 1:
            cp_async_wait_group<1>();
            break;
        default:
            cp_async_wait_group<2>();
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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("cvt.rn.tf32.f32 %0, %1;\n" : "=r"(y) : "f"(x)); 
#else
    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(y) : "f"(x));
#endif
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

__device__ __forceinline__ void ldmatrix_u32x4(
    uint32_t first[2],
    uint32_t second[2],
    const void* row_ptr)
{
    const uint32_t smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(row_ptr));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(first[0]), "=r"(first[1]),
          "=r"(second[0]), "=r"(second[1])
        : "r"(smem_addr)
        : "memory");
}

__device__ __forceinline__ void ldmatrix_u32x2(
    uint32_t out[2],
    const void* row_ptr)
{
    const uint32_t smem_addr =
        static_cast<uint32_t>(__cvta_generic_to_shared(row_ptr));

    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(out[0]), "=r"(out[1])
        : "r"(smem_addr)
        : "memory");
}

__device__ __forceinline__ int ldmatrix_group_word_offset(
    int major_tile,
    int minor_tile_even,
    int minor_tiles)
{
    return (major_tile * minor_tiles + minor_tile_even) * MMA_N * MMA_K;
}

__device__ __forceinline__ void load_b_operand_pair_from_smem(
    uint32_t first[2],
    uint32_t second[2],
    const uint32_t* __restrict__ packed_smem,
    int major_tile,
    int minor_tile_even,
    int minor_tiles)
{
    const int lane = threadIdx.x & 31;
    const int group_offset = ldmatrix_group_word_offset(major_tile, minor_tile_even, minor_tiles);

    // Lanes 0..7 provide the rows of matrix 0, lanes 8..15 matrix 1,
    // and so on. 
    const uint32_t* row_ptr = packed_smem + group_offset + 4 * lane;
    ldmatrix_u32x4(first, second, row_ptr);
}

__device__ __forceinline__ void load_b_operand_tail_from_smem(
    uint32_t out[2],
    const uint32_t* __restrict__ packed_smem,
    int major_tile,
    int minor_tile_even,
    int minor_tiles)
{
    const int lane = threadIdx.x & 31;
    const int group_offset = ldmatrix_group_word_offset(major_tile, minor_tile_even, minor_tiles);

    const uint32_t* row_ptr = packed_smem + group_offset + 4 * (lane & 15);
    ldmatrix_u32x2(out, row_ptr);
}

__device__ __forceinline__ int b_stage_slot_sw128(int row, int half)
{
    //128B swizzle
    const int band = row >> 2;
    const int chunk = 2 * (row & 3) + half;
    return 8 * band + (chunk ^ (band & 7));
}

// SW128 layout for one 16x8 FP32 tile:
//
// Split each 32B row into two 16B chunks and group four rows into
// each 128B band:
//
// band = row / 4
// logical_chunk = 2 * (row % 4) + half       // half = 0 or 1
// physical_chunk = logical_chunk ^ band
// slot = 8 * band + physical_chunk
//
// Logical-to-physical chunk order within bands:
//
// band 0: 0 1 2 3 4 5 6 7
// band 1: 1 0 3 2 5 4 7 6
// band 2: 2 3 0 1 6 7 4 5
// band 3: 3 2 1 0 7 6 5 4

__device__ __forceinline__ void load_B_swizzle_coalesced(
    uint32_t out[4],
    const float* __restrict__ B,
    uint32_t* __restrict__ warp_scratch,
    int tile_m0,
    int tile_k0,
    int M,
    int D)
{
    const int lane = threadIdx.x & 31;
    const int row = lane >> 1;
    const int half = lane & 1;
    const int m = tile_m0 + row;

    uint32_t* dst = warp_scratch + 4 * b_stage_slot_sw128(row, half);

    uint4 value = {0u, 0u, 0u, 0u};
    if (m < M) {
        if constexpr (MMA_SYNC_TF32_PADDED_D != 0) {
            const int d0 = tile_k0 + 4 * half;
            const float* row_src = B + static_cast<long long>(m) * D;

            if ((D & 3) == 0 && d0 + 3 < D) {
                value = load_global_u32x4_cs(row_src + d0);
            } else {
                if (d0 + 0 < D) {
                    value.x = load_global_u32_cs(row_src + d0 + 0);
                }
                if (d0 + 1 < D) {
                    value.y = load_global_u32_cs(row_src + d0 + 1);
                }
                if (d0 + 2 < D) {
                    value.z = load_global_u32_cs(row_src + d0 + 2);
                }
                if (d0 + 3 < D) {
                    value.w = load_global_u32_cs(row_src + d0 + 3);
                }
            }
        } else {
            const float* src = B + static_cast<long long>(m) * D + tile_k0 + 4 * half;
            value = load_global_u32x4_cs(src);
        }
    }

    store_shared_u32x4(dst, value);
    __syncwarp();

    uint32_t raw[4];
    const int source_row = lane & 15;
    const int source_half = lane >> 4;
    const uint32_t* row_ptr = warp_scratch + 4 * b_stage_slot_sw128(source_row, source_half);
    ldmatrix_u32x4(&raw[0], &raw[2], row_ptr);

    #pragma unroll
    for (int elem = 0; elem < 4; ++elem) {
        out[elem] = f32_to_tf32_bits(__uint_as_float(raw[elem]));
    }
}

__device__ __forceinline__ void store_b_operand_for_ldmatrix(
    uint32_t* __restrict__ packed,
    int major_tile,
    int minor_tile,
    int minor_tiles,
    int lane,
    uint2 fragment)
{
    const int minor_tile_even = minor_tile & ~1;
    const int fragment_in_group = minor_tile & 1;
    const int first_matrix = 2 * fragment_in_group;
    const int group_offset = ldmatrix_group_word_offset(major_tile, minor_tile_even, minor_tiles);

    packed[group_offset + (first_matrix + 0) * LDMATRIX_WORDS + lane] = fragment.x;
    packed[group_offset + (first_matrix + 1) * LDMATRIX_WORDS + lane] = fragment.y;
}


__device__ __forceinline__ void pack_tile_pair_tf32(
    const float* __restrict__ A,
    const float* __restrict__ C,
    uint32_t* __restrict__ A_packed,
    uint32_t* __restrict__ C_packed,
    int n0,
    int N,
    int D,
    int packed_pair_idx)
{
    const int lane = packed_pair_idx & 31;
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

    store_b_operand_for_ldmatrix(A_packed, inner_tile, n_tile, N_TILES, lane, a_pair);

    store_b_operand_for_ldmatrix(C_packed, n_tile, inner_tile, K_TILES, lane, c_pair);
}

__device__ __forceinline__ void issue_tile_copy(
    int stage,
    int tile,
    int tid,
    const uint32_t* __restrict__ A_packed_global,
    const uint32_t* __restrict__ C_packed_global,
    uint32_t* __restrict__ Apacked_smem,
    uint32_t* __restrict__ Cpacked_smem)
{
    const int bytes = PACKED_TILE_ELEMS * sizeof(uint32_t);
    const int chunks = bytes / 16;
    const unsigned char* tile_a = reinterpret_cast<const unsigned char*>(A_packed_global + tile * PACKED_TILE_ELEMS);
    const unsigned char* tile_c = reinterpret_cast<const unsigned char*>(C_packed_global + tile * PACKED_TILE_ELEMS);

    unsigned char* smem_a = reinterpret_cast<unsigned char*>(packed_tile(Apacked_smem, stage));
    unsigned char* smem_c = reinterpret_cast<unsigned char*>(packed_tile(Cpacked_smem, stage));

    #pragma unroll 1
    for (int chunk = tid; chunk < chunks; chunk += blockDim.x) {
        const int off = chunk * 16;
        cp_async_shared_global_16(smem_a + off, tile_a + off);
        cp_async_shared_global_16(smem_c + off, tile_c + off);
    }

    cp_async_commit_group();
}

__device__ __forceinline__ void compute_tile_phase1(
    const uint32_t* __restrict__ Apacked_smem,
    uint32_t b_regs[M_TILES][K_TILES][4],
    float s[M_TILES][N_TILES][4])
{
    #pragma unroll
    for (int k_tile = 0; k_tile < K_TILES; ++k_tile) {
        uint32_t A_mma_fragment[N_TILES][2];

        #pragma unroll
        for (int n_pair = 0; n_pair < N_TILES / 2; ++n_pair) {
            const int n_tile = 2 * n_pair;
            load_b_operand_pair_from_smem(
                A_mma_fragment[n_tile],
                A_mma_fragment[n_tile + 1],
                Apacked_smem,
                k_tile,
                n_tile,
                N_TILES);
        }

        if constexpr ((N_TILES & 1) != 0) {
            load_b_operand_tail_from_smem(
                A_mma_fragment[N_TILES - 1],
                Apacked_smem,
                k_tile,
                N_TILES - 1,
                N_TILES);
        }

        #pragma unroll
        for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {

            #pragma unroll
            for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {
                mma_m16n8k8_tf32(s[m_tile][n_tile], b_regs[m_tile][k_tile], A_mma_fragment[n_tile]);
            }
        }
    }
}

__device__ __forceinline__ void compute_tile_phase2(
    const uint32_t* __restrict__ Cpacked_smem,
    const float s[M_TILES][N_TILES][4],
    float y_regs[M_TILES][K_TILES][4])
{
    #pragma unroll
    for (int n_tile = 0; n_tile < N_TILES; ++n_tile) {

        uint32_t S_mma_fragment[M_TILES][4];
        #pragma unroll
        for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
            acc_frag_to_a_regs_relu_tf32( //MMA accumulator fragments and MMA A-operands have a different register layout.
                s[m_tile][n_tile],        //Combined with the packing kernel's column permutation, this helper converts between them, applies relu and converts to TF32.
                S_mma_fragment[m_tile]);
        }

        #pragma unroll
        for (int k_pair = 0; k_pair < K_TILES / 2; ++k_pair) {
            const int k_tile = 2 * k_pair;
            uint32_t C_mma_fragment[2][2];

            load_b_operand_pair_from_smem(
                C_mma_fragment[0],
                C_mma_fragment[1],
                Cpacked_smem,
                n_tile,
                k_tile,
                K_TILES);

            #pragma unroll
            for (int fragment = 0; fragment < 2; ++fragment) {
                #pragma unroll
                for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
                    mma_m16n8k8_tf32(
                        y_regs[m_tile][k_tile + fragment],
                        S_mma_fragment[m_tile],
                        C_mma_fragment[fragment]);
                }
            }
        }

        if constexpr ((K_TILES & 1) != 0) {
            uint32_t C_mma_fragment[2];
            load_b_operand_tail_from_smem(
                C_mma_fragment,
                Cpacked_smem,
                n_tile,
                K_TILES - 1,
                K_TILES);

            #pragma unroll
            for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
                mma_m16n8k8_tf32(
                    y_regs[m_tile][K_TILES - 1],
                    S_mma_fragment[m_tile],
                    C_mma_fragment);
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
    const int num_tiles = (N + BN - 1) / BN;
    const int tile = blockIdx.y;

    if (tile >= num_tiles) {
        return;
    }

    const int n0 = tile * BN;
    for (int packed_pair_idx = packed_pair_global_idx; packed_pair_idx < PACKED_TILE_ELEMS / 2; packed_pair_idx += packed_pair_stride) {
        pack_tile_pair_tf32(
            A,
            C,
            A_packed + tile * PACKED_TILE_ELEMS,
            C_packed + tile * PACKED_TILE_ELEMS,
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
    uint32_t* Cpacked_smem = Apacked_smem + MMA_SYNC_TF32_STAGES * PACKED_TILE_ELEMS;

    uint32_t b_regs[M_TILES][K_TILES][4];
    uint32_t* warp_b_scratch = Apacked_smem + warp * MMA_M * MMA_K;

    #pragma unroll
    for (int m_tile = 0; m_tile < M_TILES; ++m_tile) {
        #pragma unroll
        for (int k_tile = 0; k_tile < K_TILES; ++k_tile) {
            load_B_swizzle_coalesced(
                b_regs[m_tile][k_tile],
                B,
                warp_b_scratch,
                warp_m0 + m_tile * MMA_M,
                k_tile * MMA_K,
                M,
                D);
        }
    }

    __syncthreads();

    const int num_tiles = (N + BN - 1) / BN;
    if (num_tiles == 0) {
        return;
    }

    const int initial_tiles = num_tiles < MMA_SYNC_TF32_STAGES ? num_tiles : MMA_SYNC_TF32_STAGES;

    //We want to overlap MMAs with copying A/C from global memory to shared memory via cp.async.
    //Example for 2 stages:
    //We issue copies of A/C Panel 0 and Panel 1 ahead of the compute loop, setting up the pipeline.
    //Entering the pipeline loop, we ensure the Panel 0 copy is complete (cp.async.wait_group 1), then compute on Panel 0, issue copies of Panel 2, 
    //wait on Panel 1, compute on Panel 1 while Panel 2 is still in-flight etc.
    #pragma unroll
    for (int tile = 0; tile < MMA_SYNC_TF32_STAGES; ++tile) {
        if (tile < initial_tiles) {
            issue_tile_copy(
                tile,
                tile,
                tid,
                A_packed_global,
                C_packed_global,
                Apacked_smem,
                Cpacked_smem);
        }
    }
    __syncthreads();

    float y[M_TILES][K_TILES][4] = {0.f};

    for (int tile = 0; tile < num_tiles; ++tile) {
        const int newer_tiles_left = num_tiles - tile - 1;
        const int newer_groups = newer_tiles_left < (MMA_SYNC_TF32_STAGES - 1) ? newer_tiles_left : (MMA_SYNC_TF32_STAGES - 1);
        const int stage = tile % MMA_SYNC_TF32_STAGES;

        cp_async_wait_group(newer_groups);
        __syncthreads();

        float s[M_TILES][N_TILES][4] = {0.f};

        compute_tile_phase1(
            packed_tile(Apacked_smem, stage),
            b_regs,
            s);

        compute_tile_phase2(
            packed_tile(Cpacked_smem, stage),
            s,
            y);

        __syncthreads();

        const int next_tile = tile + MMA_SYNC_TF32_STAGES;

        if (next_tile < num_tiles) {
            issue_tile_copy(
                stage,
                next_tile,
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
