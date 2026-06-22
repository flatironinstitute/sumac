#include <cuda/std/cstdint>

#ifndef WGMMA_TF32_KERNEL_NAME
#define WGMMA_TF32_KERNEL_NAME relu_bat_c_tf32_wgmma
#endif

#ifndef WGMMA_TF32_PACK_KERNEL_NAME
#define WGMMA_TF32_PACK_KERNEL_NAME relu_bat_c_tf32_wgmma_pack
#endif


// Hopper/SM90+ warp-group async MMA variant of Y=ReLU(B A.T)C
//
// A and C are row-major contiguous [N, D_f] tensors, B is row-major
// contiguous [M, D_f], and Y is row-major contiguous [M, D_f].
//
// This kernel does a two-level contraction:
//     S = ReLU(B @ A.T)
//     Y = S @ C
//
// Important for WGMMA:
//  One compute unit is a warpgroup (128 threads), not one warp.
//  WGMMA B operands must come from shared-memory descriptors, so A/C panels
//  are converted to TF32 and prepacked once in a separate launch.
//  The main kernel uses one producer warpgroup to move prepacked A/C panels
//  to shared memory with cp.async.bulk copies while the remaining
//  warpgroups consume the previous stage with WGMMA.

static constexpr int WGMMA_M = 64;
static constexpr int WGMMA_S_N = WGMMA_S_N_SHAPE;
static constexpr int WGMMA_Y_N = WGMMA_Y_N_SHAPE;
static constexpr int WGMMA_K = 8;
static constexpr int WGMMA_S_ACC_REGS = WGMMA_S_N / 2;
static constexpr int WGMMA_Y_ACC_REGS = WGMMA_Y_N / 2;

static constexpr int WARP_NTHREADS = 32;
static constexpr int WARPGROUP_NTHREADS = 4 * WARP_NTHREADS;
static constexpr int WARPS_PER_WARPGROUP = 4;
static constexpr int COMPUTE_WARPGROUPS_PER_BLOCK = BM / WGMMA_M;
static constexpr int THREADS_PER_BLOCK = (COMPUTE_WARPGROUPS_PER_BLOCK + 1) * WARPGROUP_NTHREADS;

static constexpr int K_TILES = D_f / WGMMA_K;
static constexpr int OUT_TILES = D_f / WGMMA_Y_N;
static constexpr int N_TILES = BN / WGMMA_S_N;
static constexpr int S_SUBK_TILES = WGMMA_S_N / WGMMA_K;
static constexpr int PANEL_ELEMS = BN * D_f;

// No-swizzle TF32 K-major descriptor tiles for WGMMA B operands.
static constexpr int GMMA_TF32_PER_128B = 4;
static constexpr int WGMMA_B_SBO_ELEMS = 8 * GMMA_TF32_PER_128B;
static constexpr int WGMMA_A_LBO_ELEMS = WGMMA_M * GMMA_TF32_PER_128B;
static constexpr int WGMMA_S_B_LBO_ELEMS = WGMMA_S_N * GMMA_TF32_PER_128B;
static constexpr int WGMMA_Y_B_LBO_ELEMS = WGMMA_Y_N * GMMA_TF32_PER_128B;
static constexpr int WGMMA_A_TILE_ELEMS = 2 * WGMMA_A_LBO_ELEMS;
static constexpr int WGMMA_S_B_TILE_ELEMS = 2 * WGMMA_S_B_LBO_ELEMS;
static constexpr int WGMMA_Y_B_TILE_ELEMS = 2 * WGMMA_Y_B_LBO_ELEMS;
static constexpr cuda::std::uint64_t GMMA_DESC_START_MASK = 0x3fffull;

__device__ static constexpr cuda::std::uint64_t desc_units_from_elems(int elems)
{
    return ((static_cast<cuda::std::uint64_t>(elems) * 4) >> 4) &
           GMMA_DESC_START_MASK;
}

__device__ static constexpr cuda::std::uint64_t desc_const(int lbo_elems, int sbo_elems)
{
    return (desc_units_from_elems(lbo_elems) << 16) |
           (desc_units_from_elems(sbo_elems) << 32);
}

static constexpr cuda::std::uint64_t WGMMA_A_DESC_CONST = desc_const(WGMMA_A_LBO_ELEMS, WGMMA_B_SBO_ELEMS);
static constexpr cuda::std::uint64_t WGMMA_S_B_DESC_CONST = desc_const(WGMMA_S_B_LBO_ELEMS, WGMMA_B_SBO_ELEMS);
static constexpr cuda::std::uint64_t WGMMA_Y_B_DESC_CONST = desc_const(WGMMA_Y_B_LBO_ELEMS, WGMMA_B_SBO_ELEMS);

static constexpr cuda::std::uint64_t WGMMA_S_B_K_DESC_DELTA = (static_cast<cuda::std::uint64_t>(WGMMA_S_B_TILE_ELEMS) * 4) >> 4;
static constexpr cuda::std::uint64_t WGMMA_S_B_NB_DESC_DELTA = K_TILES * WGMMA_S_B_K_DESC_DELTA;

static constexpr cuda::std::uint64_t WGMMA_Y_B_OUT_DESC_DELTA = (static_cast<cuda::std::uint64_t>(WGMMA_Y_B_TILE_ELEMS) * 4) >> 4;
static constexpr cuda::std::uint64_t WGMMA_Y_B_PANEL_K_DESC_DELTA = OUT_TILES * WGMMA_Y_B_OUT_DESC_DELTA;

static constexpr cuda::std::uint64_t WGMMA_A_K_DESC_DELTA = (static_cast<cuda::std::uint64_t>(WGMMA_A_TILE_ELEMS) * 4) >> 4;
static constexpr cuda::std::uint64_t B_SMEM_WG_DESC_DELTA = K_TILES * WGMMA_A_K_DESC_DELTA;

static constexpr int PACKED_PANEL_ELEMS = 2 * PANEL_ELEMS;
static constexpr int B_SMEM_WG_ELEMS = K_TILES * WGMMA_A_TILE_ELEMS;
static constexpr int B_PACK_ROW_BLOCKS_PER_WG = WARPS_PER_WARPGROUP;
static constexpr int B_PACK_ROWS_PER_WARP = WGMMA_M / B_PACK_ROW_BLOCKS_PER_WG;
static constexpr int B_PACK_TILES_PER_WG = K_TILES * B_PACK_ROW_BLOCKS_PER_WG;
static constexpr int PANEL_BYTES = PANEL_ELEMS * sizeof(cuda::std::uint32_t);

static constexpr cuda::std::uint64_t PANEL_DESC_DELTA = (PANEL_BYTES >> 4);
static constexpr cuda::std::uint64_t PACKED_PANEL_DESC_DELTA = ((2 * PANEL_BYTES) >> 4);
static constexpr int PACK_LOAD_COLS = ((D_f >= 32) && ((D_f % 16) == 0)) ? 16 : WGMMA_K;
static constexpr int PACK_LOAD_ROWS = WARP_NTHREADS / PACK_LOAD_COLS;

static_assert(COMPUTE_WARPGROUPS_PER_BLOCK <= 7, "named barriers support up to seven compute warpgroups");


using uint32_t = cuda::std::uint32_t;
using uint64_t = cuda::std::uint64_t;

__device__ __forceinline__ unsigned char* align_smem_128(unsigned char* p)
{
    const unsigned long long addr = reinterpret_cast<unsigned long long>(p);
    const unsigned long long aligned = (addr + 127ull) & ~127ull;
    return reinterpret_cast<unsigned char*>(aligned);
}

__device__ __forceinline__ unsigned shared_addr_u32(const void* ptr)
{
    unsigned long long addr64;
    asm volatile(
        "cvta.to.shared.u64 %0, %1;\n"
        : "=l"(addr64)
        : "l"(ptr));
    return static_cast<unsigned>(addr64);
}

__device__ __forceinline__ uint64_t shared_desc_start(const void* ptr)
{
    return (static_cast<uint64_t>(shared_addr_u32(ptr)) >> 4) &
           GMMA_DESC_START_MASK;
}

__device__ __forceinline__ unsigned long long global_addr_u64(const void* ptr)
{
    unsigned long long addr64;
    asm volatile(
        "cvta.to.global.u64 %0, %1;\n"
        : "=l"(addr64)
        : "l"(ptr));
    return addr64;
}

__device__ __forceinline__ void fence_proxy_async_shared_cta()
{
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrier_init_shared(
    unsigned long long* barrier,
    int count)
{
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :
        : "r"(shared_addr_u32(barrier)), "r"(count)
        : "memory");
}

__device__ __forceinline__ unsigned long long mbarrier_arrive_expect_tx_shared(
    unsigned long long* barrier,
    int bytes)
{
    unsigned long long state;
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
        "%0, [%1], %2;\n"
        : "=l"(state)
        : "r"(shared_addr_u32(barrier)), "r"(bytes)
        : "memory");
    return state;
}

__device__ __forceinline__ void mbarrier_wait_shared(
    unsigned long long* barrier,
    unsigned long long state)
{
    const unsigned barrier_addr = shared_addr_u32(barrier);
    unsigned done;
    do {
        asm volatile(
            "{ .reg .pred p;\n"
            "  mbarrier.try_wait.acquire.cta.shared::cta.b64 "
            "p, [%1], %2;\n"
            "  selp.u32 %0, 1, 0, p;\n"
            "}\n"
            : "=r"(done)
            : "r"(barrier_addr), "l"(state)
            : "memory");
    } while (!done);
}

__device__ __forceinline__ void mbarrier_arrive_shared(
    unsigned long long* barrier)
{
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
        :
        : "r"(shared_addr_u32(barrier))
        : "memory");
}

__device__ __forceinline__ void cp_async_bulk_shared_global(
    void* dst_shared,
    const void* src_global,
    int bytes,
    unsigned long long* complete_barrier)
{
    asm volatile(
        "cp.async.bulk.shared::cluster.global."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];\n"
        :
        : "r"(shared_addr_u32(dst_shared)),
          "l"(global_addr_u64(src_global)),
          "r"(bytes),
          "r"(shared_addr_u32(complete_barrier))
        : "memory");
}

__device__ __forceinline__ void wgmma_fence()
{
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_group()
{
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait_group_0()
{
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

template <int Regs>
__device__ __forceinline__ void setmaxnreg_dec()
{
    asm volatile(
        "setmaxnreg.dec.sync.aligned.u32 %0;\n"
        :
        : "n"(Regs)
        : "memory");
}

__device__ __forceinline__ bool elect_sync(unsigned membermask)
{
    unsigned is_leader;
    asm volatile(
        "{ .reg .pred p;\n"
        "  elect.sync _|p, %1;\n"
        "  selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(is_leader)
        : "r"(membermask));
    return is_leader != 0;
}

__device__ __forceinline__ bool elect_one_from_warpgroup_warp0(int local_tid)
{
    const unsigned full_warp_mask = 0xffffffffu;
    const int local_warp = local_tid >> 5;
    const int uniform_local_warp = __shfl_sync(full_warp_mask, local_warp, 0);

    return uniform_local_warp == 0 && elect_sync(full_warp_mask);
}

__device__ __forceinline__ int perm_logical_to_phys_k8(int k_logical)
{
    return ((k_logical & 3) << 1) | (k_logical >> 2);
}

__device__ __forceinline__ int perm_logical_to_phys_n(int n_logical)
{
    return (n_logical & ~7) | perm_logical_to_phys_k8(n_logical & 7);
}

__device__ __forceinline__ int wgmma_b_smem_index_from_coords(
    int k,
    int n,
    int leading_elems)
{
    return (k >> 2) * leading_elems +
           (n >> 3) * WGMMA_B_SBO_ELEMS +
           (n & 7) * GMMA_TF32_PER_128B +
           (k & 3);
}

__device__ __forceinline__ uint32_t tf32_from_float(float v)
{
    uint32_t out;
    asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(out) : "f"(v));
    return out;
}

__device__ __forceinline__ uint32_t tf32_relu_from_float(float v)
{
    uint32_t out;
    asm volatile("cvt.rn.relu.tf32.f32 %0, %1;\n" : "=r"(out) : "f"(v));
    return out;
}

__device__ __forceinline__ uint4 tf32_from_float4(float4 v)
{
    uint4 out;
    out.x = tf32_from_float(v.x);
    out.y = tf32_from_float(v.y);
    out.z = tf32_from_float(v.z);
    out.w = tf32_from_float(v.w);
    return out;
}

__device__ __forceinline__ void store_shared_u32x4(
    uint32_t* ptr,
    uint4 v)
{
    asm volatile(
        "st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
        :
        : "r"(shared_addr_u32(ptr)), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
        : "memory");
}

__device__ __forceinline__ void sync_compute_warpgroup(int compute_wg)
{
    switch (compute_wg) {
    case 0:
        asm volatile("bar.sync 1, 128;\n" ::: "memory");
        break;
#if COMPUTE_WARPGROUPS_PER_BLOCK > 1
    case 1:
        asm volatile("bar.sync 2, 128;\n" ::: "memory");
        break;
#endif
#if COMPUTE_WARPGROUPS_PER_BLOCK > 2
    case 2:
        asm volatile("bar.sync 3, 128;\n" ::: "memory");
        break;
#endif
#if COMPUTE_WARPGROUPS_PER_BLOCK > 3
    case 3:
        asm volatile("bar.sync 4, 128;\n" ::: "memory");
        break;
#endif
#if COMPUTE_WARPGROUPS_PER_BLOCK > 4
    case 4:
        asm volatile("bar.sync 5, 128;\n" ::: "memory");
        break;
#endif
#if COMPUTE_WARPGROUPS_PER_BLOCK > 5
    case 5:
        asm volatile("bar.sync 6, 128;\n" ::: "memory");
        break;
#endif
#if COMPUTE_WARPGROUPS_PER_BLOCK > 6
    case 6:
        asm volatile("bar.sync 7, 128;\n" ::: "memory");
        break;
#endif
    }
}

__device__ __forceinline__ void pack_coords_from_linear(
    int idx,
    int& n_local,
    int& d)
{
    const int elem = idx % WARP_NTHREADS;
    const int tile = idx / WARP_NTHREADS;
    const int row = elem / PACK_LOAD_COLS;
    const int col = elem - row * PACK_LOAD_COLS;
    const int col_tile = tile % (D_f / PACK_LOAD_COLS);
    const int row_tile = tile / (D_f / PACK_LOAD_COLS);

    n_local = row_tile * PACK_LOAD_ROWS + row;
    d = col_tile * PACK_LOAD_COLS + col;
}

__device__ __forceinline__ void pack_panel_element_tf32(
    const float* __restrict__ A,
    const float* __restrict__ C,
    uint32_t* __restrict__ A_packed,
    uint32_t* __restrict__ C_packed,
    int n0,
    int N,
    int raw_idx)
{
    int n_local;
    int d;
    pack_coords_from_linear(raw_idx, n_local, d);

    float av = 0.0f;
    float cv = 0.0f;
    const int n = n0 + n_local;
    if (n < N) {
        const long long offset = static_cast<long long>(n) * D_f + d;
        av = A[offset];
        cv = C[offset];
    }

    const int nb = n_local / WGMMA_S_N;
    const int n_in_tile = n_local - nb * WGMMA_S_N;
    const int panel_k = n_local / WGMMA_K;
    const int k_in_panel = n_local - panel_k * WGMMA_K;
    const int d_tile = d / WGMMA_K;
    const int d_in_tile = d - d_tile * WGMMA_K;
    const int out_tile = d / WGMMA_Y_N;
    const int d_in_out_tile = d - out_tile * WGMMA_Y_N;

    const int a_elem = wgmma_b_smem_index_from_coords(
        d_in_tile,
        perm_logical_to_phys_n(n_in_tile),
        WGMMA_S_B_LBO_ELEMS);
    const int c_elem = wgmma_b_smem_index_from_coords(
        k_in_panel,
        d_in_out_tile,
        WGMMA_Y_B_LBO_ELEMS);

    const uint32_t av_tf32 = tf32_from_float(av);
    const uint32_t cv_tf32 = tf32_from_float(cv);

    A_packed[(nb * K_TILES + d_tile) * WGMMA_S_B_TILE_ELEMS + a_elem] = av_tf32;
    C_packed[(panel_k * OUT_TILES + out_tile) * WGMMA_Y_B_TILE_ELEMS + c_elem] = cv_tf32;
}

#define WGMMA_D_LIST_16 "{%0, %1, %2, %3, %4, %5, %6, %7}"
#define WGMMA_A_LIST_16 "{%8, %9, %10, %11}"
#define WGMMA_DESC_16 "%12"
#define WGMMA_SCALE_16 "%13"
#define WGMMA_SS_DESC_A_16 "%8"
#define WGMMA_SS_DESC_B_16 "%9"
#define WGMMA_SS_SCALE_16 "%10"
#define WGMMA_D_OPERANDS_16(d) \
    "+f"((d)[0]), "+f"((d)[1]), "+f"((d)[2]), "+f"((d)[3]), \
    "+f"((d)[4]), "+f"((d)[5]), "+f"((d)[6]), "+f"((d)[7])

#define WGMMA_D_LIST_32 \
    "{%0, %1, %2, %3, %4, %5, %6, %7, " \
    "%8, %9, %10, %11, %12, %13, %14, %15}"
#define WGMMA_A_LIST_32 "{%16, %17, %18, %19}"
#define WGMMA_DESC_32 "%20"
#define WGMMA_SCALE_32 "%21"
#define WGMMA_SS_DESC_A_32 "%16"
#define WGMMA_SS_DESC_B_32 "%17"
#define WGMMA_SS_SCALE_32 "%18"
#define WGMMA_D_OPERANDS_32(d) \
    "+f"((d)[0]), "+f"((d)[1]), "+f"((d)[2]), "+f"((d)[3]), \
    "+f"((d)[4]), "+f"((d)[5]), "+f"((d)[6]), "+f"((d)[7]), \
    "+f"((d)[8]), "+f"((d)[9]), "+f"((d)[10]), "+f"((d)[11]), \
    "+f"((d)[12]), "+f"((d)[13]), "+f"((d)[14]), "+f"((d)[15])

#define WGMMA_D_LIST_64 \
    "{%0, %1, %2, %3, %4, %5, %6, %7, " \
    "%8, %9, %10, %11, %12, %13, %14, %15, " \
    "%16, %17, %18, %19, %20, %21, %22, %23, " \
    "%24, %25, %26, %27, %28, %29, %30, %31}"
#define WGMMA_A_LIST_64 "{%32, %33, %34, %35}"
#define WGMMA_DESC_64 "%36"
#define WGMMA_SCALE_64 "%37"
#define WGMMA_SS_DESC_A_64 "%32"
#define WGMMA_SS_DESC_B_64 "%33"
#define WGMMA_SS_SCALE_64 "%34"
#define WGMMA_D_OPERANDS_64(d) \
    "+f"((d)[0]), "+f"((d)[1]), "+f"((d)[2]), "+f"((d)[3]), \
    "+f"((d)[4]), "+f"((d)[5]), "+f"((d)[6]), "+f"((d)[7]), \
    "+f"((d)[8]), "+f"((d)[9]), "+f"((d)[10]), "+f"((d)[11]), \
    "+f"((d)[12]), "+f"((d)[13]), "+f"((d)[14]), "+f"((d)[15]), \
    "+f"((d)[16]), "+f"((d)[17]), "+f"((d)[18]), "+f"((d)[19]), \
    "+f"((d)[20]), "+f"((d)[21]), "+f"((d)[22]), "+f"((d)[23]), \
    "+f"((d)[24]), "+f"((d)[25]), "+f"((d)[26]), "+f"((d)[27]), \
    "+f"((d)[28]), "+f"((d)[29]), "+f"((d)[30]), "+f"((d)[31])

#define WGMMA_D_LIST_128 \
    "{%0, %1, %2, %3, %4, %5, %6, %7, " \
    "%8, %9, %10, %11, %12, %13, %14, %15, " \
    "%16, %17, %18, %19, %20, %21, %22, %23, " \
    "%24, %25, %26, %27, %28, %29, %30, %31, " \
    "%32, %33, %34, %35, %36, %37, %38, %39, " \
    "%40, %41, %42, %43, %44, %45, %46, %47, " \
    "%48, %49, %50, %51, %52, %53, %54, %55, " \
    "%56, %57, %58, %59, %60, %61, %62, %63}"
#define WGMMA_A_LIST_128 "{%64, %65, %66, %67}"
#define WGMMA_DESC_128 "%68"
#define WGMMA_SCALE_128 "%69"
#define WGMMA_SS_DESC_A_128 "%64"
#define WGMMA_SS_DESC_B_128 "%65"
#define WGMMA_SS_SCALE_128 "%66"
#define WGMMA_D_OPERANDS_128(d) \
    "+f"((d)[0]), "+f"((d)[1]), "+f"((d)[2]), "+f"((d)[3]), \
    "+f"((d)[4]), "+f"((d)[5]), "+f"((d)[6]), "+f"((d)[7]), \
    "+f"((d)[8]), "+f"((d)[9]), "+f"((d)[10]), "+f"((d)[11]), \
    "+f"((d)[12]), "+f"((d)[13]), "+f"((d)[14]), "+f"((d)[15]), \
    "+f"((d)[16]), "+f"((d)[17]), "+f"((d)[18]), "+f"((d)[19]), \
    "+f"((d)[20]), "+f"((d)[21]), "+f"((d)[22]), "+f"((d)[23]), \
    "+f"((d)[24]), "+f"((d)[25]), "+f"((d)[26]), "+f"((d)[27]), \
    "+f"((d)[28]), "+f"((d)[29]), "+f"((d)[30]), "+f"((d)[31]), \
    "+f"((d)[32]), "+f"((d)[33]), "+f"((d)[34]), "+f"((d)[35]), \
    "+f"((d)[36]), "+f"((d)[37]), "+f"((d)[38]), "+f"((d)[39]), \
    "+f"((d)[40]), "+f"((d)[41]), "+f"((d)[42]), "+f"((d)[43]), \
    "+f"((d)[44]), "+f"((d)[45]), "+f"((d)[46]), "+f"((d)[47]), \
    "+f"((d)[48]), "+f"((d)[49]), "+f"((d)[50]), "+f"((d)[51]), \
    "+f"((d)[52]), "+f"((d)[53]), "+f"((d)[54]), "+f"((d)[55]), \
    "+f"((d)[56]), "+f"((d)[57]), "+f"((d)[58]), "+f"((d)[59]), \
    "+f"((d)[60]), "+f"((d)[61]), "+f"((d)[62]), "+f"((d)[63])

#define WGMMA_A_DESC_SCALE_OPERANDS(a, desc_b, scale_d) \
    "r"((a)[0]), "r"((a)[1]), "r"((a)[2]), "r"((a)[3]), \
    "l"(desc_b), "r"(scale_d)

#define WGMMA_DESC_DESC_SCALE_OPERANDS(desc_a, desc_b, scale_d) \
    "l"(desc_a), "l"(desc_b), "r"(scale_d)

template <int N>
struct wgmmaTF32;

template <>
struct wgmmaTF32<16> {
    static __device__ __forceinline__ void rs(
        float (&d)[8],
        const uint32_t (&a)[4],
        uint64_t desc_b,
        int scale_d)
    {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, " WGMMA_SCALE_16 ", 0;\n"
            "  wgmma.mma_async.sync.aligned.m64n16k8.f32.tf32.tf32 "
            "  " WGMMA_D_LIST_16 ", "
            "  " WGMMA_A_LIST_16 ", "
            "  " WGMMA_DESC_16 ", "
            "  p, 1, 1;\n"
            "}\n"
            : WGMMA_D_OPERANDS_16(d)
            : WGMMA_A_DESC_SCALE_OPERANDS(a, desc_b, scale_d)
            : "memory");
    }

    static __device__ __forceinline__ void ss(
        float (&d)[8],
        uint64_t desc_a,
        uint64_t desc_b,
        int scale_d)
    {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, " WGMMA_SS_SCALE_16 ", 0;\n"
            "  wgmma.mma_async.sync.aligned.m64n16k8.f32.tf32.tf32 "
            "  " WGMMA_D_LIST_16 ", "
            "  " WGMMA_SS_DESC_A_16 ", "
            "  " WGMMA_SS_DESC_B_16 ", "
            "  p, 1, 1;\n"
            "}\n"
            : WGMMA_D_OPERANDS_16(d)
            : WGMMA_DESC_DESC_SCALE_OPERANDS(desc_a, desc_b, scale_d)
            : "memory");
    }
};

template <>
struct wgmmaTF32<32> {
    static __device__ __forceinline__ void rs(
        float (&d)[16],
        const uint32_t (&a)[4],
        uint64_t desc_b,
        int scale_d)
    {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, " WGMMA_SCALE_32 ", 0;\n"
            "  wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32 "
            "  " WGMMA_D_LIST_32 ", "
            "  " WGMMA_A_LIST_32 ", "
            "  " WGMMA_DESC_32 ", "
            "  p, 1, 1;\n"
            "}\n"
            : WGMMA_D_OPERANDS_32(d)
            : WGMMA_A_DESC_SCALE_OPERANDS(a, desc_b, scale_d)
            : "memory");
    }

    static __device__ __forceinline__ void ss(
        float (&d)[16],
        uint64_t desc_a,
        uint64_t desc_b,
        int scale_d)
    {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, " WGMMA_SS_SCALE_32 ", 0;\n"
            "  wgmma.mma_async.sync.aligned.m64n32k8.f32.tf32.tf32 "
            "  " WGMMA_D_LIST_32 ", "
            "  " WGMMA_SS_DESC_A_32 ", "
            "  " WGMMA_SS_DESC_B_32 ", "
            "  p, 1, 1;\n"
            "}\n"
            : WGMMA_D_OPERANDS_32(d)
            : WGMMA_DESC_DESC_SCALE_OPERANDS(desc_a, desc_b, scale_d)
            : "memory");
    }
};

template <>
struct wgmmaTF32<64> {
    static __device__ __forceinline__ void rs(
        float (&d)[32],
        const uint32_t (&a)[4],
        uint64_t desc_b,
        int scale_d)
    {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, " WGMMA_SCALE_64 ", 0;\n"
            "  wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 "
            "  " WGMMA_D_LIST_64 ", "
            "  " WGMMA_A_LIST_64 ", "
            "  " WGMMA_DESC_64 ", "
            "  p, 1, 1;\n"
            "}\n"
            : WGMMA_D_OPERANDS_64(d)
            : WGMMA_A_DESC_SCALE_OPERANDS(a, desc_b, scale_d)
            : "memory");
    }

    static __device__ __forceinline__ void ss(
        float (&d)[32],
        uint64_t desc_a,
        uint64_t desc_b,
        int scale_d)
    {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, " WGMMA_SS_SCALE_64 ", 0;\n"
            "  wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32 "
            "  " WGMMA_D_LIST_64 ", "
            "  " WGMMA_SS_DESC_A_64 ", "
            "  " WGMMA_SS_DESC_B_64 ", "
            "  p, 1, 1;\n"
            "}\n"
            : WGMMA_D_OPERANDS_64(d)
            : WGMMA_DESC_DESC_SCALE_OPERANDS(desc_a, desc_b, scale_d)
            : "memory");
    }
};

template <>
struct wgmmaTF32<128> {
    static __device__ __forceinline__ void rs(
        float (&d)[64],
        const uint32_t (&a)[4],
        uint64_t desc_b,
        int scale_d)
    {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, " WGMMA_SCALE_128 ", 0;\n"
            "  wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32 "
            "  " WGMMA_D_LIST_128 ", "
            "  " WGMMA_A_LIST_128 ", "
            "  " WGMMA_DESC_128 ", "
            "  p, 1, 1;\n"
            "}\n"
            : WGMMA_D_OPERANDS_128(d)
            : WGMMA_A_DESC_SCALE_OPERANDS(a, desc_b, scale_d)
            : "memory");
    }

    static __device__ __forceinline__ void ss(
        float (&d)[64],
        uint64_t desc_a,
        uint64_t desc_b,
        int scale_d)
    {
        asm volatile(
            "{\n"
            "  .reg .pred p;\n"
            "  setp.ne.b32 p, " WGMMA_SS_SCALE_128 ", 0;\n"
            "  wgmma.mma_async.sync.aligned.m64n128k8.f32.tf32.tf32 "
            "  " WGMMA_D_LIST_128 ", "
            "  " WGMMA_SS_DESC_A_128 ", "
            "  " WGMMA_SS_DESC_B_128 ", "
            "  p, 1, 1;\n"
            "}\n"
            : WGMMA_D_OPERANDS_128(d)
            : WGMMA_DESC_DESC_SCALE_OPERANDS(desc_a, desc_b, scale_d)
            : "memory");
    }
};

// KERNEL_START

extern "C" __global__
void WGMMA_TF32_PACK_KERNEL_NAME(
    const float* __restrict__ A,
    const float* __restrict__ C,
    uint32_t* __restrict__ A_packed,
    uint32_t* __restrict__ C_packed,
    int N)
{
    const int raw_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int raw_stride = blockDim.x * gridDim.x;
    const int num_panels = (N + BN - 1) / BN;
    const int panel = blockIdx.y;

    if (panel >= num_panels) {
        return;
    }

    const int n0 = panel * BN;
    for (int raw_idx = raw_global_idx; raw_idx < PANEL_ELEMS; raw_idx += raw_stride) {
        pack_panel_element_tf32(
            A,
            C,
            A_packed + panel * PANEL_ELEMS,
            C_packed + panel * PANEL_ELEMS,
            n0,
            N,
            raw_idx);
    }
}

#if !WGMMA_TF32_PACK_ONLY
extern "C" __global__
__launch_bounds__(THREADS_PER_BLOCK, 1)
void WGMMA_TF32_KERNEL_NAME(
    const uint32_t* __restrict__ A_packed_global,
    const uint32_t* __restrict__ C_packed_global,
    const float* __restrict__ B,
    float* __restrict__ Y,
    int N,
    int M,
    int D)
{

    (void)D;

    const int tid = threadIdx.x;
    const int warpgroup_id = tid >> 7;
    const bool is_producer_warpgroup = warpgroup_id == COMPUTE_WARPGROUPS_PER_BLOCK;
    const int producer_tid = tid - COMPUTE_WARPGROUPS_PER_BLOCK * WARPGROUP_NTHREADS;
    const int consumer_tid = tid;
    const int tid128 = consumer_tid & 127;
    const int lane = consumer_tid & 31;
    const int warp_in_wg = tid128 >> 5;
    const int compute_wg = warpgroup_id;
    const int tid4 = lane & 3;
    const int row8 = lane >> 2;

    const int block_m0 = blockIdx.x * BM;
    const int wg_m0 = block_m0 + compute_wg * WGMMA_M;
    const int frag_row0 = warp_in_wg * 16 + row8;
    const int frag_col0 = tid4 << 1;

    extern __shared__ unsigned char dynamic_smem[];
    unsigned char* smem = align_smem_128(dynamic_smem);
    uint32_t* smem_panels = reinterpret_cast<uint32_t*>(smem);
#if WGMMA_FIRST_MMA_SS
    uint32_t* b_smem = smem_panels + SMEM_COPY_STAGES * PACKED_PANEL_ELEMS;
#endif
    const uint64_t smem_panels_start = shared_desc_start(smem_panels);
    __shared__ unsigned long long stage_ready[SMEM_COPY_STAGES];
    __shared__ unsigned long long stage_done[SMEM_COPY_STAGES];
    __shared__ unsigned long long stage_ready_state[SMEM_COPY_STAGES];
    __shared__ unsigned long long stage_done_state[SMEM_COPY_STAGES];
    __shared__ int stage_ready_panel[SMEM_COPY_STAGES];

    if (tid == 0) {
        #pragma unroll
        for (int stage = 0; stage < SMEM_COPY_STAGES; ++stage) {
            mbarrier_init_shared(&stage_ready[stage], 1);
            mbarrier_init_shared(&stage_done[stage], COMPUTE_WARPGROUPS_PER_BLOCK + 1);
            stage_ready_state[stage] = 0;
            stage_done_state[stage] = 0;
            stage_ready_panel[stage] = -1;
        }
    }
    fence_proxy_async_shared_cta();
    __syncthreads();

    const int num_panels = (N + BN - 1) / BN;
    if (num_panels == 0) {
        return;
    }

#if WGMMA_USE_SETMAXNREG
    if (is_producer_warpgroup) {
        setmaxnreg_dec<WGMMA_PRODUCER_MAX_REGS>();
    }
#endif

    if (is_producer_warpgroup) {
        if (elect_one_from_warpgroup_warp0(producer_tid)) {
            #pragma unroll 1
            for (int panel = 0; panel < num_panels; ++panel) {

                const int stage = panel % SMEM_COPY_STAGES;
                uint32_t* stage_a = smem_panels + stage * PACKED_PANEL_ELEMS;
                uint32_t* stage_c = stage_a + PANEL_ELEMS;

                if (panel >= SMEM_COPY_STAGES) {
                    mbarrier_wait_shared(
                        &stage_done[stage],
                        stage_done_state[stage]);
                }
                stage_done_state[stage] = mbarrier_arrive_expect_tx_shared(&stage_done[stage], 0);
                stage_ready_state[stage] = mbarrier_arrive_expect_tx_shared(&stage_ready[stage], 2 * PANEL_BYTES);
                __threadfence_block();

                stage_ready_panel[stage] = panel;
                cp_async_bulk_shared_global(
                    stage_a,
                    A_packed_global + panel * PANEL_ELEMS,
                    PANEL_BYTES,
                    &stage_ready[stage]);

                cp_async_bulk_shared_global(
                    stage_c,
                    C_packed_global + panel * PANEL_ELEMS,
                    PANEL_BYTES,
                    &stage_ready[stage]);
            }
        }
        return;
    }

#if WGMMA_FIRST_MMA_SS
    #pragma unroll 1
    for (int pack_tile = warp_in_wg; pack_tile < B_PACK_TILES_PER_WG; pack_tile += B_PACK_ROW_BLOCKS_PER_WG) {
        const int row_block = pack_tile % B_PACK_ROW_BLOCKS_PER_WG;
        const int kp = pack_tile / B_PACK_ROW_BLOCKS_PER_WG;

        const int half_warp = lane >> 4;
        const int lane16 = lane & 15;
        const int row_in_8 = lane16 & 7;
        const int k_group = lane16 >> 3;
        const int row = row_block * B_PACK_ROWS_PER_WARP + half_warp * 8 + row_in_8;
        const int d = kp * WGMMA_K + k_group * GMMA_TF32_PER_128B;
        const int m = wg_m0 + row;

        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (m < M) {
            v = reinterpret_cast<const float4*>(&B[static_cast<long long>(m) * D_f + d])[0];
        }

        const uint4 v_tf32 = tf32_from_float4(v);
        const int elem = wgmma_b_smem_index_from_coords(k_group * GMMA_TF32_PER_128B, row, WGMMA_A_LBO_ELEMS);
        store_shared_u32x4(&b_smem[compute_wg * B_SMEM_WG_ELEMS + kp * WGMMA_A_TILE_ELEMS + elem], v_tf32);
    }
    sync_compute_warpgroup(compute_wg);

    uint64_t b_desc[K_TILES];
    const uint64_t b_smem_start = (smem_panels_start + SMEM_COPY_STAGES * PACKED_PANEL_DESC_DELTA + compute_wg * B_SMEM_WG_DESC_DELTA) & GMMA_DESC_START_MASK;
    #pragma unroll
    for (int kp = 0; kp < K_TILES; ++kp) {
        const uint64_t b_start = (b_smem_start + kp * WGMMA_A_K_DESC_DELTA) & GMMA_DESC_START_MASK;
        b_desc[kp] = b_start | WGMMA_A_DESC_CONST;
    }
#else
    uint32_t b_regs[K_TILES][4];

    #pragma unroll
    for (int kp = 0; kp < K_TILES; ++kp) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int row = frag_row0 + ((q & 1) << 3);
            const int col = tid4 + ((q >> 1) << 2);

            const int m = wg_m0 + row;
            const int d = kp * WGMMA_K + col;

            float v = 0.0f;
            if (m < M) {
                v = B[static_cast<long long>(m) * D_f + d];
            }

            b_regs[kp][q] = tf32_from_float(v);
        }
    }
#endif

    float y[OUT_TILES][WGMMA_Y_ACC_REGS] = {0.f};

    for (int panel = 0; panel < num_panels; ++panel) {
        const int stage = panel % SMEM_COPY_STAGES;

        while (((volatile int*)stage_ready_panel)[stage] != panel) {}

        const unsigned long long ready_state = ((volatile unsigned long long*)stage_ready_state)[stage];
        mbarrier_wait_shared(&stage_ready[stage], ready_state);
        fence_proxy_async_shared_cta();

        //We use WGMMA (warp group mma D += A*B) in RS or SS mode, depending on rank D.
        //RS: accumulator/output matrix lives in registers, A lives in registers, B in smem.
        //SS: accumulator/output matrix lives in registers, A and B in smem.
        //We need to construct WGMMA descriptors that encode the shared memory location and layout of smem matrix fragments.
        //descriptors are 64-bit:
        //Bits 0-13 are the start address of the matrix in 16-byte units
        //Bits 16-29 are the leading dimension byte offset in 16-byte units
        //Bits 32-45 are the stride dimension byte offset in 16-byte units
        //Bits 49-51 are the matrix base offset if swizzling is used
        //Bits 62-63 specifies the swizzling mode
        //We use no-swizzle: desc = start_16B_units | (lbo_16B_units << 16) | (sbo_16B_units << 32);
        //A matrix descriptors below
        const uint64_t stage_a_start = (smem_panels_start + stage * PACKED_PANEL_DESC_DELTA) & GMMA_DESC_START_MASK;
        const uint64_t stage_a_desc_base = stage_a_start | WGMMA_S_B_DESC_CONST;
        const uint64_t stage_c_desc_base = (stage_a_start + PANEL_DESC_DELTA) | WGMMA_Y_B_DESC_CONST;

        float s_pipe[WGMMA_S_ACC_REGS] = {0.f};

        #pragma unroll
        for (int nb = 0; nb < N_TILES; ++nb) {
            uint64_t a_desc = stage_a_desc_base + nb * WGMMA_S_B_NB_DESC_DELTA;

            wgmma_fence();
#if WGMMA_FIRST_MMA_SS
            wgmmaTF32<WGMMA_S_N>::ss(s_pipe, b_desc[0], a_desc, 0);
#else
            wgmmaTF32<WGMMA_S_N>::rs(s_pipe, b_regs[0], a_desc, 0);
#endif
            #pragma unroll
            for (int kp = 1; kp < K_TILES; ++kp) {
                a_desc += WGMMA_S_B_K_DESC_DELTA;
#if WGMMA_FIRST_MMA_SS
                wgmmaTF32<WGMMA_S_N>::ss(s_pipe, b_desc[kp], a_desc, 1);
#else
                wgmmaTF32<WGMMA_S_N>::rs(s_pipe, b_regs[kp], a_desc, 1);
#endif
            }
            wgmma_commit_group();
            wgmma_wait_group_0();

            uint32_t s_as_a_frag[S_SUBK_TILES][4];
            #pragma unroll
            for (int sub_k = 0; sub_k < S_SUBK_TILES; ++sub_k) {
                const int base = sub_k * 4;
                s_as_a_frag[sub_k][0] = tf32_relu_from_float(s_pipe[base + 0]);
                s_as_a_frag[sub_k][1] = tf32_relu_from_float(s_pipe[base + 2]);
                s_as_a_frag[sub_k][2] = tf32_relu_from_float(s_pipe[base + 1]);
                s_as_a_frag[sub_k][3] = tf32_relu_from_float(s_pipe[base + 3]);
            }

            #pragma unroll
            for (int sub_k = 0; sub_k < S_SUBK_TILES; ++sub_k) {
                const int panel_k = nb * S_SUBK_TILES + sub_k;
                uint64_t c_desc = stage_c_desc_base + panel_k * WGMMA_Y_B_PANEL_K_DESC_DELTA;
                const int y_scale_d = (panel == 0 && panel_k == 0) ? 0 : 1;

                wgmma_fence();

                #pragma unroll
                for (int jc = 0; jc < OUT_TILES; ++jc) {
                    wgmmaTF32<WGMMA_Y_N>::rs(
                        y[jc],
                        s_as_a_frag[sub_k],
                        c_desc,
                        y_scale_d);
                    c_desc += WGMMA_Y_B_OUT_DESC_DELTA;
                }
            }

            wgmma_commit_group();
            wgmma_wait_group_0();
        }

        if (tid128 == 0) {
            mbarrier_arrive_shared(&stage_done[stage]);
        }
    }

    #pragma unroll
    for (int jc = 0; jc < OUT_TILES; ++jc) {
        #pragma unroll
        for (int e = 0; e < WGMMA_Y_ACC_REGS; e += 2) {
            const int row = frag_row0 + (((e & 2) >> 1) << 3);
            const int col = jc * WGMMA_Y_N + ((e >> 2) << 3) + frag_col0;

            const int m = wg_m0 + row;
            const int d = col;

            if (m < M) {
                reinterpret_cast<float2*>(&Y[m * D_f + d])[0] = make_float2(y[jc][e], y[jc][e + 1]);
            }
        }
    }
}
#endif
