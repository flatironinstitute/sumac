#include <cuda/std/cstdint>

#ifndef WGMMA_TF32_KERNEL_NAME
#define WGMMA_TF32_KERNEL_NAME relu_bat_c_tf32_wgmma
#endif

#ifndef WGMMA_TF32_PACK_KERNEL_NAME
#define WGMMA_TF32_PACK_KERNEL_NAME relu_bat_c_tf32_wgmma_pack
#endif

#ifndef WGMMA_TF32_PACK_ONLY
#define WGMMA_TF32_PACK_ONLY 0
#endif

// Hopper/SM90+ warp-group async MMA variant of Y=ReLU(B A.T)C
//
// A, B, and C are row-major contiguous with runtime leading dimension D.
// A/B K tiles use compile-time padded width D_K_F. C/Y output tiles use
// compile-time padded width D_Y_F.
//
// This kernel does a two-level contraction:
//     S = ReLU(B @ A.T)
//     Y = S @ C
//
// Important for WGMMA:
//  One compute unit is a warpgroup (128 threads), not one warp.
//  WGMMA B operands must come from shared-memory descriptors, so A/C tiles
//  are converted to TF32 and prepacked once in a separate launch.
//  The main kernel uses one producer warpgroup to move prepacked A/C tiles
//  to shared memory with cp.async.bulk copies while the remaining
//  warpgroups consume the previous stage with WGMMA.

using uint32_t = cuda::std::uint32_t;
using uint64_t = cuda::std::uint64_t;

static constexpr int WGMMA_M = 64;
static constexpr int WGMMA_K = 8;
static constexpr int K_TILES = D_K_F / WGMMA_K;
static constexpr int OUT_TILES = D_Y_F / WGMMA_Y_N;
static constexpr int A_PANEL_ELEMS = BN * D_K_F;
static constexpr int C_PANEL_ELEMS = BN * D_Y_F;

static constexpr uint64_t GMMA_DESC_START_MASK = 0x3fffull;
static constexpr int TF32_ELEMS_PER_DESC_UNIT = 4;

__device__ static constexpr uint64_t desc_units_from_elems(int elems) {
  return (static_cast<uint64_t>(elems) * sizeof(uint32_t)) >> 4;
}

__device__ static constexpr uint64_t make_desc_const(int leading_elems,
                                                     int stride_elems) {
  return ((desc_units_from_elems(leading_elems) & GMMA_DESC_START_MASK) << 16) |
         ((desc_units_from_elems(stride_elems) & GMMA_DESC_START_MASK) << 32);
}

// No-swizzle TF32 K-major layout. Descriptor offsets use 16-byte units:
// bits 0-13 encode the start address, bits 16-29 the leading offset, and
// bits 32-45 the stride offset.
template <int N> struct WgmmaTf32SmemLayout {
  static constexpr int kLeadingElems = N * TF32_ELEMS_PER_DESC_UNIT;
  static constexpr int kStrideElems = 8 * TF32_ELEMS_PER_DESC_UNIT;
  static constexpr int kTileElems = 2 * kLeadingElems;
  static constexpr uint64_t kDescConst = make_desc_const(kLeadingElems, kStrideElems);
  static constexpr uint64_t kTileDescDelta = desc_units_from_elems(kTileElems);

  static __device__ __forceinline__ int index(int k, int n) {
    return (k >> 2) * kLeadingElems + (n >> 3) * kStrideElems +
           (n & 7) * TF32_ELEMS_PER_DESC_UNIT + (k & 3);
  }
};

using WgmmaALayout = WgmmaTf32SmemLayout<WGMMA_M>;
using WgmmaSLayout = WgmmaTf32SmemLayout<WGMMA_S_N>;
using WgmmaYLayout = WgmmaTf32SmemLayout<WGMMA_Y_N>;

static_assert(WGMMA_S_N == 16 || WGMMA_S_N == 32 || WGMMA_S_N == 64 || WGMMA_S_N == 128, "WGMMA_S_N must be 16, 32, 64, or 128");
static_assert(WGMMA_Y_N == 16 || WGMMA_Y_N == 32 || WGMMA_Y_N == 64 || WGMMA_Y_N == 128, "WGMMA_Y_N must be 16, 32, 64, or 128");
static_assert(D_K_F >= WGMMA_K, "D_K_F must cover at least one K tile");
static_assert(D_Y_F >= WGMMA_Y_N, "D_Y_F must cover at least one output tile");
static_assert((D_K_F % WGMMA_K) == 0, "D_K_F must be divisible by 8");
static_assert((D_Y_F % WGMMA_Y_N) == 0, "D_Y_F must be divisible by WGMMA_Y_N");
static_assert((BN % WGMMA_S_N) == 0, "BN must be divisible by WGMMA_S_N");

#if !WGMMA_TF32_PACK_ONLY
static constexpr int COMPUTE_WARPGROUPS_PER_BLOCK = BM / WGMMA_M;
static constexpr int THREADS_PER_BLOCK = (COMPUTE_WARPGROUPS_PER_BLOCK + 1) * 128;
static constexpr int N_TILES = BN / WGMMA_S_N;
static constexpr int S_SUBK_TILES = WGMMA_S_N / WGMMA_K;

static constexpr uint64_t A_N_TILE_DESC_DELTA = K_TILES * WgmmaSLayout::kTileDescDelta;
static constexpr uint64_t C_PANEL_K_DESC_DELTA = OUT_TILES * WgmmaYLayout::kTileDescDelta;
static constexpr uint64_t B_SMEM_WG_DESC_DELTA = K_TILES * WgmmaALayout::kTileDescDelta;

static constexpr int PACKED_PANEL_ELEMS = A_PANEL_ELEMS + C_PANEL_ELEMS;
static constexpr int B_SMEM_WG_ELEMS = K_TILES * WgmmaALayout::kTileElems;
static constexpr int A_PANEL_BYTES = A_PANEL_ELEMS * sizeof(uint32_t);
static constexpr int C_PANEL_BYTES = C_PANEL_ELEMS * sizeof(uint32_t);
static constexpr int PACKED_PANEL_BYTES = PACKED_PANEL_ELEMS * sizeof(uint32_t);

static constexpr uint64_t A_PANEL_DESC_DELTA = A_PANEL_BYTES >> 4;
static constexpr uint64_t PACKED_PANEL_DESC_DELTA = PACKED_PANEL_BYTES >> 4;

static_assert(BM >= WGMMA_M && (BM % WGMMA_M) == 0, "BM must be a positive multiple of 64");
static_assert(SMEM_COPY_STAGES >= 1 && SMEM_COPY_STAGES <= 3, "SMEM_COPY_STAGES must be 1, 2, or 3");
static_assert(THREADS_PER_BLOCK <= 1024, "compute and producer warpgroups must fit in one CTA");
#endif

static constexpr int PACK_PANEL_WORK_ELEMS = A_PANEL_ELEMS > C_PANEL_ELEMS ? A_PANEL_ELEMS : C_PANEL_ELEMS;

#if !WGMMA_TF32_PACK_ONLY
__device__ __forceinline__ unsigned char *align_smem_128(unsigned char *p) {
  const unsigned long long addr = reinterpret_cast<unsigned long long>(p);
  const unsigned long long aligned = (addr + 127ull) & ~127ull;
  return reinterpret_cast<unsigned char *>(aligned);
}

__device__ __forceinline__ unsigned shared_addr_u32(const void *ptr) {
  unsigned long long addr64;
  asm volatile("cvta.to.shared.u64 %0, %1;\n" : "=l"(addr64) : "l"(ptr));
  return static_cast<unsigned>(addr64);
}

__device__ __forceinline__ uint64_t shared_desc_start(const void *ptr) {
  return (static_cast<uint64_t>(shared_addr_u32(ptr)) >> 4) & GMMA_DESC_START_MASK;
}

__device__ __forceinline__ unsigned long long global_addr_u64(const void *ptr) {
  unsigned long long addr64;
  asm volatile("cvta.to.global.u64 %0, %1;\n" : "=l"(addr64) : "l"(ptr));
  return addr64;
}

__device__ __forceinline__ void fence_proxy_async_shared_cta() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

__device__ __forceinline__ void
mbarrier_init_shared(unsigned long long *barrier, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
               :
               : "r"(shared_addr_u32(barrier)), "r"(count)
               : "memory");
}

__device__ __forceinline__ unsigned long long
mbarrier_arrive_expect_tx_shared(unsigned long long *barrier, int bytes) {
  unsigned long long state;
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
               "%0, [%1], %2;\n"
               : "=l"(state)
               : "r"(shared_addr_u32(barrier)), "r"(bytes)
               : "memory");
  return state;
}

__device__ __forceinline__ void
mbarrier_wait_shared(unsigned long long *barrier, unsigned long long state) {
  const unsigned barrier_addr = shared_addr_u32(barrier);
  unsigned done;
  do {
    asm volatile("{ .reg .pred p;\n"
                 "  mbarrier.try_wait.acquire.cta.shared::cta.b64 "
                 "p, [%1], %2;\n"
                 "  selp.u32 %0, 1, 0, p;\n"
                 "}\n"
                 : "=r"(done)
                 : "r"(barrier_addr), "l"(state)
                 : "memory");
  } while (!done);
}

__device__ __forceinline__ void
mbarrier_arrive_shared(unsigned long long *barrier) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n"
               :
               : "r"(shared_addr_u32(barrier))
               : "memory");
}

__device__ __forceinline__ void
cp_async_bulk_shared_global(void *dst_shared, const void *src_global, int bytes,
                            unsigned long long *complete_barrier) {
  asm volatile("cp.async.bulk.shared::cluster.global."
               "mbarrier::complete_tx::bytes "
               "[%0], [%1], %2, [%3];\n"
               :
               : "r"(shared_addr_u32(dst_shared)),
                 "l"(global_addr_u64(src_global)), "r"(bytes),
                 "r"(shared_addr_u32(complete_barrier))
               : "memory");
}

__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void wgmma_wait_group_0() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

__device__ __forceinline__ bool elect_sync(unsigned membermask) {
  unsigned is_leader;
  asm volatile("{ .reg .pred p;\n"
               "  elect.sync _|p, %1;\n"
               "  selp.u32 %0, 1, 0, p;\n"
               "}\n"
               : "=r"(is_leader)
               : "r"(membermask));
  return is_leader != 0;
}

__device__ __forceinline__ bool elect_one_from_warpgroup_warp0(int local_tid) {
  return local_tid < 32 && elect_sync(0xffffffffu);
}
#endif

__device__ __forceinline__ int perm_logical_to_phys_k8(int k_logical) {
  return ((k_logical & 3) << 1) | (k_logical >> 2);
}

__device__ __forceinline__ int perm_logical_to_phys_n(int n_logical) {
  return (n_logical & ~7) | perm_logical_to_phys_k8(n_logical & 7);
}

__device__ __forceinline__ uint32_t tf32_from_float(float v) {
  uint32_t out;
  asm volatile("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(out) : "f"(v));
  return out;
}

#if !WGMMA_TF32_PACK_ONLY
__device__ __forceinline__ uint32_t tf32_relu_from_float(float v) {
  uint32_t out;
  asm volatile("cvt.rn.relu.tf32.f32 %0, %1;\n" : "=r"(out) : "f"(v));
  return out;
}

__device__ __forceinline__ uint4 tf32_from_float4(float4 v) {
  uint4 out;
  out.x = tf32_from_float(v.x);
  out.y = tf32_from_float(v.y);
  out.z = tf32_from_float(v.z);
  out.w = tf32_from_float(v.w);
  return out;
}

__device__ __forceinline__ void store_shared_u32x4(uint32_t *ptr, uint4 v) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n"
               :
               : "r"(shared_addr_u32(ptr)), "r"(v.x), "r"(v.y), "r"(v.z),
                 "r"(v.w)
               : "memory");
}
#endif

template <int PaddedD>
__device__ __forceinline__ void pack_coords_from_linear(int idx, int &n_local,
                                                        int &d) {
  constexpr int load_cols = PaddedD >= 32 && (PaddedD % 16) == 0 ? 16 : WGMMA_K;
  constexpr int load_rows = 32 / load_cols;
  constexpr int col_tiles = PaddedD / load_cols;

  static_assert((PaddedD % load_cols) == 0, "packed width must be divisible by the vector width");

  const int elem = idx & 31;
  const int tile = idx >> 5;
  const int row = elem / load_cols;
  const int col = elem - row * load_cols;
  const int col_tile = tile % col_tiles;
  const int row_tile = tile / col_tiles;

  n_local = row_tile * load_rows + row;
  d = col_tile * load_cols + col;
}

__device__ __forceinline__ void pack_panel_element_tf32(
    const float *__restrict__ A, const float *__restrict__ C,
    uint32_t *__restrict__ A_packed, uint32_t *__restrict__ C_packed, int n0,
    int N, int D, int raw_idx) {
  if (raw_idx < A_PANEL_ELEMS) {
    int n_local;
    int d;
    pack_coords_from_linear<D_K_F>(raw_idx, n_local, d);

    float av = 0.0f;
    const int n = n0 + n_local;
    if (n < N && d < D) {
      av = A[static_cast<long long>(n) * D + d];
    }

    const int nb = n_local / WGMMA_S_N;
    const int n_in_tile = n_local - nb * WGMMA_S_N;
    const int d_tile = d / WGMMA_K;
    const int d_in_tile = d - d_tile * WGMMA_K;
    const int a_elem = WgmmaSLayout::index(d_in_tile, perm_logical_to_phys_n(n_in_tile));

    A_packed[(nb * K_TILES + d_tile) * WgmmaSLayout::kTileElems + a_elem] = tf32_from_float(av);
  }

  if (raw_idx < C_PANEL_ELEMS) {
    int n_local;
    int d;
    pack_coords_from_linear<D_Y_F>(raw_idx, n_local, d);

    float cv = 0.0f;
    const int n = n0 + n_local;
    if (n < N && d < D) {
      cv = C[static_cast<long long>(n) * D + d];
    }

    const int panel_k = n_local / WGMMA_K;
    const int k_in_panel = n_local - panel_k * WGMMA_K;
    const int out_tile = d / WGMMA_Y_N;
    const int d_in_out_tile = d - out_tile * WGMMA_Y_N;
    const int c_elem = WgmmaYLayout::index(k_in_panel, d_in_out_tile);

    C_packed[(panel_k * OUT_TILES + out_tile) * WgmmaYLayout::kTileElems + c_elem] = tf32_from_float(cv);
  }
}

#if !WGMMA_TF32_PACK_ONLY
#define WGMMA_D_LIST_16 "{%0, %1, %2, %3, %4, %5, %6, %7}"
#define WGMMA_D_OPERANDS_16(d)                                                 \
  "+f"((d)[0]), "+f"((d)[1]), "+f"((d)[2]), "+f"((d)[3]), "+f"((d)[4]),        \
      "+f"((d)[5]), "+f"((d)[6]), "+f"((d)[7])

#define WGMMA_D_LIST_32                                                        \
  "{%0, %1, %2, %3, %4, %5, %6, %7, "                                          \
  "%8, %9, %10, %11, %12, %13, %14, %15}"
#define WGMMA_D_OPERANDS_32(d)                                                 \
  "+f"((d)[0]), "+f"((d)[1]), "+f"((d)[2]), "+f"((d)[3]), "+f"((d)[4]),        \
      "+f"((d)[5]), "+f"((d)[6]), "+f"((d)[7]), "+f"((d)[8]), "+f"((d)[9]),    \
      "+f"((d)[10]), "+f"((d)[11]), "+f"((d)[12]), "+f"((d)[13]),              \
      "+f"((d)[14]), "+f"((d)[15])

#define WGMMA_D_LIST_64                                                        \
  "{%0, %1, %2, %3, %4, %5, %6, %7, "                                          \
  "%8, %9, %10, %11, %12, %13, %14, %15, "                                     \
  "%16, %17, %18, %19, %20, %21, %22, %23, "                                   \
  "%24, %25, %26, %27, %28, %29, %30, %31}"
#define WGMMA_D_OPERANDS_64(d)                                                 \
  "+f"((d)[0]), "+f"((d)[1]), "+f"((d)[2]), "+f"((d)[3]), "+f"((d)[4]),        \
      "+f"((d)[5]), "+f"((d)[6]), "+f"((d)[7]), "+f"((d)[8]), "+f"((d)[9]),    \
      "+f"((d)[10]), "+f"((d)[11]), "+f"((d)[12]), "+f"((d)[13]),              \
      "+f"((d)[14]), "+f"((d)[15]), "+f"((d)[16]), "+f"((d)[17]),              \
      "+f"((d)[18]), "+f"((d)[19]), "+f"((d)[20]), "+f"((d)[21]),              \
      "+f"((d)[22]), "+f"((d)[23]), "+f"((d)[24]), "+f"((d)[25]),              \
      "+f"((d)[26]), "+f"((d)[27]), "+f"((d)[28]), "+f"((d)[29]),              \
      "+f"((d)[30]), "+f"((d)[31])

#define WGMMA_D_LIST_128                                                       \
  "{%0, %1, %2, %3, %4, %5, %6, %7, "                                          \
  "%8, %9, %10, %11, %12, %13, %14, %15, "                                     \
  "%16, %17, %18, %19, %20, %21, %22, %23, "                                   \
  "%24, %25, %26, %27, %28, %29, %30, %31, "                                   \
  "%32, %33, %34, %35, %36, %37, %38, %39, "                                   \
  "%40, %41, %42, %43, %44, %45, %46, %47, "                                   \
  "%48, %49, %50, %51, %52, %53, %54, %55, "                                   \
  "%56, %57, %58, %59, %60, %61, %62, %63}"
#define WGMMA_D_OPERANDS_128(d)                                                \
  "+f"((d)[0]), "+f"((d)[1]), "+f"((d)[2]), "+f"((d)[3]), "+f"((d)[4]),        \
      "+f"((d)[5]), "+f"((d)[6]), "+f"((d)[7]), "+f"((d)[8]), "+f"((d)[9]),    \
      "+f"((d)[10]), "+f"((d)[11]), "+f"((d)[12]), "+f"((d)[13]),              \
      "+f"((d)[14]), "+f"((d)[15]), "+f"((d)[16]), "+f"((d)[17]),              \
      "+f"((d)[18]), "+f"((d)[19]), "+f"((d)[20]), "+f"((d)[21]),              \
      "+f"((d)[22]), "+f"((d)[23]), "+f"((d)[24]), "+f"((d)[25]),              \
      "+f"((d)[26]), "+f"((d)[27]), "+f"((d)[28]), "+f"((d)[29]),              \
      "+f"((d)[30]), "+f"((d)[31]), "+f"((d)[32]), "+f"((d)[33]),              \
      "+f"((d)[34]), "+f"((d)[35]), "+f"((d)[36]), "+f"((d)[37]),              \
      "+f"((d)[38]), "+f"((d)[39]), "+f"((d)[40]), "+f"((d)[41]),              \
      "+f"((d)[42]), "+f"((d)[43]), "+f"((d)[44]), "+f"((d)[45]),              \
      "+f"((d)[46]), "+f"((d)[47]), "+f"((d)[48]), "+f"((d)[49]),              \
      "+f"((d)[50]), "+f"((d)[51]), "+f"((d)[52]), "+f"((d)[53]),              \
      "+f"((d)[54]), "+f"((d)[55]), "+f"((d)[56]), "+f"((d)[57]),              \
      "+f"((d)[58]), "+f"((d)[59]), "+f"((d)[60]), "+f"((d)[61]),              \
      "+f"((d)[62]), "+f"((d)[63])

enum class WgmmaScaleOut : int {
  Overwrite = 0,
  Accumulate = 1,
};

template <int N> struct WgmmaTf32;

#define DEFINE_WGMMA_TF32(N, D_LIST, D_OPERANDS, RS_A_LIST, RS_DESC_B,         \
                          RS_SCALE_D, SS_DESC_A, SS_DESC_B, SS_SCALE_D)        \
  template <> struct WgmmaTf32<N> {                                            \
    static constexpr int kAccumulatorRegisters = N / 2;                        \
    using Accumulator = float[kAccumulatorRegisters];                          \
    static __device__ __forceinline__ void rs(Accumulator &d,                  \
                                              const uint32_t (&a)[4],          \
                                              uint64_t desc_b,                 \
                                              WgmmaScaleOut scale_out) {       \
      const int scale_d = static_cast<int>(scale_out);                         \
      asm volatile("{\n"                                                       \
                   "  .reg .pred p;\n"                                         \
                   "  setp.ne.b32 p, " RS_SCALE_D ", 0;\n"                     \
                   "  wgmma.mma_async.sync.aligned.m64n" #N                    \
                   "k8.f32.tf32.tf32 " D_LIST ", " RS_A_LIST ", " RS_DESC_B    \
                   ", p, 1, 1;\n"                                              \
                   "}\n"                                                       \
                   : D_OPERANDS(d)                                             \
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc_b),  \
                     "r"(scale_d)                                              \
                   : "memory");                                                \
    }                                                                          \
    static __device__ __forceinline__ void ss(Accumulator &d, uint64_t desc_a, \
                                              uint64_t desc_b,                 \
                                              WgmmaScaleOut scale_out) {       \
      const int scale_d = static_cast<int>(scale_out);                         \
      asm volatile("{\n"                                                       \
                   "  .reg .pred p;\n"                                         \
                   "  setp.ne.b32 p, " SS_SCALE_D ", 0;\n"                     \
                   "  wgmma.mma_async.sync.aligned.m64n" #N                    \
                   "k8.f32.tf32.tf32 " D_LIST ", " SS_DESC_A ", " SS_DESC_B    \
                   ", p, 1, 1;\n"                                              \
                   "}\n"                                                       \
                   : D_OPERANDS(d)                                             \
                   : "l"(desc_a), "l"(desc_b), "r"(scale_d)                    \
                   : "memory");                                                \
    }                                                                          \
  };

DEFINE_WGMMA_TF32(16, WGMMA_D_LIST_16, WGMMA_D_OPERANDS_16,
                  "{%8, %9, %10, %11}", "%12", "%13", "%8", "%9", "%10")
DEFINE_WGMMA_TF32(32, WGMMA_D_LIST_32, WGMMA_D_OPERANDS_32,
                  "{%16, %17, %18, %19}", "%20", "%21", "%16", "%17", "%18")
DEFINE_WGMMA_TF32(64, WGMMA_D_LIST_64, WGMMA_D_OPERANDS_64,
                  "{%32, %33, %34, %35}", "%36", "%37", "%32", "%33", "%34")
DEFINE_WGMMA_TF32(128, WGMMA_D_LIST_128, WGMMA_D_OPERANDS_128,
                  "{%64, %65, %66, %67}", "%68", "%69", "%64", "%65", "%66")

#undef DEFINE_WGMMA_TF32
#undef WGMMA_D_LIST_16
#undef WGMMA_D_LIST_32
#undef WGMMA_D_LIST_64
#undef WGMMA_D_LIST_128
#undef WGMMA_D_OPERANDS_16
#undef WGMMA_D_OPERANDS_32
#undef WGMMA_D_OPERANDS_64
#undef WGMMA_D_OPERANDS_128
#endif

// KERNEL_START

extern "C" __global__ void
WGMMA_TF32_PACK_KERNEL_NAME(const float *__restrict__ A,
                            const float *__restrict__ C,
                            uint32_t *__restrict__ A_packed,
                            uint32_t *__restrict__ C_packed, int N, int D) {
  const int raw_global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int raw_stride = blockDim.x * gridDim.x;
  const int num_tiles = (N + BN - 1) / BN;
  const int panel = blockIdx.y;

  if (panel >= num_tiles) {
    return;
  }

  const int n0 = panel * BN;
  for (int raw_idx = raw_global_idx; raw_idx < PACK_PANEL_WORK_ELEMS; raw_idx += raw_stride) {
    pack_panel_element_tf32(A, C, A_packed + panel * A_PANEL_ELEMS,
                            C_packed + panel * C_PANEL_ELEMS, n0, N, D,
                            raw_idx);
  }
}

#if !WGMMA_TF32_PACK_ONLY
extern "C" __global__
__launch_bounds__(THREADS_PER_BLOCK, 1) void WGMMA_TF32_KERNEL_NAME(
    const uint32_t *__restrict__ A_packed_global,
    const uint32_t *__restrict__ C_packed_global, const float *__restrict__ B,
    float *__restrict__ Y, int N, int M, int D) {
  const int tid = threadIdx.x;
  const int warpgroup_id = tid >> 7;
  const int warpgroup_tid = tid & 127;
  const bool is_producer_warpgroup = warpgroup_id == COMPUTE_WARPGROUPS_PER_BLOCK;

  extern __shared__ unsigned char dynamic_smem[];
  unsigned char *smem = align_smem_128(dynamic_smem);
  uint32_t *smem_tiles = reinterpret_cast<uint32_t *>(smem);
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

  const int num_tiles = (N + BN - 1) / BN;
  if (num_tiles == 0) {
    return;
  }

  if (is_producer_warpgroup) {
    if (elect_one_from_warpgroup_warp0(warpgroup_tid)) {
      #pragma unroll 1
      for (int panel = 0; panel < num_tiles; ++panel) {

        const int stage = panel % SMEM_COPY_STAGES;
        uint32_t *stage_a = smem_tiles + stage * PACKED_PANEL_ELEMS;
        uint32_t *stage_c = stage_a + A_PANEL_ELEMS;

        if (panel >= SMEM_COPY_STAGES) {
          mbarrier_wait_shared(&stage_done[stage], stage_done_state[stage]);
        }
        stage_done_state[stage] = mbarrier_arrive_expect_tx_shared(&stage_done[stage], 0);
        stage_ready_state[stage] = mbarrier_arrive_expect_tx_shared(&stage_ready[stage], PACKED_PANEL_BYTES);
        __threadfence_block();

        stage_ready_panel[stage] = panel;
        cp_async_bulk_shared_global(stage_a, A_packed_global + panel * A_PANEL_ELEMS,
                                    A_PANEL_BYTES, &stage_ready[stage]);

        cp_async_bulk_shared_global(stage_c, C_packed_global + panel * C_PANEL_ELEMS,
                                    C_PANEL_BYTES, &stage_ready[stage]);
      }
    }
    return;
  }

  const int lane = tid & 31;
  const int warp_in_wg = warpgroup_tid >> 5;
  const int tid4 = lane & 3;
  const int row8 = lane >> 2;
  const int wg_m0 = blockIdx.x * BM + warpgroup_id * WGMMA_M;
  const int frag_row0 = warp_in_wg * 16 + row8;
  const int frag_col0 = tid4 << 1;
  const uint64_t smem_tiles_start = shared_desc_start(smem_tiles);

#if WGMMA_FIRST_MMA_SS
  uint32_t *b_smem = smem_tiles + SMEM_COPY_STAGES * PACKED_PANEL_ELEMS;

#pragma unroll 1
  for (int kp = 0; kp < K_TILES; ++kp) {
    const int half_warp = lane >> 4;
    const int lane16 = lane & 15;
    const int row_in_8 = lane16 & 7;
    const int k_group = lane16 >> 3;
    const int row = warp_in_wg * 16 + half_warp * 8 + row_in_8;
    const int d = kp * WGMMA_K + k_group * TF32_ELEMS_PER_DESC_UNIT;
    const int m = wg_m0 + row;

    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (m < M) {
      const float *row_ptr = B + static_cast<long long>(m) * D;
      if ((D % 4) == 0 && d + 3 < D) {
        v = reinterpret_cast<const float4 *>(&row_ptr[d])[0];
      } else {
        v.x = d + 0 < D ? row_ptr[d + 0] : 0.0f;
        v.y = d + 1 < D ? row_ptr[d + 1] : 0.0f;
        v.z = d + 2 < D ? row_ptr[d + 2] : 0.0f;
        v.w = d + 3 < D ? row_ptr[d + 3] : 0.0f;
      }
    }

    const uint4 v_tf32 = tf32_from_float4(v);
    const int elem = WgmmaALayout::index(k_group * TF32_ELEMS_PER_DESC_UNIT, row);
    store_shared_u32x4(&b_smem[warpgroup_id * B_SMEM_WG_ELEMS + kp * WgmmaALayout::kTileElems + elem], v_tf32);
  }

  uint64_t b_desc[K_TILES];
  const uint64_t b_smem_start = (smem_tiles_start + SMEM_COPY_STAGES * PACKED_PANEL_DESC_DELTA + warpgroup_id * B_SMEM_WG_DESC_DELTA) & GMMA_DESC_START_MASK;
  #pragma unroll
  for (int kp = 0; kp < K_TILES; ++kp) {
    const uint64_t b_start = (b_smem_start + kp * WgmmaALayout::kTileDescDelta) & GMMA_DESC_START_MASK;
    b_desc[kp] = b_start | WgmmaALayout::kDescConst;
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
      if (m < M && d < D) {
        v = B[static_cast<long long>(m) * D + d];
      }

      b_regs[kp][q] = tf32_from_float(v);
    }
  }
#endif

  using SWgmma = WgmmaTf32<WGMMA_S_N>;
  using YWgmma = WgmmaTf32<WGMMA_Y_N>;
  YWgmma::Accumulator y[OUT_TILES] = {};

  for (int panel = 0; panel < num_tiles; ++panel) {
    const int stage = panel % SMEM_COPY_STAGES;

    while (((volatile int *)stage_ready_panel)[stage] != panel) {}

    const unsigned long long ready_state = ((volatile unsigned long long *)stage_ready_state)[stage];
    mbarrier_wait_shared(&stage_ready[stage], ready_state);
    fence_proxy_async_shared_cta();

    const uint64_t stage_a_start = (smem_tiles_start + stage * PACKED_PANEL_DESC_DELTA) & GMMA_DESC_START_MASK;
    const uint64_t stage_a_desc_base = stage_a_start | WgmmaSLayout::kDescConst;
    const uint64_t stage_c_desc_base = (stage_a_start + A_PANEL_DESC_DELTA) | WgmmaYLayout::kDescConst;

    SWgmma::Accumulator s_pipe = {};

    #pragma unroll
    for (int nb = 0; nb < N_TILES; ++nb) {
      uint64_t a_desc = stage_a_desc_base + nb * A_N_TILE_DESC_DELTA;

      wgmma_fence();
      #if WGMMA_FIRST_MMA_SS
      SWgmma::ss(s_pipe, b_desc[0], a_desc, WgmmaScaleOut::Overwrite);
      #else
      SWgmma::rs(s_pipe, b_regs[0], a_desc, WgmmaScaleOut::Overwrite);
      #endif

      #pragma unroll
      for (int kp = 1; kp < K_TILES; ++kp) {
        a_desc += WgmmaSLayout::kTileDescDelta;

        #if WGMMA_FIRST_MMA_SS
        SWgmma::ss(s_pipe, b_desc[kp], a_desc, WgmmaScaleOut::Accumulate);
        #else
        SWgmma::rs(s_pipe, b_regs[kp], a_desc, WgmmaScaleOut::Accumulate);
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
        uint64_t c_desc = stage_c_desc_base + panel_k * C_PANEL_K_DESC_DELTA;
        const WgmmaScaleOut y_scale = panel == 0 && panel_k == 0 ? WgmmaScaleOut::Overwrite : WgmmaScaleOut::Accumulate;

        wgmma_fence();

        #pragma unroll
        for (int jc = 0; jc < OUT_TILES; ++jc) {
          YWgmma::rs(y[jc], s_as_a_frag[sub_k], c_desc, y_scale);
          c_desc += WgmmaYLayout::kTileDescDelta;
        }
      }

      wgmma_commit_group();
      wgmma_wait_group_0();
    }

    if (warpgroup_tid == 0) {
      mbarrier_arrive_shared(&stage_done[stage]);
    }
  }

  #pragma unroll
  for (int jc = 0; jc < OUT_TILES; ++jc) {
    #pragma unroll
    for (int e = 0; e < YWgmma::kAccumulatorRegisters; e += 2) {
      const int row = frag_row0 + (((e & 2) >> 1) << 3);
      const int col = jc * WGMMA_Y_N + ((e >> 2) << 3) + frag_col0;

      const int m = wg_m0 + row;

      if (m < M) {
        reinterpret_cast<float2 *>(&Y[static_cast<long long>(m) * D_Y_F + col])[0] = make_float2(y[jc][e], y[jc][e + 1]);
      }
    }
  }
}
#endif
