#pragma once

#include "mex.h"
#include "gpu/mxGPUArray.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef TF32_MEX_ID
#define TF32_MEX_ID "relu_bat_c_tf32_nvrtc_mex"
#endif

namespace relu_bat_c_tf32_nvrtc {

static constexpr int kPackThreads = 256;
static constexpr int kDefaultDynamicSmemLimitBytes = 48 * 1024;
static constexpr const char* kMmaSyncKernelPath =
    "../../src/sumac/kernels/relu_batc_tf32_jit/kernel_mma_sync_tf32.cu";
static constexpr const char* kWgmmaKernelPath =
    "../../src/sumac/kernels/relu_batc_tf32_jit/kernel_wgmma_tf32_tma.cu";

enum class Tf32Mode {
    MmaSync,
    Wgmma,
};

struct DeviceInfo {
    int ordinal = 0;
    int major = 0;
    int minor = 0;
    int max_threads_per_block = 0;
    int max_dynamic_smem_bytes = 0;
};

struct MmaSyncConfig {
    int BM = 0;
    int BN = 0;
    int D_f = 0;
    int M_TILES = 0;
    int num_stages = 0;
};

struct WgmmaConfig {
    int BM = 0;
    int BN = 0;
    int D_k_f = 0;
    int D_y_f = 0;
    int WGMMA_S_N = 0;
    int WGMMA_Y_N = 0;
    int num_stages = 0;
    bool first_mma_ss = false;
};

struct LaunchPlan {
    Tf32Mode mode = Tf32Mode::MmaSync;
    DeviceInfo device;
    MmaSyncConfig mma;
    WgmmaConfig wgmma;
    int output_ld = 0;
    int threads_per_block = 0;
    int dynamic_smem_bytes = 0;
};

struct ModuleEntry {
    CUmodule module = nullptr;
    CUfunction kernel_func = nullptr;
    CUfunction pack_func = nullptr;
};

static void check_cuda_rt(cudaError_t err, const char* where);
static void register_persistent_buffers();

struct DeviceBuffer {
    void* ptr = nullptr;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }

    void allocate(size_t bytes, const char* where) {
        if (bytes == 0) {
            return;
        }
        check_cuda_rt(cudaMalloc(&ptr, bytes), where);
    }

    template <typename T>
    T* as() {
        return static_cast<T*>(ptr);
    }
};

struct PersistentDeviceBuffer {
    void* ptr = nullptr;
    size_t capacity_bytes = 0;
    std::vector<void*> retired_ptrs;

    PersistentDeviceBuffer() = default;
    PersistentDeviceBuffer(const PersistentDeviceBuffer&) = delete;
    PersistentDeviceBuffer& operator=(const PersistentDeviceBuffer&) = delete;

    void ensure(size_t bytes, const char* where) {
        if (bytes <= capacity_bytes) {
            return;
        }

        register_persistent_buffers();
        void* next_ptr = nullptr;
        check_cuda_rt(cudaMalloc(&next_ptr, bytes), where);
        if (ptr != nullptr) {
            retired_ptrs.push_back(ptr);
        }
        ptr = next_ptr;
        capacity_bytes = bytes;
    }

    template <typename T>
    T* as() {
        return static_cast<T*>(ptr);
    }

    void release() {
        if (ptr != nullptr) {
            cudaFree(ptr);
            ptr = nullptr;
            capacity_bytes = 0;
        }
        for (void* retired : retired_ptrs) {
            if (retired != nullptr) {
                cudaFree(retired);
            }
        }
        retired_ptrs.clear();
    }
};

static bool g_cuda_initialized = false;
static std::unordered_map<std::string, ModuleEntry> g_module_cache;
static bool g_persistent_buffers_registered = false;
static PersistentDeviceBuffer g_a_packed_cache;
static PersistentDeviceBuffer g_c_packed_cache;
static PersistentDeviceBuffer g_edge_score_cache;

static void release_persistent_buffers() {
    g_a_packed_cache.release();
    g_c_packed_cache.release();
    g_edge_score_cache.release();
}

static void register_persistent_buffers() {
    if (!g_persistent_buffers_registered) {
        mexAtExit(release_persistent_buffers);
        g_persistent_buffers_registered = true;
    }
}

static std::string error_id(const char* suffix) {
    return std::string(TF32_MEX_ID) + ":" + suffix;
}

[[noreturn]] static void fail(const char* suffix, const std::string& msg) {
    const std::string id = error_id(suffix);
    mexErrMsgIdAndTxt(id.c_str(), "%s", msg.c_str());
}

static void check_cuda_drv(CUresult err, const char* where) {
    if (err == CUDA_SUCCESS) {
        return;
    }

    const char* name = nullptr;
    const char* str = nullptr;
    cuGetErrorName(err, &name);
    cuGetErrorString(err, &str);

    std::ostringstream oss;
    oss << where << " failed: "
        << (name ? name : "CUDA_ERROR")
        << " - "
        << (str ? str : "unknown");
    fail("CUDA", oss.str());
}

static void check_nvrtc(nvrtcResult err, const char* where) {
    if (err == NVRTC_SUCCESS) {
        return;
    }

    std::ostringstream oss;
    oss << where << " failed: " << nvrtcGetErrorString(err);
    fail("NVRTC", oss.str());
}

static void check_cuda_rt(cudaError_t err, const char* where) {
    if (err == cudaSuccess) {
        return;
    }

    std::ostringstream oss;
    oss << where << " failed: " << cudaGetErrorString(err);
    fail("CUDA", oss.str());
}

static int round_up(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

static int checked_int_dim(mwSize value, const char* name) {
    if (value > static_cast<mwSize>(std::numeric_limits<int>::max())) {
        std::ostringstream oss;
        oss << name << " exceeds int range.";
        fail("shape", oss.str());
    }
    return static_cast<int>(value);
}

static mwSize scalar_offset(const mxArray* arr, const char* name) {
    if (!mxIsNumeric(arr) || mxIsComplex(arr) || mxGetNumberOfElements(arr) != 1) {
        std::ostringstream oss;
        oss << name << " must be a real numeric scalar.";
        fail("offset", oss.str());
    }

    const double value = mxGetScalar(arr);
    if (value < 0.0 ||
        value > static_cast<double>(std::numeric_limits<mwSize>::max())) {
        std::ostringstream oss;
        oss << name << " is out of range.";
        fail("offset", oss.str());
    }

    return static_cast<mwSize>(value);
}

static std::string read_text_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        std::ostringstream oss;
        oss << "Could not open CUDA source file: " << path;
        fail("FileOpen", oss.str());
    }

    std::ostringstream buffer;
    buffer << ifs.rdbuf();

    if (!ifs.good() && !ifs.eof()) {
        std::ostringstream oss;
        oss << "Error reading CUDA source file: " << path;
        fail("FileRead", oss.str());
    }

    return buffer.str();
}

static std::string cuda_include_option() {
    const char* cuda_path = std::getenv("CUDA_HOME");
    if (!cuda_path) {
        cuda_path = std::getenv("CUDA_PATH");
    }

    if (!cuda_path) {
        fail("CUDAPath",
             "Set CUDA_HOME or CUDA_PATH so NVRTC can find cuda/std headers.");
    }

    return std::string("--include-path=") + cuda_path + "/include";
}

static DeviceInfo current_device_info() {
    if (!g_cuda_initialized) {
        check_cuda_drv(cuInit(0), "cuInit");
        g_cuda_initialized = true;
    }

    CUcontext ctx = nullptr;
    check_cuda_drv(cuCtxGetCurrent(&ctx), "cuCtxGetCurrent");
    if (ctx == nullptr) {
        fail("Context", "No active CUDA context. Create a gpuArray or call gpuDevice first.");
    }

    DeviceInfo info;
    check_cuda_drv(cuCtxGetDevice(&info.ordinal), "cuCtxGetDevice");
    check_cuda_drv(
        cuDeviceGetAttribute(
            &info.major,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            info.ordinal),
        "cuDeviceGetAttribute(major)");
    check_cuda_drv(
        cuDeviceGetAttribute(
            &info.minor,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            info.ordinal),
        "cuDeviceGetAttribute(minor)");
    check_cuda_drv(
        cuDeviceGetAttribute(
            &info.max_threads_per_block,
            CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
            info.ordinal),
        "cuDeviceGetAttribute(max_threads_per_block)");

    int default_smem = 0;
    int optin_smem = 0;
    check_cuda_drv(
        cuDeviceGetAttribute(
            &default_smem,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            info.ordinal),
        "cuDeviceGetAttribute(max_shared_memory_per_block)");
    check_cuda_drv(
        cuDeviceGetAttribute(
            &optin_smem,
            CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            info.ordinal),
        "cuDeviceGetAttribute(max_shared_memory_per_block_optin)");

    info.max_dynamic_smem_bytes = std::max(default_smem, optin_smem);
    if (info.major == 9) {
        info.max_dynamic_smem_bytes =
            std::max(info.max_dynamic_smem_bytes, 227 * 1024);
    }

    return info;
}
//These parameters come from autotuning for the connectome dataset benchmark with the sumac pytorch version on RTX Pro 6000 Blackwell.
//Since we don't do autotuning for the matlab version, these might not be ideal in all cases.
static MmaSyncConfig default_mma_sync_config(int D) {
    MmaSyncConfig cfg;
    if (D <= 16) {
        cfg.BM = 256;
        cfg.BN = 64;
        cfg.M_TILES = 4;
        cfg.num_stages = 3;
    } else if (D <= 32) {
        cfg.BM = 256;
        cfg.BN = 16;
        cfg.M_TILES = 4;
        cfg.num_stages = 2;
    } else if (D <= 64) {
        cfg.BM = 128;
        cfg.BN = 32;
        cfg.M_TILES = 2;
        cfg.num_stages = 2;
    } else if (D <= 128) {
        cfg.BM = 128;
        cfg.BN = 32;
        cfg.M_TILES = 1;
        cfg.num_stages = 2;
    } else {
        cfg.BM = 128;
        cfg.BN = 16;
        cfg.M_TILES = 1;
        cfg.num_stages = 1;
    }
    cfg.D_f = round_up(D, 8);
    return cfg;
}
//These parameters come from autotuning for the connectome dataset benchmark with the sumac pytorch version on H100.
//Since we don't do autotuning for the matlab version, these might not be ideal in all cases.
static WgmmaConfig default_wgmma_config(int D) {
    WgmmaConfig cfg;
    if (D <= 16) {
        cfg.BM = 320;
        cfg.BN = 128;
        cfg.WGMMA_S_N = 64;
        cfg.WGMMA_Y_N = 16;
        cfg.num_stages = 2;
        cfg.first_mma_ss = false;
    } else if (D <= 32) {
        cfg.BM = 256;
        cfg.BN = 128;
        cfg.WGMMA_S_N = 64;
        cfg.WGMMA_Y_N = 32;
        cfg.num_stages = 2;
        cfg.first_mma_ss = false;
    } else if (D <= 64) {
        cfg.BM = 192;
        cfg.BN = 128;
        cfg.WGMMA_S_N = 64;
        cfg.WGMMA_Y_N = 64;
        cfg.num_stages = 2;
        cfg.first_mma_ss = false;
    } else if (D <= 128) {
        cfg.BM = 192;
        cfg.BN = 64;
        cfg.WGMMA_S_N = 64;
        cfg.WGMMA_Y_N = 128;
        cfg.num_stages = 2;
        cfg.first_mma_ss = true;
    } else {
        cfg.BM = 128;
        cfg.BN = 16;
        cfg.WGMMA_S_N = 16;
        cfg.WGMMA_Y_N = 64;
        cfg.num_stages = 2;
        cfg.first_mma_ss = true;
    }
    cfg.D_k_f = round_up(D, 8);
    cfg.D_y_f = round_up(D, cfg.WGMMA_Y_N);
    return cfg;
}

static int mma_sync_threads_per_block(const MmaSyncConfig& cfg) {
    const int warp_m_rows = cfg.M_TILES * 16;
    return (cfg.BM / warp_m_rows) * 32;
}

static int wgmma_threads_per_block(const WgmmaConfig& cfg) {
    return (cfg.BM / 64 + 1) * 128;
}

static int mma_sync_dynamic_smem_bytes(const MmaSyncConfig& cfg) {
    return 2 * cfg.num_stages * cfg.BN * cfg.D_f * 4 + 127;
}

static int wgmma_dynamic_smem_bytes(const WgmmaConfig& cfg) {
    int elems = cfg.num_stages * cfg.BN * (cfg.D_k_f + cfg.D_y_f);
    if (cfg.first_mma_ss) {
        elems += cfg.BM * cfg.D_k_f;
    }
    return elems * 4 + 127;
}

static bool valid_mma_sync_config(
    const DeviceInfo& device,
    const MmaSyncConfig& cfg,
    int D) {
    if (device.major < 8 || D < 1) {
        return false;
    }
    if ((cfg.BN % 8) != 0 ||
        cfg.num_stages < 1 ||
        cfg.num_stages > 3) {
        return false;
    }
    const int warp_m_rows = cfg.M_TILES * 16;
    if (warp_m_rows <= 0 || (cfg.BM % warp_m_rows) != 0) {
        return false;
    }
    const int warps = cfg.BM / warp_m_rows;
    if (warps < 1 || warps > 8) {
        return false;
    }
    if (mma_sync_threads_per_block(cfg) > device.max_threads_per_block) {
        return false;
    }
    return mma_sync_dynamic_smem_bytes(cfg) <= device.max_dynamic_smem_bytes;
}

static bool valid_wgmma_config(
    const DeviceInfo& device,
    const WgmmaConfig& cfg,
    int D) {
    if (device.major != 9 || D < 1) {
        return false;
    }
    if ((cfg.WGMMA_S_N != 16 && cfg.WGMMA_S_N != 32 &&
         cfg.WGMMA_S_N != 64 && cfg.WGMMA_S_N != 128) ||
        (cfg.WGMMA_Y_N != 16 && cfg.WGMMA_Y_N != 32 &&
         cfg.WGMMA_Y_N != 64 && cfg.WGMMA_Y_N != 128)) {
        return false;
    }
    if ((cfg.BN % cfg.WGMMA_S_N) != 0 ||
        (cfg.BM % 64) != 0 ||
        cfg.num_stages < 1 ||
        cfg.num_stages > 3) {
        return false;
    }
    if (wgmma_threads_per_block(cfg) > device.max_threads_per_block) {
        return false;
    }
    return wgmma_dynamic_smem_bytes(cfg) <= device.max_dynamic_smem_bytes;
}

static LaunchPlan make_launch_plan(int D) {
    const DeviceInfo device = current_device_info();
    const MmaSyncConfig mma = default_mma_sync_config(D);
    const WgmmaConfig wgmma = default_wgmma_config(D);

    LaunchPlan plan;
    plan.device = device;
    plan.mma = mma;
    plan.wgmma = wgmma;

    if (valid_wgmma_config(device, wgmma, D)) {
        plan.mode = Tf32Mode::Wgmma;
        plan.output_ld = wgmma.D_y_f;
        plan.threads_per_block = wgmma_threads_per_block(wgmma);
        plan.dynamic_smem_bytes = wgmma_dynamic_smem_bytes(wgmma);
        return plan;
    }

    if (valid_mma_sync_config(device, mma, D)) {
        plan.mode = Tf32Mode::MmaSync;
        plan.output_ld = mma.D_f;
        plan.threads_per_block = mma_sync_threads_per_block(mma);
        plan.dynamic_smem_bytes = mma_sync_dynamic_smem_bytes(mma);
        return plan;
    }

    std::ostringstream oss;
    oss << "No valid TF32 relu_bat_c launch configuration for D=" << D
        << " on compute capability " << device.major << "." << device.minor
        << ".";
    fail("config", oss.str());
}

static std::string mma_sync_arch_option(const DeviceInfo& device) {
    if (device.major >= 12) {
        return "--gpu-architecture=compute_90";
    }

    std::ostringstream oss;
    oss << "--gpu-architecture=compute_" << device.major << device.minor;
    return oss.str();
}

static bool nvrtc_header_supports_cubin() {
#if defined(NVRTC_VERSION) && NVRTC_VERSION >= 11010
    return true;
#else
    return false;
#endif
}

static std::string wgmma_arch_option() {
    return nvrtc_header_supports_cubin()
               ? "--gpu-architecture=sm_90a"
               : "--gpu-architecture=compute_90a";
}

static std::string build_mma_sync_source(const MmaSyncConfig& cfg) {
    std::ostringstream src;
    src << "#define BM " << cfg.BM << "\n";
    src << "#define BN " << cfg.BN << "\n";
    src << "#define D_f " << cfg.D_f << "\n";
    src << "#define M_TILES " << cfg.M_TILES << "\n";
    src << "#define MMA_SYNC_TF32_STAGES " << cfg.num_stages << "\n";
    src << "#define MMA_SYNC_TF32_KERNEL_NAME relu_bat_c_tf32_mma_sync\n";
    src << "#define MMA_SYNC_TF32_PACK_KERNEL_NAME relu_bat_c_tf32_mma_sync_pack\n";
    src << "\n";
    src << read_text_file(kMmaSyncKernelPath);
    return src.str();
}

static std::string build_wgmma_source(const WgmmaConfig& cfg) {
    std::ostringstream src;
    src << "#define BM " << cfg.BM << "\n";
    src << "#define BN " << cfg.BN << "\n";
    src << "#define D_f " << cfg.D_y_f << "\n";
    src << "#define D_K_F " << cfg.D_k_f << "\n";
    src << "#define D_Y_F " << cfg.D_y_f << "\n";
    src << "#define WGMMA_S_N_SHAPE " << cfg.WGMMA_S_N << "\n";
    src << "#define WGMMA_Y_N_SHAPE " << cfg.WGMMA_Y_N << "\n";
    src << "#define SMEM_COPY_STAGES " << cfg.num_stages << "\n";
    src << "#define WGMMA_TF32_KERNEL_NAME relu_bat_c_tf32_wgmma\n";
    src << "#define WGMMA_TF32_PACK_KERNEL_NAME relu_bat_c_tf32_wgmma_pack\n";
    src << "#define WGMMA_TF32_PACK_ONLY 0\n";
    src << "#define WGMMA_FIRST_MMA_SS " << (cfg.first_mma_ss ? 1 : 0) << "\n";
    src << "\n";
    src << read_text_file(kWgmmaKernelPath);
    return src.str();
}

static std::vector<char> compile_program_to_image(
    const std::string& source,
    const std::string& name,
    const std::vector<std::string>& options,
    bool prefer_cubin) {
    nvrtcProgram prog = nullptr;
    check_nvrtc(
        nvrtcCreateProgram(&prog, source.c_str(), name.c_str(), 0, nullptr, nullptr),
        "nvrtcCreateProgram");

    std::vector<const char*> opt_ptrs;
    opt_ptrs.reserve(options.size());
    for (const std::string& opt : options) {
        opt_ptrs.push_back(opt.c_str());
    }

    nvrtcResult compile_res =
        nvrtcCompileProgram(prog, static_cast<int>(opt_ptrs.size()), opt_ptrs.data());

    size_t log_size = 0;
    check_nvrtc(nvrtcGetProgramLogSize(prog, &log_size), "nvrtcGetProgramLogSize");
    if (log_size > 1) {
        std::vector<char> log(log_size);
        check_nvrtc(nvrtcGetProgramLog(prog, log.data()), "nvrtcGetProgramLog");
        mexPrintf("%s\n", log.data());
    }

    if (compile_res != NVRTC_SUCCESS) {
        check_nvrtc(nvrtcDestroyProgram(&prog), "nvrtcDestroyProgram");
        fail("Compile", "NVRTC compilation failed.");
    }

#if defined(NVRTC_VERSION) && NVRTC_VERSION >= 11010
    if (prefer_cubin) {
        size_t cubin_size = 0;
        check_nvrtc(nvrtcGetCUBINSize(prog, &cubin_size), "nvrtcGetCUBINSize");
        if (cubin_size > 0) {
            std::vector<char> cubin(cubin_size);
            check_nvrtc(nvrtcGetCUBIN(prog, cubin.data()), "nvrtcGetCUBIN");
            check_nvrtc(nvrtcDestroyProgram(&prog), "nvrtcDestroyProgram");
            return cubin;
        }
    }
#else
    (void)prefer_cubin;
#endif

    size_t ptx_size = 0;
    check_nvrtc(nvrtcGetPTXSize(prog, &ptx_size), "nvrtcGetPTXSize");
    std::vector<char> ptx(ptx_size);
    check_nvrtc(nvrtcGetPTX(prog, ptx.data()), "nvrtcGetPTX");
    check_nvrtc(nvrtcDestroyProgram(&prog), "nvrtcDestroyProgram");
    return ptx;
}

static void maybe_set_dynamic_smem(CUfunction func, int smem_bytes) {
    if (smem_bytes < kDefaultDynamicSmemLimitBytes) {
        return;
    }
    check_cuda_drv(
        cuFuncSetAttribute(
            func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            smem_bytes),
        "cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)");
}

static ModuleEntry get_or_build_mma_sync_module(
    const DeviceInfo& device,
    const MmaSyncConfig& cfg,
    int smem_bytes) {
    std::ostringstream keyss;
    keyss << kMmaSyncKernelPath
          << "|cc=" << device.major << device.minor
          << "|BM=" << cfg.BM
          << "|BN=" << cfg.BN
          << "|D_f=" << cfg.D_f
          << "|M_TILES=" << cfg.M_TILES
          << "|stages=" << cfg.num_stages;
    const std::string key = keyss.str();

    auto it = g_module_cache.find(key);
    if (it != g_module_cache.end()) {
        maybe_set_dynamic_smem(it->second.kernel_func, smem_bytes);
        return it->second;
    }

    const std::string source = build_mma_sync_source(cfg);
    const std::string cuda_inc = cuda_include_option();
    const std::vector<std::string> options = {
        "--std=c++17",
        "--use_fast_math",
        mma_sync_arch_option(device),
        cuda_inc,
    };

    std::vector<char> image =
        compile_program_to_image(source, kMmaSyncKernelPath, options, false);

    ModuleEntry entry;
    check_cuda_drv(cuModuleLoadData(&entry.module, image.data()), "cuModuleLoadData");
    check_cuda_drv(
        cuModuleGetFunction(
            &entry.kernel_func,
            entry.module,
            "relu_bat_c_tf32_mma_sync"),
        "cuModuleGetFunction(relu_bat_c_tf32_mma_sync)");
    check_cuda_drv(
        cuModuleGetFunction(
            &entry.pack_func,
            entry.module,
            "relu_bat_c_tf32_mma_sync_pack"),
        "cuModuleGetFunction(relu_bat_c_tf32_mma_sync_pack)");
    maybe_set_dynamic_smem(entry.kernel_func, smem_bytes);

    g_module_cache.emplace(key, entry);
    return entry;
}

static ModuleEntry get_or_build_wgmma_module(
    const WgmmaConfig& cfg,
    int smem_bytes) {
    const std::string arch_option = wgmma_arch_option();
    const bool prefer_cubin = nvrtc_header_supports_cubin();

    std::ostringstream keyss;
    keyss << kWgmmaKernelPath
          << "|BM=" << cfg.BM
          << "|BN=" << cfg.BN
          << "|D_k_f=" << cfg.D_k_f
          << "|D_y_f=" << cfg.D_y_f
          << "|S_N=" << cfg.WGMMA_S_N
          << "|Y_N=" << cfg.WGMMA_Y_N
          << "|stages=" << cfg.num_stages
          << "|mode=" << (cfg.first_mma_ss ? "SS" : "RS");
    const std::string key = keyss.str();

    auto it = g_module_cache.find(key);
    if (it != g_module_cache.end()) {
        maybe_set_dynamic_smem(it->second.kernel_func, smem_bytes);
        return it->second;
    }

    const std::string source = build_wgmma_source(cfg);
    const std::string cuda_inc = cuda_include_option();
    const std::vector<std::string> options = {
        "--std=c++17",
        "--use_fast_math",
        "-diag-suppress=177",
        arch_option,
        cuda_inc,
    };

    std::vector<char> image =
        compile_program_to_image(
            source,
            kWgmmaKernelPath,
            options,
            prefer_cubin);

    ModuleEntry entry;
    check_cuda_drv(cuModuleLoadData(&entry.module, image.data()), "cuModuleLoadData");
    check_cuda_drv(
        cuModuleGetFunction(
            &entry.kernel_func,
            entry.module,
            "relu_bat_c_tf32_wgmma"),
        "cuModuleGetFunction(relu_bat_c_tf32_wgmma)");
    check_cuda_drv(
        cuModuleGetFunction(
            &entry.pack_func,
            entry.module,
            "relu_bat_c_tf32_wgmma_pack"),
        "cuModuleGetFunction(relu_bat_c_tf32_wgmma_pack)");
    maybe_set_dynamic_smem(entry.kernel_func, smem_bytes);

    g_module_cache.emplace(key, entry);
    return entry;
}

static void launch_mma_sync_dense(
    const LaunchPlan& plan,
    const float* A,
    const float* B,
    const float* C,
    float* Y,
    int N,
    int M,
    int D) {
    const MmaSyncConfig& cfg = plan.mma;
    const int num_panels = (N + cfg.BN - 1) / cfg.BN;

    if (num_panels == 0) {
        check_cuda_rt(
            cudaMemset(Y, 0, static_cast<size_t>(M) * cfg.D_f * sizeof(float)),
            "cudaMemset(Y)");
        return;
    }

    const size_t panel_elems =
        static_cast<size_t>(cfg.BN) * static_cast<size_t>(cfg.D_f);
    const size_t packed_bytes =
        static_cast<size_t>(num_panels) * panel_elems * sizeof(std::uint32_t);
    g_a_packed_cache.ensure(packed_bytes, "cudaMalloc(A_packed)");
    g_c_packed_cache.ensure(packed_bytes, "cudaMalloc(C_packed)");

    ModuleEntry entry =
        get_or_build_mma_sync_module(plan.device, cfg, plan.dynamic_smem_bytes);

    std::uint32_t* A_packed_ptr = g_a_packed_cache.as<std::uint32_t>();
    std::uint32_t* C_packed_ptr = g_c_packed_cache.as<std::uint32_t>();
    void* pack_args[] = {
        reinterpret_cast<void*>(&A),
        reinterpret_cast<void*>(&C),
        &A_packed_ptr,
        &C_packed_ptr,
        &N,
        &D,
    };

    const unsigned int panel_pairs =
        static_cast<unsigned int>(panel_elems / 2);
    const unsigned int pack_grid_x =
        (panel_pairs + kPackThreads - 1) / kPackThreads;

    check_cuda_drv(
        cuLaunchKernel(
            entry.pack_func,
            pack_grid_x, static_cast<unsigned int>(num_panels), 1,
            kPackThreads, 1, 1,
            0, nullptr,
            pack_args, nullptr),
        "cuLaunchKernel(pack)");

    void* kernel_args[] = {
        &A_packed_ptr,
        &C_packed_ptr,
        reinterpret_cast<void*>(&B),
        &Y,
        &N,
        &M,
        &D,
    };

    const unsigned int grid_x =
        static_cast<unsigned int>((M + cfg.BM - 1) / cfg.BM);

    check_cuda_drv(
        cuLaunchKernel(
            entry.kernel_func,
            grid_x, 1, 1,
            static_cast<unsigned int>(plan.threads_per_block), 1, 1,
            static_cast<unsigned int>(plan.dynamic_smem_bytes), nullptr,
            kernel_args, nullptr),
        "cuLaunchKernel(compute)");
}

static void launch_wgmma_dense(
    const LaunchPlan& plan,
    const float* A,
    const float* B,
    const float* C,
    float* Y,
    int N,
    int M,
    int D) {
    const WgmmaConfig& cfg = plan.wgmma;
    const int num_panels = (N + cfg.BN - 1) / cfg.BN;

    if (num_panels == 0) {
        check_cuda_rt(
            cudaMemset(Y, 0, static_cast<size_t>(M) * cfg.D_y_f * sizeof(float)),
            "cudaMemset(Y)");
        return;
    }

    const size_t a_panel_elems =
        static_cast<size_t>(cfg.BN) * static_cast<size_t>(cfg.D_k_f);
    const size_t c_panel_elems =
        static_cast<size_t>(cfg.BN) * static_cast<size_t>(cfg.D_y_f);
    g_a_packed_cache.ensure(
        static_cast<size_t>(num_panels) * a_panel_elems * sizeof(std::uint32_t),
        "cudaMalloc(A_packed)");
    g_c_packed_cache.ensure(
        static_cast<size_t>(num_panels) * c_panel_elems * sizeof(std::uint32_t),
        "cudaMalloc(C_packed)");

    ModuleEntry entry = get_or_build_wgmma_module(cfg, plan.dynamic_smem_bytes);

    std::uint32_t* A_packed_ptr = g_a_packed_cache.as<std::uint32_t>();
    std::uint32_t* C_packed_ptr = g_c_packed_cache.as<std::uint32_t>();
    void* pack_args[] = {
        reinterpret_cast<void*>(&A),
        reinterpret_cast<void*>(&C),
        &A_packed_ptr,
        &C_packed_ptr,
        &N,
        &D,
    };

    const size_t pack_work_elems = std::max(a_panel_elems, c_panel_elems);
    const unsigned int pack_grid_x =
        static_cast<unsigned int>((pack_work_elems + kPackThreads - 1) / kPackThreads);

    check_cuda_drv(
        cuLaunchKernel(
            entry.pack_func,
            pack_grid_x, static_cast<unsigned int>(num_panels), 1,
            kPackThreads, 1, 1,
            0, nullptr,
            pack_args, nullptr),
        "cuLaunchKernel(pack)");

    void* kernel_args[] = {
        &A_packed_ptr,
        &C_packed_ptr,
        reinterpret_cast<void*>(&B),
        &Y,
        &N,
        &M,
        &D,
    };

    const unsigned int grid_x =
        static_cast<unsigned int>((M + cfg.BM - 1) / cfg.BM);

    check_cuda_drv(
        cuLaunchKernel(
            entry.kernel_func,
            grid_x, 1, 1,
            static_cast<unsigned int>(plan.threads_per_block), 1, 1,
            static_cast<unsigned int>(plan.dynamic_smem_bytes), nullptr,
            kernel_args, nullptr),
        "cuLaunchKernel(compute)");
}

static void launch_dense(
    const LaunchPlan& plan,
    const float* A,
    const float* B,
    const float* C,
    float* Y,
    int N,
    int M,
    int D) {
    if (M == 0 || D == 0) {
        return;
    }

    if (plan.mode == Tf32Mode::Wgmma) {
        launch_wgmma_dense(plan, A, B, C, Y, N, M, D);
    } else {
        launch_mma_sync_dense(plan, A, B, C, Y, N, M, D);
    }
}

static int correction_columns_per_row(int D) {
    int columns = 1;
    while (columns < D && columns < 256) {
        columns <<= 1;
    }
    return columns;
}

static int log2_power_of_two(int value) {
    int log2 = 0;
    while ((1 << log2) < value) {
        ++log2;
    }
    return log2;
}

static int correction_rows_per_block(int D) {
    const int columns = correction_columns_per_row(D);
    const int target_threads = (columns < 256) ? 128 : 256;
    return std::max(1, target_threads / columns);
}

static float* ensure_edge_score_buffer(size_t edge_capacity) {
    if (edge_capacity == 0) {
        return nullptr;
    }
    g_edge_score_cache.ensure(
        edge_capacity * sizeof(float),
        "cudaMalloc(edge_scores)");
    return g_edge_score_cache.as<float>();
}

__global__ void compact_padded_y_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int D,
    int M,
    int src_ld) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = D * M;
    if (idx >= total) {
        return;
    }

    const int d = idx % D;
    const int m = idx / D;
    dst[static_cast<long long>(m) * D + d] =
        src[static_cast<long long>(m) * src_ld + d];
}

__global__ void sparse_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const long long* __restrict__ row_ptr,
    const long long* __restrict__ edge_i,
    float* __restrict__ edge_score,
    int M,
    int D) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= M) {
        return;
    }

    const long long first = row_ptr[row];
    const long long last = row_ptr[row + 1];
    if (first == last) {
        return;
    }

    extern __shared__ float b_row[];
    const float* __restrict__ b_src = B + static_cast<long long>(row) * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        b_row[d] = b_src[d];
    }
    __syncthreads();

    for (long long p = first + threadIdx.x; p < last; p += blockDim.x) {
        const int i = static_cast<int>(edge_i[p]);
        const float* __restrict__ a_row =
            A + static_cast<long long>(i) * D;
        float score = 0.0f;
        for (int k = 0; k < D; ++k) {
            score = fmaf(b_row[k], a_row[k], score);
        }
        edge_score[p] = score;
    }
}

__global__ void subtract_sparse_correction_kernel(
    const float* __restrict__ C,
    const long long* __restrict__ row_ptr,
    const long long* __restrict__ edge_i,
    const float* __restrict__ edge_val,
    const float* __restrict__ edge_score,
    float* __restrict__ Y,
    int D,
    int M,
    int y_ld,
    int columns_per_row,
    int rows_per_block,
    int column_shift) {
    const int local_row = static_cast<int>(threadIdx.x) >> column_shift;
    const int local_d = static_cast<int>(threadIdx.x) & (columns_per_row - 1);
    const int row = static_cast<int>(blockIdx.x) * rows_per_block + local_row;
    const int d = static_cast<int>(blockIdx.y) * columns_per_row + local_d;

    if (row >= M || d >= D) {
        return;
    }

    float correction = 0.0f;
    const long long first = row_ptr[row];
    const long long last = row_ptr[row + 1];

    for (long long p = first; p < last; ++p) {
        const int i = static_cast<int>(edge_i[p]);
        const float lij = edge_score[p];
        const float wij = edge_val[p] - lij + fmaxf(lij, 0.0f);
        correction = fmaf(wij, C[static_cast<long long>(i) * D + d], correction);
    }

    Y[static_cast<long long>(row) * y_ld + d] -= correction;
}

static void compact_padded_y(
    const float* src,
    float* dst,
    int D,
    int M,
    int src_ld) {
    if (src_ld == D) {
        return;
    }

    const int threads = 256;
    const int total = D * M;
    const int blocks = (total + threads - 1) / threads;
    if (blocks > 0) {
        compact_padded_y_kernel<<<blocks, threads>>>(src, dst, D, M, src_ld);
        check_cuda_rt(cudaGetLastError(), "compact_padded_y_kernel launch");
    }
}

static void subtract_sparse_correction(
    const float* A,
    const float* B,
    const float* C,
    const long long* row_ptr,
    const long long* edge_i,
    const float* edge_val,
    float* edge_score_ptr,
    size_t edge_capacity,
    float* Y,
    int D,
    int M,
    int y_ld) {
    if (M == 0 || D == 0 || edge_capacity == 0 || edge_score_ptr == nullptr) {
        return;
    }

    const int score_threads = 128;
    const size_t score_smem_bytes = static_cast<size_t>(D) * sizeof(float);
    sparse_matmul_kernel<<<M, score_threads, score_smem_bytes>>>(
        A, B, row_ptr, edge_i, edge_score_ptr, M, D);
    check_cuda_rt(cudaGetLastError(), "sparse_matmul_kernel launch");

    const int corr_columns = correction_columns_per_row(D);
    const int corr_rows = correction_rows_per_block(D);
    const int corr_threads = corr_columns * corr_rows;
    const int corr_column_shift = log2_power_of_two(corr_columns);
    dim3 corr_grid(
        static_cast<unsigned int>((M + corr_rows - 1) / corr_rows),
        static_cast<unsigned int>((D + corr_columns - 1) / corr_columns));
    subtract_sparse_correction_kernel<<<corr_grid, corr_threads>>>(
        C,
        row_ptr,
        edge_i,
        edge_val,
        edge_score_ptr,
        Y,
        D,
        M,
        y_ld,
        corr_columns,
        corr_rows,
        corr_column_shift);
    check_cuda_rt(cudaGetLastError(), "subtract_sparse_correction_kernel launch");
}

}  // namespace relu_bat_c_tf32_nvrtc
