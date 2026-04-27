#include "mex.h"
#include "gpu/mxGPUArray.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

static void fail(const char* id, const std::string& msg) {
    mexErrMsgIdAndTxt(id, "%s", msg.c_str());
}

static void checkCudaDrv(CUresult err, const char* where) {
    if (err != CUDA_SUCCESS) {
        const char* name = nullptr;
        const char* str = nullptr;
        cuGetErrorName(err, &name);
        cuGetErrorString(err, &str);
        std::ostringstream oss;
        oss << where << " failed: "
            << (name ? name : "CUDA_ERROR")
            << " - "
            << (str ? str : "unknown");
        fail("nvrtc_relu_bat_c_fused_mex:CUDA", oss.str());
    }
}

static void checkNvrtc(nvrtcResult err, const char* where) {
    if (err != NVRTC_SUCCESS) {
        std::ostringstream oss;
        oss << where << " failed: " << nvrtcGetErrorString(err);
        fail("nvrtc_relu_bat_c_fused_mex:NVRTC", oss.str());
    }
}

static std::string read_text_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        std::ostringstream oss;
        oss << "Could not open CUDA source file: " << path;
        fail("nvrtc_relu_bat_c_fused_mex:FileOpen", oss.str());
    }

    std::ostringstream buffer;
    buffer << ifs.rdbuf();

    if (!ifs.good() && !ifs.eof()) {
        std::ostringstream oss;
        oss << "Error reading CUDA source file: " << path;
        fail("nvrtc_relu_bat_c_fused_mex:FileRead", oss.str());
    }

    return buffer.str();
}

static std::string build_source_from_file(const std::string& kernel_path, int BK, int MS, int V, int NUM_THREADS) {
    std::ostringstream src;
    src << "#define BK " << BK << "\n";
    src << "#define MS " << MS << "\n";
    src << "#define V "  << V  << "\n";
    src << "#define NUM_THREADS " << NUM_THREADS << "\n";
    src << "\n";
    src << read_text_file(kernel_path);
    return src.str();
}

struct ModuleEntry {
    CUmodule module = nullptr;
    CUfunction func = nullptr;
};

static std::unordered_map<std::string, ModuleEntry> g_cache;
static bool g_cuda_initialized = false;

static ModuleEntry get_or_build_module(const std::string& kernel_path, int BK, int MS, int V, int NUM_THREADS)
{
    std::ostringstream keyss;
    keyss << kernel_path << "|BK=" << BK << "|MS=" << MS << "|V=" << V;
    const std::string key = keyss.str();

    auto it = g_cache.find(key);
    if (it != g_cache.end()) {
        return it->second;
    }

    if (!g_cuda_initialized) {
        checkCudaDrv(cuInit(0), "cuInit");
        g_cuda_initialized = true;
    }

    CUcontext ctx = nullptr;
    checkCudaDrv(cuCtxGetCurrent(&ctx), "cuCtxGetCurrent");
    if (ctx == nullptr) {
        fail("nvrtc_relu_bat_c_fused_mex:Context",
             "No active CUDA context. Create a gpuArray or call gpuDevice first.");
    }

    int device = 0;
    checkCudaDrv(cuCtxGetDevice(&device), "cuCtxGetDevice");

    int major = 0, minor = 0;
    checkCudaDrv(
        cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
        "cuDeviceGetAttribute(major)");
    checkCudaDrv(
        cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
        "cuDeviceGetAttribute(minor)");

    const std::string src = build_source_from_file(kernel_path, BK, MS, V, NUM_THREADS);

    nvrtcProgram prog;
    checkNvrtc(
        nvrtcCreateProgram(&prog, src.c_str(), kernel_path.c_str(), 0, nullptr, nullptr),
        "nvrtcCreateProgram");

    // std::string gpu_arch = "--gpu-architecture=compute_" +
                        //    std::to_string(major) + std::to_string(minor); //Matlabs packaged cuda is too old to know sm120 ... forcing older arch here
    std::string gpu_arch = "--gpu-architecture=compute_90";
    const char* opts[] = {
        "--std=c++14",
        "--use_fast_math",
        gpu_arch.c_str()
    };

    nvrtcResult compile_res = nvrtcCompileProgram(prog, 3, opts);

    size_t log_size = 0;
    checkNvrtc(nvrtcGetProgramLogSize(prog, &log_size), "nvrtcGetProgramLogSize");
    if (log_size > 1) {
        std::vector<char> log(log_size);
        checkNvrtc(nvrtcGetProgramLog(prog, log.data()), "nvrtcGetProgramLog");
        mexPrintf("%s\n", log.data());
    }

    if (compile_res != NVRTC_SUCCESS) {
        checkNvrtc(nvrtcDestroyProgram(&prog), "nvrtcDestroyProgram");
        fail("nvrtc_relu_bat_c_fused_mex:Compile", "NVRTC compilation failed.");
    }

    size_t ptx_size = 0;
    checkNvrtc(nvrtcGetPTXSize(prog, &ptx_size), "nvrtcGetPTXSize");

    std::vector<char> ptx(ptx_size);
    checkNvrtc(nvrtcGetPTX(prog, ptx.data()), "nvrtcGetPTX");
    checkNvrtc(nvrtcDestroyProgram(&prog), "nvrtcDestroyProgram");

    CUmodule module = nullptr;
    checkCudaDrv(cuModuleLoadData(&module, ptx.data()), "cuModuleLoadData");

    CUfunction func = nullptr;
    checkCudaDrv(
        cuModuleGetFunction(&func, module, "relu_bat_c_fused_kernel_float4_sync"),
        "cuModuleGetFunction");

    ModuleEntry entry;
    entry.module = module;
    entry.func = func;
    g_cache.emplace(key, entry);
    return entry;
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    mxInitGPU();

    if (nrhs != 8) {
        fail("nvrtc_relu_bat_c_fused_mex:nrhs",
             "Usage: Yt = nvrtc_relu_bat_c_fused_mex(At, Bt, Ct, kernelPath, BK, MS, V, threads)");
    }
    if (nlhs > 1) {
        fail("nvrtc_relu_bat_c_fused_mex:nlhs", "One output expected.");
    }

    const mxGPUArray* At_gpu = mxGPUCreateFromMxArray(prhs[0]);
    const mxGPUArray* Bt_gpu = mxGPUCreateFromMxArray(prhs[1]);
    const mxGPUArray* Ct_gpu = mxGPUCreateFromMxArray(prhs[2]);

    if (mxGPUGetClassID(At_gpu) != mxSINGLE_CLASS ||
        mxGPUGetClassID(Bt_gpu) != mxSINGLE_CLASS ||
        mxGPUGetClassID(Ct_gpu) != mxSINGLE_CLASS) {
        fail("nvrtc_relu_bat_c_fused_mex:type", "At, Bt, Ct must be gpuArray(single).");
    }

    if (!mxIsChar(prhs[3])) {
        fail("nvrtc_relu_bat_c_fused_mex:path", "kernelPath must be a character vector or string scalar.");
    }

    char* kernel_path_c = mxArrayToString(prhs[3]);
    if (kernel_path_c == nullptr) {
        fail("nvrtc_relu_bat_c_fused_mex:path", "Failed to read kernelPath.");
    }
    std::string kernel_path(kernel_path_c);
    mxFree(kernel_path_c);

    const int BK = static_cast<int>(mxGetScalar(prhs[4]));
    const int MS = static_cast<int>(mxGetScalar(prhs[5]));
    const int V  = static_cast<int>(mxGetScalar(prhs[6]));
    const int threads = static_cast<int>(mxGetScalar(prhs[7]));

    const mwSize* At_dims = mxGPUGetDimensions(At_gpu); // D x N
    const mwSize* Bt_dims = mxGPUGetDimensions(Bt_gpu); // D x M
    const mwSize* Ct_dims = mxGPUGetDimensions(Ct_gpu); // D x N

    const int D = static_cast<int>(At_dims[0]);
    const int N = static_cast<int>(At_dims[1]);
    const int M = static_cast<int>(Bt_dims[1]);

    if ((int)Bt_dims[0] != D || (int)Ct_dims[0] != D || (int)Ct_dims[1] != N) {
        fail("nvrtc_relu_bat_c_fused_mex:shape", "Dimension mismatch among At, Bt, Ct.");
    }

    if (D != 4 * V) {
        fail("nvrtc_relu_bat_c_fused_mex:V", "Expected D == 4 * V for this float4 kernel.");
    }

    mwSize out_dims[2] = { static_cast<mwSize>(D), static_cast<mwSize>(M) };
    mxGPUArray* Yt_gpu = mxGPUCreateGPUArray(
        2, out_dims, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    const float* At_ptr = static_cast<const float*>(mxGPUGetDataReadOnly(At_gpu));
    const float* Bt_ptr = static_cast<const float*>(mxGPUGetDataReadOnly(Bt_gpu));
    const float* Ct_ptr = static_cast<const float*>(mxGPUGetDataReadOnly(Ct_gpu));
    float* Yt_ptr = static_cast<float*>(mxGPUGetData(Yt_gpu));

    ModuleEntry entry = get_or_build_module(kernel_path, BK, MS, V, threads);

    void* args[] = {
        (void*)&At_ptr,
        (void*)&Bt_ptr,
        (void*)&Ct_ptr,
        (void*)&Yt_ptr,
        (void*)&N,
        (void*)&M,
        (void*)&D
    };

    const int rows_per_block = MS * threads;
    const unsigned int grid_x = (M + rows_per_block - 1) / rows_per_block;

    checkCudaDrv(
        cuLaunchKernel(entry.func,
                       grid_x, 1, 1,
                       threads, 1, 1,
                       0, 0,
                       args, nullptr),
        "cuLaunchKernel");

    checkCudaDrv(cuCtxSynchronize(), "cuCtxSynchronize");

    plhs[0] = mxGPUCreateMxArrayOnGPU(Yt_gpu);

    mxGPUDestroyGPUArray(At_gpu);
    mxGPUDestroyGPUArray(Bt_gpu);
    mxGPUDestroyGPUArray(Ct_gpu);
    mxGPUDestroyGPUArray(Yt_gpu);
}