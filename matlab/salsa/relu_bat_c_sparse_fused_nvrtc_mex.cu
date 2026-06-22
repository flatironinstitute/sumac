#include "mex.h"
#include "gpu/mxGPUArray.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cublas_v2.h>

#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

const std::string kernelPath = "../../relu_batc_jit/kernel_fused.cu";
const std::string kernelPathMixed = "../../relu_batc_jit/kernel_fused_mixed.cu";

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
        fail("nvrtc_relu_bat_c_sparse_fused_mex:CUDA", oss.str());
    }
}

static void checkNvrtc(nvrtcResult err, const char* where) {
    if (err != NVRTC_SUCCESS) {
        std::ostringstream oss;
        oss << where << " failed: " << nvrtcGetErrorString(err);
        fail("nvrtc_relu_bat_c_sparse_fused_mex:NVRTC", oss.str());
    }
}

static void checkCudaRt(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << where << " failed: " << cudaGetErrorString(err);
        fail("nvrtc_relu_bat_c_sparse_fused_mex:CUDA", oss.str());
    }
}

static const char* cublas_status_to_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        default: return "CUBLAS_STATUS_UNKNOWN";
    }
}

static void checkCublas(cublasStatus_t err, const char* where) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        std::ostringstream oss;
        oss << where << " failed: " << cublas_status_to_string(err);
        fail("nvrtc_relu_bat_c_sparse_fused_mex:CUBLAS", oss.str());
    }
}

__global__ void relu_inplace_kernel(float* x, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = fmaxf(x[i], 0.0f);
    }
}

__global__ void subtract_stepc_from_y_kernel(float* __restrict__ Yt,
                                             const float* __restrict__ Ct,
                                             const float* __restrict__ scores,
                                             const long long* __restrict__ row_ptr,
                                             const long long* __restrict__ edge_i,
                                             const float* __restrict__ edge_val,
                                             int D,
                                             int M) {
    const int row = static_cast<int>(blockIdx.x);
    const int d = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;

    if (row >= M || d >= D) {
        return;
    }

    float correction = 0.0f;
    const long long first = row_ptr[row];
    const long long last = row_ptr[row + 1];

    for (long long p = first; p < last; ++p) {
        const int i = static_cast<int>(edge_i[p]);
        const float lij = scores[row + static_cast<size_t>(i) * M];
        const float wij = edge_val[p] - lij + fmaxf(lij, 0.0f);
        correction = fmaf(wij, Ct[d + static_cast<size_t>(i) * D], correction);
    }

    Yt[d + static_cast<size_t>(row) * D] = -correction;
}

static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        checkCublas(cublasCreate(&handle), "cublasCreate");
        checkCublas(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
        checkCublas(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH), "cublasSetMathMode");
    }
    return handle;
}

static void run_cublas_sparse_relu_fallback(const float* At_ptr,
                                            const float* Bt_ptr,
                                            const float* Ct_ptr,
                                            const long long* row_ptr,
                                            const long long* edge_i,
                                            const float* edge_val,
                                            float* Yt_ptr,
                                            int D,
                                            int N,
                                            int M) {
    float* tmp_ptr = nullptr;
    const size_t tmp_elems = static_cast<size_t>(M) * static_cast<size_t>(N);
    checkCudaRt(cudaMalloc(reinterpret_cast<void**>(&tmp_ptr), tmp_elems * sizeof(float)),
                "cudaMalloc(tmp)");

    cublasHandle_t handle = get_cublas_handle();
    const float alpha = 1.0f;
    const float beta0 = 0.0f;
    const float beta1 = 1.0f;

    // Tmp(M x N) = Bt'(M x D) * At(D x N).
    checkCublas(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                            M, N, D,
                            &alpha,
                            Bt_ptr, D,
                            At_ptr, D,
                            &beta0,
                            tmp_ptr, M),
                "cublasSgemm(Bt' * At)");

    // Initialize Yt to -stepC
    const int corr_threads = 256;
    dim3 corr_grid(static_cast<unsigned int>(M),
                   static_cast<unsigned int>((D + corr_threads - 1) / corr_threads));
    if (M > 0 && D > 0) {
        subtract_stepc_from_y_kernel<<<corr_grid, corr_threads>>>(
            Yt_ptr, Ct_ptr, tmp_ptr, row_ptr, edge_i, edge_val, D, M);
        checkCudaRt(cudaGetLastError(), "subtract_stepc_from_y_kernel launch");
    }

    const int relu_threads = 256;
    const unsigned int relu_blocks =
        static_cast<unsigned int>((tmp_elems + relu_threads - 1) / relu_threads);
    if (tmp_elems > 0) {
        relu_inplace_kernel<<<relu_blocks, relu_threads>>>(tmp_ptr, tmp_elems);
        checkCudaRt(cudaGetLastError(), "relu_inplace_kernel launch");
    }

    // Yt(D x M) += Ct(D x N) * ReLU(Tmp)'(N x M).
    checkCublas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                            D, M, N,
                            &alpha,
                            Ct_ptr, D,
                            tmp_ptr, M,
                            &beta1,
                            Yt_ptr, D),
                "cublasSgemm(Ct * Tmp')");

    checkCudaRt(cudaFree(tmp_ptr), "cudaFree(tmp)");
    checkCudaRt(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

static std::string read_text_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        std::ostringstream oss;
        oss << "Could not open CUDA source file: " << path;
        fail("nvrtc_relu_bat_c_sparse_fused_mex:FileOpen", oss.str());
    }

    std::ostringstream buffer;
    buffer << ifs.rdbuf();

    if (!ifs.good() && !ifs.eof()) {
        std::ostringstream oss;
        oss << "Error reading CUDA source file: " << path;
        fail("nvrtc_relu_bat_c_sparse_fused_mex:FileRead", oss.str());
    }

    return buffer.str();
}

static std::string build_source_from_file(int BK, int MS, int V, int NUM_THREADS) {
    std::ostringstream src;
    src << "#define BK " << BK << "\n";
    src << "#define MS " << MS << "\n";
    src << "#define V "  << V  << "\n";
    src << "#define NUM_THREADS " << NUM_THREADS << "\n";
    src << "\n";
    src << read_text_file(kernelPath);
    return src.str();
}

static std::string build_source_mixed_from_file(int BK, int MS, int V, int R, int NUM_THREADS) {
    std::ostringstream src;
    src << "#define BK " << BK << "\n";
    src << "#define MS " << MS << "\n";
    src << "#define V "  << V  << "\n";
    src << "#define R "  << R  << "\n";
    src << "#define NUM_THREADS " << NUM_THREADS << "\n";
    src << "\n";
    src << read_text_file(kernelPathMixed);
    return src.str();
}

struct ModuleEntry {
    CUmodule module = nullptr;
    CUfunction func = nullptr;
};

static std::unordered_map<std::string, ModuleEntry> g_cache;
static bool g_cuda_initialized = false;

static ModuleEntry get_or_build_module(int BK, int MS, int V, int NUM_THREADS)
{
    std::ostringstream keyss;
    keyss << kernelPath << "|BK=" << BK << "|MS=" << MS << "|V=" << V;
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
        fail("nvrtc_relu_bat_c_sparse_fused_mex:Context",
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

    const std::string src = build_source_from_file(BK, MS, V, NUM_THREADS);

    nvrtcProgram prog;
    checkNvrtc(
        nvrtcCreateProgram(&prog, src.c_str(), kernelPath.c_str(), 0, nullptr, nullptr),
        "nvrtcCreateProgram");

    std::string gpu_arch = "--gpu-architecture=compute_" +
                           std::to_string(major) + std::to_string(minor);
    if (major >= 12) {
        gpu_arch = "--gpu-architecture=compute_90";
    }
    const char* opts[] = {
        "--std=c++17",
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
        fail("nvrtc_relu_bat_c_sparse_fused_mex:Compile", "NVRTC compilation failed.");
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
        cuModuleGetFunction(&func, module, "relu_bat_c_sparse_fused_kernel_float4_sync"),
        "cuModuleGetFunction");

    ModuleEntry entry;
    entry.module = module;
    entry.func = func;
    g_cache.emplace(key, entry);
    return entry;
}

static ModuleEntry get_or_build_module_mixed(int BK, int MS, int V, int R, int NUM_THREADS)
{
    std::ostringstream keyss;
    keyss << kernelPathMixed << "|BK=" << BK << "|MS=" << MS << "|V=" << V << "|R=" << R;
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
        fail("nvrtc_relu_bat_c_sparse_fused_mex:Context",
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

    const std::string src = build_source_mixed_from_file(BK, MS, V, R, NUM_THREADS);

    nvrtcProgram prog;
    checkNvrtc(
        nvrtcCreateProgram(&prog, src.c_str(), kernelPathMixed.c_str(), 0, nullptr, nullptr),
        "nvrtcCreateProgram");

    std::string gpu_arch = "--gpu-architecture=compute_" +
                           std::to_string(major) + std::to_string(minor);
    if (major >= 12) {
        gpu_arch = "--gpu-architecture=compute_90";
    }
    const char* opts[] = {
        "--std=c++17",
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
        fail("nvrtc_relu_bat_c_sparse_fused_mex:Compile", "NVRTC compilation failed.");
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
        cuModuleGetFunction(&func, module, "relu_bat_c_sparse_fused_kernel_mixed_sync"),
        "cuModuleGetFunction");

    ModuleEntry entry;
    entry.module = module;
    entry.func = func;
    g_cache.emplace(key, entry);
    return entry;
}

static int checked_int_dim(mwSize value, const char* name) {
    if (value > static_cast<mwSize>(std::numeric_limits<int>::max())) {
        std::ostringstream oss;
        oss << name << " exceeds int range.";
        fail("nvrtc_relu_bat_c_sparse_fused_mex:shape", oss.str());
    }
    return static_cast<int>(value);
}

static mwSize scalar_offset(const mxArray* arr, const char* name) {
    if (!mxIsNumeric(arr) || mxIsComplex(arr) || mxGetNumberOfElements(arr) != 1) {
        std::ostringstream oss;
        oss << name << " must be a real numeric scalar.";
        fail("nvrtc_relu_bat_c_sparse_fused_mex:offset", oss.str());
    }

    const double value = mxGetScalar(arr);
    if (value < 0.0 || value > static_cast<double>(std::numeric_limits<mwSize>::max())) {
        std::ostringstream oss;
        oss << name << " is out of range.";
        fail("nvrtc_relu_bat_c_sparse_fused_mex:offset", oss.str());
    }

    return static_cast<mwSize>(value);
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    mxInitGPU();

    if (nrhs != 6 && nrhs != 8) {
        fail("nvrtc_relu_bat_c_sparse_fused_mex:nrhs",
             "Usage: Yt = relu_bat_c_sparse_fused_nvrtc_mex(At, Bt, Ct, rowPtr, edgeI, edgeVal[, rowPtrBase, edgeBase])");
    }
    if (nlhs > 1) {
        fail("nvrtc_relu_bat_c_sparse_fused_mex:nlhs", "One output expected.");
    }

    const mxGPUArray* At_gpu = mxGPUCreateFromMxArray(prhs[0]);
    const mxGPUArray* Bt_gpu = mxGPUCreateFromMxArray(prhs[1]);
    const mxGPUArray* Ct_gpu = mxGPUCreateFromMxArray(prhs[2]);
    const mxGPUArray* row_ptr_gpu = mxGPUCreateFromMxArray(prhs[3]);
    const mxGPUArray* edge_i_gpu = mxGPUCreateFromMxArray(prhs[4]);
    const mxGPUArray* edge_val_gpu = mxGPUCreateFromMxArray(prhs[5]);

    if (mxGPUGetClassID(At_gpu) != mxSINGLE_CLASS ||
        mxGPUGetClassID(Bt_gpu) != mxSINGLE_CLASS ||
        mxGPUGetClassID(Ct_gpu) != mxSINGLE_CLASS ||
        mxGPUGetClassID(edge_val_gpu) != mxSINGLE_CLASS) {
        fail("nvrtc_relu_bat_c_sparse_fused_mex:type",
             "At, Bt, Ct, and edgeVal must be gpuArray(single).");
    }

    if (mxGPUGetClassID(row_ptr_gpu) != mxINT64_CLASS ||
        mxGPUGetClassID(edge_i_gpu) != mxINT64_CLASS) {
        fail("nvrtc_relu_bat_c_sparse_fused_mex:type",
             "rowPtr and edgeI must be gpuArray(int64).");
    }

    const mwSize* At_dims = mxGPUGetDimensions(At_gpu); // D x N
    const mwSize* Bt_dims = mxGPUGetDimensions(Bt_gpu); // D x M
    const mwSize* Ct_dims = mxGPUGetDimensions(Ct_gpu); // D x N

    const int D = checked_int_dim(At_dims[0], "D");
    const int N = checked_int_dim(At_dims[1], "N");
    const int M = checked_int_dim(Bt_dims[1], "M");

    if ((int)Bt_dims[0] != D || (int)Ct_dims[0] != D || (int)Ct_dims[1] != N) {
        fail("nvrtc_relu_bat_c_sparse_fused_mex:shape", "Dimension mismatch among At, Bt, Ct.");
    }

    const mwSize n_row_ptr = mxGPUGetNumberOfElements(row_ptr_gpu);
    const mwSize n_edge_i = mxGPUGetNumberOfElements(edge_i_gpu);
    const mwSize n_edge_val = mxGPUGetNumberOfElements(edge_val_gpu);
    const mwSize row_ptr_offset = (nrhs == 8) ? scalar_offset(prhs[6], "rowPtrBase") : 0;
    const mwSize edge_offset = (nrhs == 8) ? scalar_offset(prhs[7], "edgeBase") : 0;

    if (nrhs == 6 && n_row_ptr != static_cast<mwSize>(M + 1)) {
        fail("nvrtc_relu_bat_c_sparse_fused_mex:shape",
             "rowPtr must have length size(Bt,2)+1.");
    }
    if (nrhs == 8 && row_ptr_offset + static_cast<mwSize>(M + 1) > n_row_ptr) {
        fail("nvrtc_relu_bat_c_sparse_fused_mex:shape",
             "rowPtrBase plus size(Bt,2)+1 exceeds rowPtr length.");
    }
    if (n_edge_i != n_edge_val) {
        fail("nvrtc_relu_bat_c_sparse_fused_mex:shape",
             "edgeI and edgeVal must have the same number of elements.");
    }
    if (edge_offset > n_edge_i) {
        fail("nvrtc_relu_bat_c_sparse_fused_mex:shape",
             "edgeBase exceeds edgeI length.");
    }

    const int BK = 32;
    int MS = 1;
    if (D < 32) {
        MS = 2;
    } else if (D < 64) {
        MS = 2;
    }
    const int V = D / 4;
    const int R = D % 4;
    const int threads = 128;
    const bool use_cublas_fallback = (D >= 128);

    mwSize out_dims[2] = { static_cast<mwSize>(D), static_cast<mwSize>(M) };
    mxGPUArray* Yt_gpu = mxGPUCreateGPUArray(
        2, out_dims, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    const float* At_ptr = static_cast<const float*>(mxGPUGetDataReadOnly(At_gpu));
    const float* Bt_ptr = static_cast<const float*>(mxGPUGetDataReadOnly(Bt_gpu));
    const float* Ct_ptr = static_cast<const float*>(mxGPUGetDataReadOnly(Ct_gpu));
    const long long* row_ptr_data = static_cast<const long long*>(mxGPUGetDataReadOnly(row_ptr_gpu));
    const long long* edge_i_data = static_cast<const long long*>(mxGPUGetDataReadOnly(edge_i_gpu));
    const float* edge_val_data = static_cast<const float*>(mxGPUGetDataReadOnly(edge_val_gpu));
    const long long* row_ptr = row_ptr_data + row_ptr_offset;
    const long long* edge_i = edge_i_data + edge_offset;
    const float* edge_val = edge_val_data + edge_offset;
    float* Yt_ptr = static_cast<float*>(mxGPUGetData(Yt_gpu));

    if (use_cublas_fallback) {
        run_cublas_sparse_relu_fallback(
            At_ptr, Bt_ptr, Ct_ptr, row_ptr, edge_i, edge_val, Yt_ptr, D, N, M);
    } else {
        ModuleEntry entry = (R == 0) ?
            get_or_build_module(BK, MS, V, threads) :
            get_or_build_module_mixed(BK, MS, V, R, threads);

        void* args[] = {
            (void*)&At_ptr,
            (void*)&Bt_ptr,
            (void*)&Ct_ptr,
            (void*)&row_ptr,
            (void*)&edge_i,
            (void*)&edge_val,
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
    }

    plhs[0] = mxGPUCreateMxArrayOnGPU(Yt_gpu);

    mxGPUDestroyGPUArray(At_gpu);
    mxGPUDestroyGPUArray(Bt_gpu);
    mxGPUDestroyGPUArray(Ct_gpu);
    mxGPUDestroyGPUArray(row_ptr_gpu);
    mxGPUDestroyGPUArray(edge_i_gpu);
    mxGPUDestroyGPUArray(edge_val_gpu);
    mxGPUDestroyGPUArray(Yt_gpu);
}
