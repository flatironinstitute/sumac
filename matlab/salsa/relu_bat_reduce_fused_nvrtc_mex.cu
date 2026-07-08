#include "mex.h"
#include "gpu/mxGPUArray.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <cublas_v2.h>

#include <cub/cub.cuh>

#include <cstdint>
#include <cstdlib>
#include <limits>

#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

const std::string kernelPath = "../../src/sumac/kernels/relu_bat_reduce_jit/kernel_nnz.cu";
const std::string kernelPath_mixed = "../../src/sumac/kernels/relu_bat_reduce_jit/kernel_nnz_mixed.cu";

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
        fail("nvrtc_relu_bat_reduce_fused_mex:CUDA", oss.str());
    }
}

static void checkNvrtc(nvrtcResult err, const char* where) {
    if (err != NVRTC_SUCCESS) {
        std::ostringstream oss;
        oss << where << " failed: " << nvrtcGetErrorString(err);
        fail("nvrtc_relu_bat_reduce_fused_mex:NVRTC", oss.str());
    }
}

static void checkCudaRt(cudaError_t err, const char* where) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << where << " failed: " << cudaGetErrorString(err);
        fail("nvrtc_relu_bat_reduce_fused_mex:CUDA", oss.str());
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
        fail("nvrtc_relu_bat_reduce_fused_mex:CUBLAS", oss.str());
    }
}

__global__ void relu_inplace_kernel(float* x, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = fmaxf(x[i], 0.0f);
    }
}


struct AsDouble {
    __host__ __device__ __forceinline__
    double operator()(const float& x) const {
        return static_cast<double>(x);
    }
};

struct SquareAsDouble {
    __host__ __device__ __forceinline__
    double operator()(const float& x) const {
        const double y = static_cast<double>(x);
        return y * y;
    }
};

struct IsPositiveULL {
    __host__ __device__ __forceinline__
    unsigned long long operator()(const float& x) const {
        return static_cast<unsigned long long>(x > 0.0f);
    }
};

template <typename InputIt, typename OutputT>
static void cub_sum_reduce(InputIt d_in,
                           OutputT* d_out,
                           int num_items,
                           const char* where) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    checkCudaRt(
        cub::DeviceReduce::Sum(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items),
        where);

    checkCudaRt(
        cudaMalloc(&d_temp_storage, temp_storage_bytes),
        "cudaMalloc(CUB temp storage)");

    checkCudaRt(
        cub::DeviceReduce::Sum(
            d_temp_storage, temp_storage_bytes, d_in, d_out, num_items),
        where);

    checkCudaRt(cudaFree(d_temp_storage), "cudaFree(CUB temp storage)");
}

static void run_cublas_cub_relu_reduce_fallback(
    const float* At_ptr,                 
    const float* Bt_ptr,                 
    double* out_sum_ptr,
    double* out_ssq_ptr,
    unsigned long long* out_nnz_ptr,
    int D,
    int N,
    int M) {

    const size_t tmp_elems_size =
        static_cast<size_t>(M) * static_cast<size_t>(N);

    if (tmp_elems_size > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fail("nvrtc_relu_bat_reduce_fused_mex:CUBSize",
             "CUB fallback currently requires M*N <= INT_MAX.");
    }

    const int tmp_elems = static_cast<int>(tmp_elems_size);

    float* tmp_ptr = nullptr;
    checkCudaRt(
        cudaMalloc(reinterpret_cast<void**>(&tmp_ptr),
                   tmp_elems_size * sizeof(float)),
        "cudaMalloc(tmp)");

    cublasHandle_t handle = nullptr;
    checkCublas(cublasCreate(&handle), "cublasCreate");

    const float alpha = 1.0f;
    const float beta = 0.0f;

    checkCublas(
        cublasSgemm(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    M, N, D,
                    &alpha,
                    Bt_ptr, D,
                    At_ptr, D,
                    &beta,
                    tmp_ptr, M),
        "cublasSgemm(Bt' * At)");

    const int relu_threads = 256;
    const unsigned int relu_blocks =
        static_cast<unsigned int>((tmp_elems_size + relu_threads - 1) /
                                  relu_threads);

    if (tmp_elems_size > 0) {
        relu_inplace_kernel<<<relu_blocks, relu_threads>>>(tmp_ptr, tmp_elems_size);
        checkCudaRt(cudaGetLastError(), "relu_inplace_kernel launch");
    }

    cub::TransformInputIterator<double, AsDouble, const float*>
        sum_it(tmp_ptr, AsDouble());

    cub::TransformInputIterator<double, SquareAsDouble, const float*>
        ssq_it(tmp_ptr, SquareAsDouble());

    cub::TransformInputIterator<unsigned long long, IsPositiveULL, const float*>
        nnz_it(tmp_ptr, IsPositiveULL());

    if (tmp_elems > 0) {
        cub_sum_reduce(sum_it, out_sum_ptr, tmp_elems, "CUB reduce sum");
        cub_sum_reduce(ssq_it, out_ssq_ptr, tmp_elems, "CUB reduce ssq");
        cub_sum_reduce(nnz_it, out_nnz_ptr, tmp_elems, "CUB reduce nnz");
    }

    checkCudaRt(cudaFree(tmp_ptr), "cudaFree(tmp)");
    checkCublas(cublasDestroy(handle), "cublasDestroy");
    checkCudaRt(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
}

static std::string read_text_file(const std::string& path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        std::ostringstream oss;
        oss << "Could not open CUDA source file: " << path;
        fail("nvrtc_relu_bat_reduce_fused_mex:FileOpen", oss.str());
    }

    std::ostringstream buffer;
    buffer << ifs.rdbuf();

    if (!ifs.good() && !ifs.eof()) {
        std::ostringstream oss;
        oss << "Error reading CUDA source file: " << path;
        fail("nvrtc_relu_bat_reduce_fused_mex:FileRead", oss.str());
    }

    return buffer.str();
}

static std::string build_source_from_file(const std::string& kernel_path, int BK, int MS, int V) {
    std::ostringstream src;
    src << "#define BK " << BK << "\n";
    src << "#define MS " << MS << "\n";
    src << "#define V "  << V  << "\n";
    src << "\n";
    src << read_text_file(kernel_path);
    return src.str();
}

static std::string build_source_mixed_from_file(const std::string& kernel_path, int BK, int MS, int V, int R) {
    std::ostringstream src;
    src << "#define BK " << BK << "\n";
    src << "#define MS " << MS << "\n";
    src << "#define V "  << V  << "\n";
    src << "#define R "  << R  << "\n";
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

static ModuleEntry get_or_build_module(int BK, int MS, int V)
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
        fail("nvrtc_relu_bat_reduce_fused_mex:Context",
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

    const std::string src = build_source_from_file(kernelPath, BK, MS, V);

    nvrtcProgram prog;
    checkNvrtc(
        nvrtcCreateProgram(&prog, src.c_str(), kernelPath.c_str(), 0, nullptr, nullptr),
        "nvrtcCreateProgram");

    std::string gpu_arch = "--gpu-architecture=compute_" +
                           std::to_string(major) + std::to_string(minor);
    if (major >= 12) {
        gpu_arch = "--gpu-architecture=compute_90"; //Matlabs packaged cuda is too old to know sm120 ... force older arch here if running on new gpu
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
        fail("nvrtc_relu_bat_reduce_fused_mex:Compile", "NVRTC compilation failed.");
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
        cuModuleGetFunction(&func, module, "relu_bat_reduce_kernel_float4_sync"),
        "cuModuleGetFunction");

    ModuleEntry entry;
    entry.module = module;
    entry.func = func;
    g_cache.emplace(key, entry);
    return entry;
}

static std::string cuda_include_option() {
    const char* cuda_path = std::getenv("CUDA_HOME");
    if (!cuda_path) cuda_path = std::getenv("CUDA_PATH");

    if (!cuda_path) {
        fail("nvrtc_relu_bat_reduce_fused_mex:CUDAPath",
             "Set CUDA_HOME or CUDA_PATH so NVRTC can find cuda/std headers.");
    }

    return std::string("--include-path=") + cuda_path + "/include";
}

static ModuleEntry get_or_build_module_mixed(int BK, int MS, int V, int R)
{
    std::ostringstream keyss;
    keyss << kernelPath_mixed << "|BK=" << BK << "|MS=" << MS << "|V=" << V << "|R=" << R;
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
        fail("nvrtc_relu_bat_reduce_fused_mex:Context",
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

    const std::string src = build_source_mixed_from_file(kernelPath_mixed, BK, MS, V, R);

    nvrtcProgram prog;
    checkNvrtc(
        nvrtcCreateProgram(&prog, src.c_str(), kernelPath_mixed.c_str(), 0, nullptr, nullptr),
        "nvrtcCreateProgram");

    std::string gpu_arch = "--gpu-architecture=compute_" +
                           std::to_string(major) + std::to_string(minor);
    // std::string gpu_arch = "--gpu-architecture=compute_90"; //Matlabs packaged cuda is too old to know sm120 ... forcing older arch here if needed
    if (major >= 12) {
        gpu_arch = "--gpu-architecture=compute_90"; //Matlabs packaged cuda is too old to know sm120 ... force older arch here if running on new gpu
    }
    std::string cuda_inc = cuda_include_option();
    const char* opts[] = {
        "--std=c++17",
        "--use_fast_math",
        gpu_arch.c_str(),
        cuda_inc.c_str()
    };

    nvrtcResult compile_res = nvrtcCompileProgram(prog, 4, opts);

    size_t log_size = 0;
    checkNvrtc(nvrtcGetProgramLogSize(prog, &log_size), "nvrtcGetProgramLogSize");
    if (log_size > 1) {
        std::vector<char> log(log_size);
        checkNvrtc(nvrtcGetProgramLog(prog, log.data()), "nvrtcGetProgramLog");
        mexPrintf("%s\n", log.data());
    }

    if (compile_res != NVRTC_SUCCESS) {
        checkNvrtc(nvrtcDestroyProgram(&prog), "nvrtcDestroyProgram");
        fail("nvrtc_relu_bat_reduce_fused_mex:Compile", "NVRTC compilation failed.");
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
        cuModuleGetFunction(&func, module, "relu_bat_reduce_kernel_mixed_sync"),
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

    if (nrhs != 2) {
        fail("nvrtc_relu_bat_reduce_fused_mex:nrhs",
             "Usage: [sum, ssq, nnz] = nvrtc_relu_bat_reduce_fused_mex(At, Bt)");
    }

    if (nlhs != 3) {
        fail("nvrtc_relu_bat_reduce_fused_mex:nlhs",
             "Usage: [sum, ssq, nnz] = nvrtc_relu_bat_reduce_fused_mex(At, Bt)");
    }

    const mxGPUArray* At_gpu = mxGPUCreateFromMxArray(prhs[0]);
    const mxGPUArray* Bt_gpu = mxGPUCreateFromMxArray(prhs[1]);

    if (mxGPUGetClassID(At_gpu) != mxSINGLE_CLASS ||
        mxGPUGetClassID(Bt_gpu) != mxSINGLE_CLASS) {
        fail("nvrtc_relu_bat_reduce_fused_mex:type",
             "At and Bt must be gpuArray(single).");
    }

    if (mxGPUGetComplexity(At_gpu) != mxREAL ||
        mxGPUGetComplexity(Bt_gpu) != mxREAL) {
        fail("nvrtc_relu_bat_reduce_fused_mex:complexity",
             "At and Bt must be real gpuArray(single).");
    }

    if (mxGPUGetNumberOfDimensions(At_gpu) != 2 ||
        mxGPUGetNumberOfDimensions(Bt_gpu) != 2) {
        fail("nvrtc_relu_bat_reduce_fused_mex:ndims",
             "At and Bt must be 2-D arrays.");
    }

    const mwSize* At_dims = mxGPUGetDimensions(At_gpu); // D x N
    const mwSize* Bt_dims = mxGPUGetDimensions(Bt_gpu); // D x M

    if (At_dims[0] != Bt_dims[0]) {
        fail("nvrtc_relu_bat_reduce_fused_mex:shape",
             "Dimension mismatch: At and Bt must both be D x K arrays.");
    }

    if (At_dims[0] > static_cast<mwSize>(std::numeric_limits<int>::max()) ||
        At_dims[1] > static_cast<mwSize>(std::numeric_limits<int>::max()) ||
        Bt_dims[1] > static_cast<mwSize>(std::numeric_limits<int>::max())) {
        fail("nvrtc_relu_bat_reduce_fused_mex:shape",
             "D, N, and M must fit in int.");
    }

    const int D = static_cast<int>(At_dims[0]);
    const int N = static_cast<int>(At_dims[1]);
    const int M = static_cast<int>(Bt_dims[1]);

    const float* At_ptr =
        static_cast<const float*>(mxGPUGetDataReadOnly(At_gpu));
    const float* Bt_ptr =
        static_cast<const float*>(mxGPUGetDataReadOnly(Bt_gpu));

    mwSize scalar_dims[2] = {1, 1};

    mxGPUArray* sum_gpu = mxGPUCreateGPUArray(
        2, scalar_dims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    mxGPUArray* ssq_gpu = mxGPUCreateGPUArray(
        2, scalar_dims, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    mxGPUArray* nnz_gpu = mxGPUCreateGPUArray(
        2, scalar_dims, mxUINT64_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    double* out_sum_ptr =
        static_cast<double*>(mxGPUGetData(sum_gpu));

    double* out_ssq_ptr =
        static_cast<double*>(mxGPUGetData(ssq_gpu));

    unsigned long long* out_nnz_ptr =
        static_cast<unsigned long long*>(mxGPUGetData(nnz_gpu));

    checkCudaRt(cudaMemset(out_sum_ptr, 0, sizeof(double)),
                "cudaMemset(out_sum)");
    checkCudaRt(cudaMemset(out_ssq_ptr, 0, sizeof(double)),
                "cudaMemset(out_ssq)");
    checkCudaRt(cudaMemset(out_nnz_ptr, 0, sizeof(unsigned long long)),
                "cudaMemset(out_nnz)");

    if (D >= 128) {
        run_cublas_cub_relu_reduce_fallback(
            At_ptr, Bt_ptr,
            out_sum_ptr, out_ssq_ptr, out_nnz_ptr,
            D, N, M);
    } else {
        const int BK = 32;

        int MS = 1;
        if (D < 32) {
            MS = 4;
        } else if (D < 64) {
            MS = 2;
        } else {
            MS = 1;
        }

        const int V = D / 4;
        const int R = D % 4;
        const int threads = 128;

        ModuleEntry entry =
            (R == 0)
            ? get_or_build_module(BK, MS, V)
            : get_or_build_module_mixed(BK, MS, V, R);

        const float* K_A_ptr = Bt_ptr;  
        const float* K_B_ptr = At_ptr;  

        void* args[] = {
            reinterpret_cast<void*>(&K_A_ptr),
            reinterpret_cast<void*>(&K_B_ptr),
            reinterpret_cast<void*>(&out_sum_ptr),
            reinterpret_cast<void*>(&out_ssq_ptr),
            reinterpret_cast<void*>(&out_nnz_ptr),
            reinterpret_cast<void*>(const_cast<int*>(&M)),
            reinterpret_cast<void*>(const_cast<int*>(&N)),
            reinterpret_cast<void*>(const_cast<int*>(&D))
        };

        const int rows_per_block = MS * threads;
        const unsigned int grid_x =
            static_cast<unsigned int>((M + rows_per_block - 1) / rows_per_block);

        const unsigned int grid_y =
            static_cast<unsigned int>((N + BK - 1) / BK);

        const unsigned int shmem_bytes =
            static_cast<unsigned int>(
                2 * threads * sizeof(float) +
                threads * sizeof(unsigned long long));

        checkCudaDrv(
            cuLaunchKernel(entry.func,
                           grid_x, grid_y, 1,
                           threads, 1, 1,
                           shmem_bytes,
                           nullptr,
                           args,
                           nullptr),
            "cuLaunchKernel");

        checkCudaDrv(cuCtxSynchronize(), "cuCtxSynchronize");
    }

    plhs[0] = mxGPUCreateMxArrayOnGPU(sum_gpu);
    plhs[1] = mxGPUCreateMxArrayOnGPU(ssq_gpu);
    plhs[2] = mxGPUCreateMxArrayOnGPU(nnz_gpu);

    mxGPUDestroyGPUArray(sum_gpu);
    mxGPUDestroyGPUArray(ssq_gpu);
    mxGPUDestroyGPUArray(nnz_gpu);

    mxGPUDestroyGPUArray(At_gpu);
    mxGPUDestroyGPUArray(Bt_gpu);
}
