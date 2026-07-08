#define TF32_MEX_ID "relu_bat_c_sparse_tf32_nvrtc_mex"
#include "relu_bat_c_tf32_nvrtc_common.hpp"

using namespace relu_bat_c_tf32_nvrtc;

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
    mxInitGPU();

    if (nrhs != 6 && nrhs != 8) {
        fail(
            "nrhs",
            "Usage: Yt = relu_bat_c_sparse_tf32_nvrtc_mex(At, Bt, Ct, rowPtr, edgeI, edgeVal[, rowPtrBase, edgeBase])");
    }
    if (nlhs > 1) {
        fail("nlhs", "One output expected.");
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
        fail("type", "At, Bt, Ct, and edgeVal must be gpuArray(single).");
    }
    if (mxGPUGetClassID(row_ptr_gpu) != mxINT64_CLASS ||
        mxGPUGetClassID(edge_i_gpu) != mxINT64_CLASS) {
        fail("type", "rowPtr and edgeI must be gpuArray(int64).");
    }
    if (mxGPUGetComplexity(At_gpu) != mxREAL ||
        mxGPUGetComplexity(Bt_gpu) != mxREAL ||
        mxGPUGetComplexity(Ct_gpu) != mxREAL ||
        mxGPUGetComplexity(row_ptr_gpu) != mxREAL ||
        mxGPUGetComplexity(edge_i_gpu) != mxREAL ||
        mxGPUGetComplexity(edge_val_gpu) != mxREAL) {
        fail("complexity", "All inputs must be real gpuArrays.");
    }
    if (mxGPUGetNumberOfDimensions(At_gpu) != 2 ||
        mxGPUGetNumberOfDimensions(Bt_gpu) != 2 ||
        mxGPUGetNumberOfDimensions(Ct_gpu) != 2) {
        fail("ndims", "At, Bt, and Ct must be 2-D arrays.");
    }

    const mwSize* At_dims = mxGPUGetDimensions(At_gpu); // D x N
    const mwSize* Bt_dims = mxGPUGetDimensions(Bt_gpu); // D x M
    const mwSize* Ct_dims = mxGPUGetDimensions(Ct_gpu); // D x N

    const int D = checked_int_dim(At_dims[0], "D");
    const int N = checked_int_dim(At_dims[1], "N");
    const int M = checked_int_dim(Bt_dims[1], "M");

    if (D < 1) {
        fail("shape", "D must be positive.");
    }
    if (Bt_dims[0] != At_dims[0] ||
        Ct_dims[0] != At_dims[0] ||
        Ct_dims[1] != At_dims[1]) {
        fail("shape", "Dimension mismatch among At, Bt, and Ct.");
    }

    const mwSize n_row_ptr = mxGPUGetNumberOfElements(row_ptr_gpu);
    const mwSize n_edge_i = mxGPUGetNumberOfElements(edge_i_gpu);
    const mwSize n_edge_val = mxGPUGetNumberOfElements(edge_val_gpu);
    const mwSize row_ptr_offset =
        (nrhs == 8) ? scalar_offset(prhs[6], "rowPtrBase") : 0;
    const mwSize edge_offset =
        (nrhs == 8) ? scalar_offset(prhs[7], "edgeBase") : 0;

    if (nrhs == 6 && n_row_ptr != static_cast<mwSize>(M + 1)) {
        fail("shape", "rowPtr must have length size(Bt,2)+1.");
    }
    if (nrhs == 8 &&
        row_ptr_offset + static_cast<mwSize>(M + 1) > n_row_ptr) {
        fail("shape", "rowPtrBase plus size(Bt,2)+1 exceeds rowPtr length.");
    }
    if (n_edge_i != n_edge_val) {
        fail("shape", "edgeI and edgeVal must have the same number of elements.");
    }
    if (edge_offset > n_edge_i) {
        fail("shape", "edgeBase exceeds edgeI length.");
    }

    mwSize out_dims[2] = {
        static_cast<mwSize>(D),
        static_cast<mwSize>(M),
    };
    mxGPUArray* Yt_gpu = mxGPUCreateGPUArray(
        2, out_dims, mxSINGLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    const float* At_ptr = static_cast<const float*>(mxGPUGetDataReadOnly(At_gpu));
    const float* Bt_ptr = static_cast<const float*>(mxGPUGetDataReadOnly(Bt_gpu));
    const float* Ct_ptr = static_cast<const float*>(mxGPUGetDataReadOnly(Ct_gpu));
    const long long* row_ptr_data =
        static_cast<const long long*>(mxGPUGetDataReadOnly(row_ptr_gpu));
    const long long* edge_i_data =
        static_cast<const long long*>(mxGPUGetDataReadOnly(edge_i_gpu));
    const float* edge_val_data =
        static_cast<const float*>(mxGPUGetDataReadOnly(edge_val_gpu));

    const long long* row_ptr = row_ptr_data + row_ptr_offset;
    const long long* edge_i = edge_i_data + edge_offset;
    const float* edge_val = edge_val_data + edge_offset;
    const size_t edge_capacity =
        static_cast<size_t>(n_edge_i - edge_offset);
    float* edge_score_ptr = ensure_edge_score_buffer(edge_capacity);

    float* Yt_ptr = static_cast<float*>(mxGPUGetData(Yt_gpu));

    const LaunchPlan plan = make_launch_plan(D);
    DeviceBuffer Y_pad;
    float* dense_y_ptr = Yt_ptr;
    if (plan.output_ld != D) {
        const size_t bytes =
            static_cast<size_t>(M) * static_cast<size_t>(plan.output_ld) * sizeof(float);
        Y_pad.allocate(bytes, "cudaMalloc(Y_pad)");
        dense_y_ptr = Y_pad.as<float>();
    }

    launch_dense(plan, At_ptr, Bt_ptr, Ct_ptr, dense_y_ptr, N, M, D);
    subtract_sparse_correction(
        At_ptr,
        Bt_ptr,
        Ct_ptr,
        row_ptr,
        edge_i,
        edge_val,
        edge_score_ptr,
        edge_capacity,
        dense_y_ptr,
        D,
        M,
        plan.output_ld);
    compact_padded_y(dense_y_ptr, Yt_ptr, D, M, plan.output_ld);
    check_cuda_rt(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    plhs[0] = mxGPUCreateMxArrayOnGPU(Yt_gpu);

    mxGPUDestroyGPUArray(At_gpu);
    mxGPUDestroyGPUArray(Bt_gpu);
    mxGPUDestroyGPUArray(Ct_gpu);
    mxGPUDestroyGPUArray(row_ptr_gpu);
    mxGPUDestroyGPUArray(edge_i_gpu);
    mxGPUDestroyGPUArray(edge_val_gpu);
    mxGPUDestroyGPUArray(Yt_gpu);
}
