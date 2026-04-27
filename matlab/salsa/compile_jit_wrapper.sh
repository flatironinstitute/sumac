#!/bin/bash
#Script to compile jit wrapper on FI clusters
module load matlab cuda/12.8.0

matlab -batch "setenv('NVCC_APPEND_FLAGS','-allow-unsupported-compiler'); mexcuda('-lnvrtc','-lcuda','relu_bat_c_fused_nvrtc_mex.cu')"