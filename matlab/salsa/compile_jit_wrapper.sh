#!/bin/bash
#Script to compile jit wrapper on FI clusters
module load cuda gcc/11.5.0 matlab

matlab -batch "setenv('NVCC_APPEND_FLAGS','-allow-unsupported-compiler'); mexcuda('-lnvrtc','-lcuda','-lcublas','relu_bat_c_fused_nvrtc_mex.cu')"
