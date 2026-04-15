#!/bin/bash
#Script to compile jit wrapper on FI clusters
module load matlab cuda

matlab -batch "mexcuda('-lnvrtc','-lcuda','relu_bat_c_fused_nvrtc_mex.cu')"