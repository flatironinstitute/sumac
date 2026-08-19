%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DATASETS FOR THIS CODE ARE AVAILABLE AT
%   https://users.flatironinstitute.org/~lsaul/sparse_matrices
%
% SUGGESTED HYPERPARAMETERS:
%
%     dataset       rank        step-size        mini-batches
%      DIGITS        >=4                1                 100
%  CONNECTOME          4                1                  25
%  CONNECTOME          8                1                  50
%  CONNECTOME       >=16                1                 100
%     BIGRAMS          4             0.01                  50
%     BIGRAMS          8              0.1                  50
%     BIGRAMS       >=16              0.1                 100
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% SETUP
format short g;
format compact;

% PATH
parent_dir = fileparts(fileparts(mfilename('fullpath')));
addpath([parent_dir '/sumac']);

% OPTIONS
opts = struct(...
   'use_gpu',true,... % HARDWARE
   'use_cuda_mex',true,...
   'use_tf32',true,...
   'conserve_memory',false,...
   'block_size_MB',250,...
   'seed',0,...              % INITIALIZATION AND CONVERGENCE
   'max_iterate',10,...
   'time_limit',3600,...
   'tol_abs',1e-2,...
   'step_size',1,...         % OPTIMIZATION
   'momentum',0.9,...
   'nbatch',100,...
   'stats_interval',2);

% RANK
r = 16;

% CONNECTOME
S = load("connectome_139K.mat").("connectome"); 

% DIGITS
% S = load("digits_knn_70K.mat").("S");

% BIGRAMS
% S = load("bigrams_250K.mat").("bigrams");
% S = S./(sum(S,2)+realmin);
% opts.step_size = 0.1;
 
% SUMAC WITH RANK r
[A,B,stats,opts] = sumac(S,r,opts);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
