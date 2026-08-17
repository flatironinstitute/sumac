%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% function [A,B,stats,opts] = sumac(S,r,opts)
%
% Attempts to solve the subzero matrix completion problem S = max(0,A*B').
%
% ****** Input ******
%   S : m-by-n matrix, sparse and nonnegative
%   r : rank of matrix decomposition
%   opts: user options
%       **** Hardware:
%       .use_gpu (default: canUseGPU())
%       .use_cuda_mex (default: false)
%       .use_tf32 (default: false)
%       .conserve_memory (default: false)
%       .block_size_MB (default: 250)
%       **** Initialization:
%       .init_file: name of matfile with user-initialized A,B
%       .seed = random seed (default: random)
%       **** Convergence:
%       .max_iterate = maximum number of iterations (default: 1000) 
%       .time_limit = time limit in seconds (default: Inf)
%       .tol_abs = absolute tolerance for convergence (default: 1e-2)
%       **** Optimization:
%       .step_size (default: 1)
%       .momentum (default: 0.9)
%       .nbatch (default: 100)     
%       .stats_interval = how often to evaluate cost (default: 5)
% 
% ****** Output ******
%   A,B : m-by-r and n-by-r matrices
% stats : stats.rmse = normalized root-mean-squared error
%         stats.jacc = weighted jaccard distance
%         stats.cost = auxiliary loss function
%         stats.tsec = wall-clock time in seconds
%  opts : options
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A,B,stats,opts] = sumac(S,r,opts)

% START THE CLOCK
tStart = tic;

% VERIFY THAT S IS SPARSE AND NONNEGATIVE
if (~issparse(S) || min(nonzeros(S))<0)
  err_msg = 'the input matrix should be sparse and nonnegative.';
  error(['   ' mfilename ': ' err_msg]);
end

% OPTIONS
opts = set_options(opts);
display_header(S,r,opts);

% INITIALIZE
S = single(S);
fprintf(1,'\nInitializing ...\n');
[A,B] = salsa_init(S,r,opts);

% EXTRACT NON-ZERO ROWS AND COLUMNS
rows = any(S,2);
cols = any(S,1);
Snz = S(rows,cols);

% STOCHASTIC ALTERNATING LEAST SQUARES
fprintf(1,'\nUpdating ...\n');
[A(rows,:),B(cols,:),stats] = salsa_loop(Snz,A(rows,:),B(cols,:),opts);
[A,B] = gather(A,B);

% DONE
fprintf(1,'\n');
toc(tStart);


end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function opts = set_options(user_opts)

% DEFAULT OPTIONS
opts_default = ...
  struct('use_gpu',canUseGPU(),'use_cuda_mex',false,'use_tf32',false,...
    'conserve_memory',false,'block_size_MB',250,'seed',randi(intmax),...
    'max_iterate',1000,'time_limit',Inf,'tol_abs',1e-2,...
    'stats_interval',5,'momentum',0.9,'nbatch',100,'step_size',1);

% NO USER INPUT?
if (isempty(user_opts))
  opts = opts_default;
  return;
end

% DEFAULT FOR NON-SPECIFIED OPTIONS
opts = user_opts;
fields = fieldnames(opts_default);
for f=1:length(fields)
  if (~isfield(opts,fields{f}))
    opts.(fields{f}) = opts_default.(fields{f});
  end
end

% CONFIRM GPU
opts.use_gpu = opts.use_gpu && canUseGPU();
opts.use_cuda_mex = opts.use_cuda_mex && opts.use_gpu; 
opts.use_tf32 = opts.use_tf32 && opts.use_cuda_mex; 

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function display_header(S,r,opts)

% PROBLEM SIZE
[m,n] = size(S);
fprintf(1,'\n  Input to sumac is %d x %d matrix with %d nonzeros.\n',...
  m,n,nnz(S));
fprintf(1,'  Attempting to complete with rank %d matrix.\n',r);  

% GPU?
if (opts.use_gpu)
  gpu_name = gpuDevice().Name;
  gpu_memory = gpuDevice().AvailableMemory/1e9;
  fprintf(1,'\n  GPU is available: %s with %f GB memory.\n\n',...
    gpu_name,gpu_memory);
 end

% OPTIONS
fprintf(1,'  Options:\n');
disp(opts);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
