%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% function [A,B,opts] = sumac(S,d,opts)
%
% Attempts to solve the subzero matrix completion problem S = max(0,A*B').
%
% ****** Input ******
%   S : m-by-n matrix, sparse and nonnegative
%   d : rank of matrix decomposition
%   opts: user options
%       .max_iterate = maximum number of iterations (default: 1000) 
%       .time_limit = time limit in seconds (default: Inf)
%       .tol_abs = absolute tolerance for convergence (default: 1e-2)
%       .seed = random seed (default: random)
%       .momentum (default: 0.9)
%       .block_size_MB (default: 250)
%       .stats_interval (default: 5)
%       .initfile = filename (*.mat) with user-initialized A,B
%       .nbatch (default: 100)
%       .lrate (default: 1)
% 
% ****** Output ******
%   A,B    : m-by-d and n-by-d matrices
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A,B,stats,opts] = sumac(S,d,opts)

% START THE CLOCK
tStart = tic;

% VERIFY THAT S IS SPARSE AND NONNEGATIVE
if (~issparse(S) || min(nonzeros(S))<0)
  err_msg = 'the input matrix should be sparse and nonnegative.';
  error(['   ' mfilename ': ' err_msg]);
end

% OPTIONS
opts = set_options(opts);
display_header(S,d,opts);

% INITIALIZE
S = single(S);
[A,B] = salsa_init(S,d,opts);

% EXTRACT NON-EMPTY ROWS AND COLUMNS
r = any(S,2);
c = any(S,1);

% STOCHASTIC ALTERNATING LEAST SQUARES
[A(r,:),B(c,:),stats] = salsa_loop(S(r,c),A(r,:),B(c,:),opts);
[A,B] = gather(A,B);

% TOTAL ELAPSED TIME
fprintf(1,'\n');
toc(tStart);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function opts = set_options(user_opts)

% DEFAULT OPTIONS
opts_default = ...
  struct('max_iterate',1000,'time_limit',Inf,'tol_abs',1e-2,...
    'seed',randi(intmax),'stats_interval',5,...
    'momentum',0.9,'block_size_MB',250,'nbatch',100,'lrate',1);

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

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function display_header(S,d,opts)

% PROBLEM SIZE
[m,n] = size(S);
stack = dbstack();
fprintf(1,'\n  Input to %s is %d x %d matrix with %d nonzeros.\n',...
  upper(stack(2).name),m,n,nnz(S));
fprintf(1,'  Attempting to complete with rank %d matrix.\n',d);  

% GPU?
if (canUseGPU())
  gpu_name = gpuDevice().Name;
  gpu_memory = gpuDevice().AvailableMemory/1e9;
  fprintf(1,'\n  GPU is available: %s with %f GB memory.\n\n',...
    gpu_name,gpu_memory);
end

% OPTIONS
fprintf(1,'  Options:\n');
disp(opts);
fprintf(1,'\n');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
