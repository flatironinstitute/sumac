%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A,B,stats] = salsa_loop(S,A,B,opts)

% CUDA?
if (opts.use_cuda_mex)
  [A,B,stats] = salsa_loop_cuda(S,A,B,opts);
  return;
end

% START THE CLOCK
tLoop = tic;

% INITIALIZE
[m,n] = size(S);
T = transpose(S);
dA = zeros(size(A),"single");
dB = zeros(size(B),"single");
minCost = Inf;
cost = Inf;

% GPU?
if (opts.use_gpu)
  S = gpuArray(S);
  T = gpuArray(T);
  A = gpuArray(A);
  B = gpuArray(B);
  dA = gpuArray(dA);
  dB = gpuArray(dB);
end

% MINIBATCH INDICES
mbIdxA = mb_split(size(A,1),opts.nbatch);
mbIdxB = mb_split(size(B,1),opts.nbatch);

% LOOP
for iter=1:opts.max_iterate

  % ROW/COLUMN PERMUTATIONS
  permuteA = randperm(m);
  permuteB = randperm(n);

  % MINIBATCH UPDATES
  for mb=1:opts.nbatch
    stepnum = mb + (iter-1)*opts.nbatch;
    rowsA = permuteA(mbIdxA{mb});
    rowsB = permuteB(mbIdxB{mb});
    S_mb = sparse_slice(S,T,rowsA);
    T_mb = sparse_slice(T,S,rowsB);
    [B,dB] = mb_update(S_mb,A(rowsA,:),B,dB,opts,stepnum);
    [A,dA] = mb_update(T_mb,B(rowsB,:),A,dA,opts,stepnum);
  end

  % REPORT
  tsec = toc(tLoop);
  if ((mod(iter,opts.stats_interval)==0) || (iter==opts.max_iterate) || (tsec>opts.time_limit))
    [cost,rmse,jacc,fnnz] = compute_errors(S,A,B,opts);
    print_stats(iter,cost,rmse,jacc,fnnz,tsec,cost<minCost);
  end

  % BEST
  if (cost<minCost)
    minCost = cost;
    stats.cost = cost;
    stats.rmse = rmse;
    stats.jacc = jacc;
    bestA = A;
    bestB = B;
  end

  % DONE?
  if ((tsec>opts.time_limit) || (cost<opts.tol_abs))
    break;
  end
end

% WRAP UP
stats.tsec = tsec;
[A,B] = refactor(bestA,bestB);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SPLIT nr ROWS INTO nb MINIBATCHES

function mbIdx = mb_split(nr,nb)

edge = round(linspace(0,nr,nb+1));
rowI = 1+edge(1:nb);
rowF = edge(2:end);

mbIdx = cell(nb,1);
for b=1:nb
  mbIdx{b} = rowI(b):rowF(b);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SLICE ROWS OF SPARSE MATRIX

function Sr = sparse_slice(S,T,rows)

if (isgpuarray(S))
  % SPARSE GPU ARRAYS CANNOT BE SLICED
  m = size(S,1);
  r = length(rows);
  Sr = sparse(1:r,rows,1,r,m)*S;
else
  % FASTER THAN S(rowsA,:)
  Sr = T(:,rows)';                
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MINIBATCH UPDATE

function [B,dB] = mb_update(S,A,B,dB,opts,stepnum)

% COMPUTE UPDATE
lsqB = lsq_update(S,A,B,opts);
dB = (lsqB-B)*(1-opts.momentum) + dB*opts.momentum;

% UNBIAS FOR EARLY STEPS
unbias = 1-opts.momentum^stepnum;
B = B + opts.step_size*dB/unbias;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LEAST SQUARES UPDATE
%
% Computes the least squares update for B given the minibatch of A.

function lsqB = lsq_update(S,A,B,opts)

% CONTRIBUTION FROM ELEMENTS WITH S=0
pseudoInverseAt = A/(A'*A);
Mt = max(0,B*A');       
stepM = -Mt*pseudoInverseAt;

% CONTRIBUTION AND CORRECTION FROM ELEMENTS WITH S>0
[m,n] = size(S);
[i,j,Sij] = find(S);
Lij = masked_low_rank_product(i,j,A,B,opts);
Ct = sparse(j,i,Sij-min(0,Lij),n,m);
stepC = Ct*pseudoInverseAt;

% UPDATE
lsqB = B + stepM + stepC;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%