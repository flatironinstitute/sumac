%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A,B,stats] = salsa_loop(S,A,B,opts)

% START THE CLOCK
tLoop = tic;

% INITIALIZE
T = transpose(S);
dA = zeros(size(A),"single");
dB = zeros(size(B),"single");
minCost = Inf;
cost = Inf;

% GPU?
if (canUseGPU())
  S = gpuArray(S);
  T = gpuArray(T);
  A = gpuArray(A);
  B = gpuArray(B);
  dA = gpuArray(dA);
  dB = gpuArray(dB);
end

% MINIBATCH INDICES
batchIdxA = batch_split(size(A,1),opts.nbatch);
batchIdxB = batch_split(size(B,1),opts.nbatch);

% LOOP
for iter=1:opts.max_iterate

  % PERMUTE
  permuteA = randperm(size(A,1));
  permuteB = randperm(size(B,1));

  % MINIBATCH UPDATES
  for mb=1:opts.nbatch
    stepnum = mb + (iter-1)*opts.nbatch;
    rowsA = permuteA(batchIdxA{mb});
    rowsB = permuteB(batchIdxB{mb});
    Sr = sparse_slice(S,T,rowsA);
    Tr = sparse_slice(T,S,rowsB);
    [B,dB] = batch_update(Sr,A(rowsA,:),B,dB,opts,stepnum);
    [A,dA] = batch_update(Tr,B(rowsB,:),A,dA,opts,stepnum);
  end

  % REPORT
  tsec = toc(tLoop);
  if ((mod(iter,opts.stats_interval)==0) || (iter==opts.max_iterate) || (tsec>opts.time_limit))
    [cost,rmse,jacc,fnnz] = compute_errors(S,A,B,opts);
    print_stats(iter,cost,rmse,jacc,fnnz,tsec);
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
% SPLIT INTO BATCHES

function batchIdx = batch_split(nrows,nbatch)

row_boundary = round(linspace(0,nrows,nbatch+1));

batchIdx = cell(nbatch,1);
for b=1:nbatch
  batchIdx{b} = (1+row_boundary(b)):row_boundary(b+1);
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
% STOCHASTIC ALTERNATING LEAST SQUARES ALGORITHM MINIBATCH UPDATE

function [B,dB] = batch_update(S,A,B,dB,opts,stepnum)

% COMPUTE UPDATE
lsqB = lsq_update(S,A,B);
dB = (lsqB-B)*(1-opts.momentum) + dB*opts.momentum;

% UNBIAS FOR EARLY STEPS
unbias = 1-opts.momentum^stepnum;
B = B + opts.lrate*dB/unbias;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LEAST SQUARES UPDATE
%
% Computes the least squares update for B given the minibatch of A.

function lsqB = lsq_update(S,A,B)

% CONTRIBUTION FROM ELEMENTS WITH S=0
pseudoInverseAt = A/(A'*A);
%Mt = max(0,B*A');       
%stepM = -Mt*pseudoInverseAt;

Yt = relu_bat_c_fused_nvrtc_mex(A.', B.', pseudoInverseAt.');

% CONTRIBUTION AND CORRECTION FROM ELEMENTS WITH S>0
[m,n] = size(S);
[i,j,Sij] = find(S);
% Lij = sum(A(i,:).*B(j,:),2);
% Lij = zeros(size(Sij));
% for k=1:size(A,2)
%   Lij = Lij + A(i,k).*B(j,k);  % LOOP TO SAVE MEMORY
% end
Lij = sum(A(i,:).*B(j,:), 2);
Mij = max(0,Lij);
Ct = sparse(j,i,Sij-Lij+Mij,n,m);
stepC = Ct*pseudoInverseAt;

% UPDATE
lsqB = B - Yt.' + stepC; % - Yt.' instead of +stepM - the kernel needs row major but matlab's layout is column major

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
