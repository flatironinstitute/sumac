%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A,B,stats] = salsa_loop(S,A,B,opts)

% START THE CLOCK
tLoop = tic;

% INITIALIZE
useGPU = canUseGPU();
T = transpose(S);
Scoo = sparse_coo_metadata(S,useGPU);
Tcoo = sparse_coo_metadata(T,useGPU);
clear T;
dA = zeros(size(A),"single");
dB = zeros(size(B),"single");
minCost = Inf;
cost = Inf;

% GPU?
if (useGPU)
  S = gpuArray(S);
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
  Spart = partition_coo_blocks(Scoo,permuteA,batchIdxA,useGPU);
  Tpart = partition_coo_blocks(Tcoo,permuteB,batchIdxB,useGPU);

  % MINIBATCH UPDATES
  for mb=1:opts.nbatch
    stepnum = mb + (iter-1)*opts.nbatch;
    rowsA = permuteA(batchIdxA{mb});
    rowsB = permuteB(batchIdxB{mb});
    SrMeta = block_metadata(Spart,mb);
    TrMeta = block_metadata(Tpart,mb);
    [B,dB] = batch_update(SrMeta,A(rowsA,:),B,dB,opts,stepnum);
    [A,dA] = batch_update(TrMeta,B(rowsB,:),A,dA,opts,stepnum);
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
% CACHE FULL SPARSE MATRIX TRIPLETS

function Scoo = sparse_coo_metadata(S,useGPU)

[m,n] = size(S);
[i,j,v] = find(S);

if (useGPU)
  i = gpuArray(i);
  j = gpuArray(j);
  v = gpuArray(v);
end

Scoo = struct('i',i,'j',j,'v',v,'m',m,'n',n);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARTITION ALL NONZEROS INTO CURRENT RANDOM ROW BLOCKS

function part = partition_coo_blocks(Scoo,permuteRows,batchIdx,useGPU)

nbatch = length(batchIdx);
blockOfRow = zeros(Scoo.m,1);
localOfRow = zeros(Scoo.m,1);
blockRows = zeros(nbatch,1);

for b=1:nbatch
  rows = permuteRows(batchIdx{b});
  nrows = length(rows);
  blockOfRow(rows) = b;
  localOfRow(rows) = 1:nrows;
  blockRows(b) = nrows;
end

if (useGPU)
  blockOfRow = gpuArray(blockOfRow);
  localOfRow = gpuArray(localOfRow);
end

edgeBlock = blockOfRow(Scoo.i);
edgeLocal = localOfRow(Scoo.i);
[edgeBlockSorted,order] = sort(edgeBlock);

part.i = edgeLocal(order);
part.j = Scoo.j(order);
part.Sij = Scoo.v(order);
part.m = blockRows;
part.n = Scoo.n;
part.offset = block_offsets(edgeBlockSorted,nbatch,useGPU);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTE BLOCK OFFSETS FROM SORTED BLOCK IDS

function offset = block_offsets(edgeBlockSorted,nbatch,useGPU)

counts = zeros(nbatch,1);
nedge = length(edgeBlockSorted);

if (nedge > 0)
  changeIdx = find(diff(edgeBlockSorted) ~= 0);
  if (useGPU)
    changeIdx = gather(changeIdx);
  end
  changeIdx = changeIdx(:);

  blockStarts = [1; changeIdx+1];
  blockEnds = [changeIdx; nedge];
  blocks = edgeBlockSorted(blockStarts);
  if (useGPU)
    blocks = gather(blocks);
  end
  blocks = blocks(:);

  counts(blocks) = blockEnds-blockStarts+1;
end

offset = [1; cumsum(counts)+1];

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SELECT LOCAL TRIPLETS FOR ONE MINIBATCH

function Smeta = block_metadata(part,mb)

first = part.offset(mb);
last = part.offset(mb+1)-1;
if (first <= last)
  idx = (first:last)';
else
  idx = zeros(0,1);
end

Smeta = struct('i',part.i(idx),'j',part.j(idx),'Sij',part.Sij(idx),...
  'm',part.m(mb),'n',part.n);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STOCHASTIC ALTERNATING LEAST SQUARES ALGORITHM MINIBATCH UPDATE

function [B,dB] = batch_update(Smeta,A,B,dB,opts,stepnum)

% COMPUTE UPDATE
lsqB = lsq_update(Smeta,A,B);
dB = (lsqB-B)*(1-opts.momentum) + dB*opts.momentum;

% UNBIAS FOR EARLY STEPS
unbias = 1-opts.momentum^stepnum;
B = B + opts.lrate*dB/unbias;

end


function w = edge_weight_arrayfun(i,j,Sij,A,B)

d = size(A,2);
w = arrayfun(@edge_weight,i,j,Sij);

  function wij = edge_weight(ii,jj,sij)
    lij = sij - sij;  % zero with same type as sij

    for k = 1:d
      lij = lij + A(ii,k) * B(jj,k);
    end

    % sij - lij + max(0,lij) == sij + max(0,-lij)
    if (lij < 0)
      wij = sij - lij;
    else
      wij = sij;
    end
  end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LEAST SQUARES UPDATE
%
% Computes the least squares update for B given the minibatch of A.

function lsqB = lsq_update(Smeta,A,B)

% CONTRIBUTION FROM ELEMENTS WITH S=0
%pseudoInverseAt = A/(A'*A);

G = A' * A;                         
Gcpu = gather(G);                    
Ginv = inv(Gcpu); % solve small inv on CPU due to Matlab's GPU path slowness.
pseudoInverseAt = A * gpuArray(Ginv);

i = Smeta.i;
j = Smeta.j;
Sij = Smeta.Sij;
m = Smeta.m;
n = Smeta.n;

if isgpuarray(A)

  Yt = relu_bat_c_fused_nvrtc_mex(A.', B.', pseudoInverseAt.');
  % baseline
  % Lij = sum(A(i,:).*B(j,:), 2);
  % Mij = max(0,Lij);
  % Ct = sparse(j,i,Sij-Lij+Mij,n,m);
  % stepC = Ct*pseudoInverseAt;

  %arrayfun version
  w = edge_weight_arrayfun(i,j,Sij,A,B);
  Ct = sparse(j,i,w,n,m);
  stepC = Ct*pseudoInverseAt;

  % UPDATE
  lsqB = B - Yt.' + stepC; % - Yt.' instead of +stepM - the kernel needs row major but matlab's layout is column major
else 
  Mt = max(0,B*A');       
  stepM = -Mt*pseudoInverseAt;

  % CONTRIBUTION AND CORRECTION FROM ELEMENTS WITH S>0
  Lij = sum(A(i,:).*B(j,:), 2);
  Mij = max(0,Lij);
  Ct = sparse(j,i,Sij-Lij+Mij,n,m);
  stepC = Ct*pseudoInverseAt;

  % UPDATE
  lsqB = B + stepM + stepC; 
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
