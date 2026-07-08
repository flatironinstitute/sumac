%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A,B,stats] = salsa_loop(S,A,B,opts)

% START THE CLOCK
tLoop = tic;

% INITIALIZE
if (~canUseGPU())
  error('salsa_loop:GPURequired',...
    'salsa_loop requires a GPU.');
end
T = transpose(S);
Scoo = sparse_coo_metadata(S);
Tcoo = sparse_coo_metadata(T);
clear T;
dA = zeros(size(A),"single");
dB = zeros(size(B),"single");
minCost = Inf;
cost = Inf;

S = gpuArray(S);
A = gpuArray(A);
B = gpuArray(B);
dA = gpuArray(dA);
dB = gpuArray(dB);

% MINIBATCH INDICES
batchIdxA = batch_split(size(A,1),opts.nbatch);
batchIdxB = batch_split(size(B,1),opts.nbatch);
batchMetaA = batch_position_metadata(size(A,1),batchIdxA);
batchMetaB = batch_position_metadata(size(B,1),batchIdxB);

% LOOP
for iter=1:opts.max_iterate

  % PERMUTE
  permuteA = randperm(size(A,1));
  permuteB = randperm(size(B,1));
  Spart = partition_coo_blocks(Scoo,permuteA,batchMetaA);
  Tpart = partition_coo_blocks(Tcoo,permuteB,batchMetaB);

  % MINIBATCH UPDATES
  for mb=1:opts.nbatch
    stepnum = mb + (iter-1)*opts.nbatch;
    rowsA = permuteA(batchIdxA{mb});
    rowsB = permuteB(batchIdxB{mb});
    SrMeta = Spart{mb};
    TrMeta = Tpart{mb};
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
% PRECOMPUTE POSITION-TO-BLOCK MAPS FOR A FIXED BATCH SPLIT

function meta = batch_position_metadata(nrows,batchIdx)

nbatch = length(batchIdx);
blockRows = zeros(nbatch,1);
blockOfPos = zeros(nrows,1);
localOfPos = zeros(nrows,1);

for b=1:nbatch
  idx = batchIdx{b};
  nblockRows = length(idx);
  blockRows(b) = nblockRows;
  blockOfPos(idx) = b;
  localOfPos(idx) = 1:nblockRows;
end

blockOfPos = gpuArray(blockOfPos);
localOfPos = gpuArray(localOfPos);

meta = struct('blockOfPos',blockOfPos,'localOfPos',localOfPos,...
  'blockRows',blockRows,'nbatch',nbatch);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CACHE FULL SPARSE MATRIX TRIPLETS

function Scoo = sparse_coo_metadata(S)

[m,n] = size(S);
[i,j,v] = find(S);

i = gpuArray(i);
j = gpuArray(j);
v = gpuArray(v);

Scoo = struct('i',i,'j',j,'v',v,'m',m,'n',n);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PARTITION ALL NONZEROS INTO CURRENT RANDOM ROW BLOCKS
%
% Builds CSR metadata for the fused GPU kernel.  Within
% each minibatch, nonzeros are grouped by the row of the factor being
% updated (j) and sorted by local fixed-factor row (i).

function part = partition_coo_blocks(Scoo,permuteRows,batchMeta)

nbatch = batchMeta.nbatch;
permuteRows = gpuArray(permuteRows(:));
blockRows = batchMeta.blockRows;
blockOfRow = zeros(Scoo.m,1,'gpuArray');
localOfRow = zeros(Scoo.m,1,'gpuArray');
blockOfRow(permuteRows) = batchMeta.blockOfPos;
localOfRow(permuteRows) = batchMeta.localOfPos;

edgeBlock = blockOfRow(Scoo.i);
edgeLocal = localOfRow(Scoo.i);

rowBase = double(Scoo.m) + 1;
colBase = rowBase * double(Scoo.n);
sortKey = double(edgeLocal) + rowBase * (double(Scoo.j)-1) + colBase * (double(edgeBlock)-1);
[~,order] = sort(sortKey);

edgeBlockSorted = edgeBlock(order);
edgeLocalSorted = edgeLocal(order);
edgeJSorted = Scoo.j(order);
edgeValSorted = Scoo.v(order);

[rowPtr,edgeI,edgeVal,rowPtrBase,edgeBase] = sparse_kernel_metadata_gpu(edgeBlockSorted, edgeJSorted, edgeLocalSorted, edgeValSorted,nbatch,Scoo.n);

part = cell(nbatch,1);
for mb=1:nbatch
  part{mb} = struct('m',blockRows(mb),'n',Scoo.n,'rowPtr',rowPtr, 'edgeI',edgeI,'edgeVal',edgeVal,'rowPtrBase',rowPtrBase(mb), 'edgeBase',edgeBase(mb));
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% BUILD CSR METADATA FOR THE FUSED KERNEL ON THE GPU

function [rowPtr,edgeI,edgeVal,rowPtrBase,edgeBase] = sparse_kernel_metadata_gpu(edgeBlock,edgeJ,edgeLocal,edgeVal,nbatch,n)

nedge = length(edgeBlock);

blockCounts = zeros(nbatch,1,'gpuArray');
rowCounts = zeros(nbatch*n,1,'gpuArray');

if (nedge > 0)
  blockChangeIdx = find(diff(edgeBlock) ~= 0);
  blockStarts = [gpuArray(1); blockChangeIdx+1];
  blockEnds = [blockChangeIdx; gpuArray(nedge)];
  blocks = edgeBlock(blockStarts);
  blockCounts(blocks) = blockEnds-blockStarts+1;

  linearRows = edgeJ + (edgeBlock-1)*n;
  rowChangeIdx = find(diff(linearRows) ~= 0);
  rowStarts = [gpuArray(1); rowChangeIdx+1];
  rowEnds = [rowChangeIdx; gpuArray(nedge)];
  rows = linearRows(rowStarts);
  rowCounts(rows) = rowEnds-rowStarts+1;
end

rowCounts = reshape(rowCounts,n,nbatch);
rowPtrByBlock = [zeros(1,nbatch,'gpuArray'); cumsum(rowCounts,1)];
rowPtr = int64(rowPtrByBlock(:));
edgeI = int64(edgeLocal - 1);
edgeVal = single(edgeVal);
rowPtrBase = int64((0:(nbatch-1))' * (n+1));
edgeBase = int64(gather([gpuArray(0); cumsum(blockCounts(1:end-1))]));

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% STOCHASTIC ALTERNATING LEAST SQUARES ALGORITHM MINIBATCH UPDATE

function [B,dB] = batch_update(Smeta,A,B,dB,opts,stepnum)

% COMPUTE UPDATE
lsqB = lsq_update(Smeta,A,B,opts);
dB = (lsqB-B)*(1-opts.momentum) + dB*opts.momentum;

% UNBIAS FOR EARLY STEPS
unbias = 1-opts.momentum^stepnum;
B = B + opts.lrate*dB/unbias;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LEAST SQUARES UPDATE
%
% Computes the least squares update for B given the minibatch of A.

function lsqB = lsq_update(Smeta,A,B,opts)

% CONTRIBUTION FROM ELEMENTS WITH S=0
%pseudoInverseAt = A/(A'*A);

G = A' * A;                         
Gcpu = gather(G);                    
Ginv = inv(Gcpu); % solve small inv on CPU due to Matlab's GPU path slowness.
pseudoInverseAt = A * gpuArray(Ginv);

if (use_tf32_relu_bat_c(opts))
  Yt = relu_bat_c_sparse_tf32_nvrtc_mex(A.', B.', pseudoInverseAt.', Smeta.rowPtr, Smeta.edgeI, Smeta.edgeVal, Smeta.rowPtrBase, Smeta.edgeBase);
else
  Yt = relu_bat_c_sparse_fused_nvrtc_mex(A.', B.', pseudoInverseAt.', Smeta.rowPtr, Smeta.edgeI, Smeta.edgeVal, Smeta.rowPtrBase, Smeta.edgeBase);
end
lsqB = B - Yt.';

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function tf32 = use_tf32_relu_bat_c(opts)

persistent gpu_supports_tf32;

requested = isfield(opts,'allow_tf32') && isscalar(opts.allow_tf32) && logical(opts.allow_tf32);
if (~requested)
  tf32 = false;
  return;
end

if (isempty(gpu_supports_tf32))
  gpu_supports_tf32 = false;
  if (canUseGPU())
    cc = sscanf(gpuDevice().ComputeCapability,'%d.%d');
    gpu_supports_tf32 = ~isempty(cc) && cc(1) >= 8;
  end
end

tf32 = gpu_supports_tf32;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
