%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTE ERRORS

function [cost,rmse,jacc,fnnz] = compute_errors(S,A,B,opts)

% MEMORY BLOCK SIZE
[n,r] = size(B);
max_bytes = opts.block_size_MB * 1e6;
bytes_per_col = whos('A').bytes/r;
cols_per_block = floor(max_bytes/bytes_per_col);

% INITIALIZE
nnzM = 0;
sumM = 0;
ssqM = 0;

% WORK WITH TRANSPOSES
if (opts.use_cuda_mex)
  At = gpuArray(single(A.'));
  Bt = gpuArray(single(B.'));
end

% LOOP OVER BLOCKS OF M=max(0,A*B')
if (opts.use_cuda_mex)
  for b=1:cols_per_block:n
    idxB = b:min(n,b+cols_per_block-1);
    [s,q,z] = relu_bat_reduce_fused_nvrtc_mex(At,Bt(:,idxB));
    sumM = sumM + s;
    ssqM = ssqM + q;
    nnzM = nnzM + double(z);
  end
else
  for b=1:cols_per_block:n
    idxB = b:min(n,b+cols_per_block-1);
    Mt = max(0,B(idxB,:)*A');
    sumM = sumM + sum(Mt,"all");
    ssqM = ssqM + sum(Mt.*Mt,"all");
    nnzM = nnzM + nnz(Mt);
  end
end

% CORRECTION FROM ELEMENTS WITH S>0
[i,j,Sij] = find(S);
Lij = masked_low_rank_product(i,j,A,B,opts);
Mij = max(0,Lij);

% NORMALIZED COST
normS = norm(S,"fro");
zerr = ssqM - sum(Mij.^2) + sum((Sij-Lij).^2);
cost = sqrt(zerr)/normS;

% NORMALIZED ROOT MEAN SQUARED ERROR
ssqe = normS^2 + ssqM - 2*sum(Sij.*Mij);
rmse = sqrt(ssqe)/normS; 

% WEIGHTED JACCARD DISTANCE
numJ = sum(min(Sij,Mij)); 
denJ = sum(S,"all") + sumM - numJ;
jacc = 1-numJ/denJ;

% FRACTION OF NONZEROS
fnnz = nnzM/nnz(S);

% GATHER IF ON GPU
[cost,rmse,jacc,fnnz] = gather(cost,rmse,jacc,fnnz);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
