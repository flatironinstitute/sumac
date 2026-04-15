%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% COMPUTE ERRORS

function [cost,rmse,jacc,fnnz] = compute_errors(S,A,B,opts)

% MEMORY BLOCK SIZE
[n,d] = size(B);
max_bytes = opts.block_size_MB * 1e6;
bytes_per_col = whos('A').bytes/d;
cols_per_block = floor(max_bytes/bytes_per_col);

% INITIALIZE
nnzM = 0;
sumM = 0;
ssqM = 0;

% COMPUTE M=max(0,A*B') IN BLOCKS
for b=1:cols_per_block:n
  idxB = b:min(n,b+cols_per_block-1);
  Mt = sparse(max(0,B(idxB,:)*A'));
  sumM = sumM + sum(Mt,"all");
  ssqM = ssqM + norm(Mt,"fro")^2;
  nnzM = nnzM + nnz(Mt);
end

% CORRECTION FROM ELEMENTS WITH S>0
[i,j,Sij] = find(S);
Lij = zeros(size(Sij));
for k=1:d
  Lij = Lij + A(i,k).*B(j,k);  % LOOP TO SAVE MEMORY
end
Mij = max(0,Lij);

% NORMALIZED COST
normS = norm(S,"fro");
zerr = ssqM - sum(Mij.^2) + sum((Sij-Lij).^2);
cost = sqrt(zerr)/normS;

% NORMALIZED ROOT MEAN SQUARED ERROR
ssqS = normS^2;
ssqe = ssqS + ssqM - 2*sum(Sij.*Mij);  % SUM SQUARED ERROR
rmse = sqrt(ssqe)/normS; 

% WEIGHTED JACCARD DISTANCE
numJ = sum(min(Sij,Mij)); 
denJ = sum(S,"all") + sumM - numJ;
jacc = 1-numJ/denJ;

% FRACTION OF NONZEROS
fnnz = nnzM/nnz(S);

% COLLECT
[cost,rmse,jacc,fnnz] = gather(cost,rmse,jacc,fnnz);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
