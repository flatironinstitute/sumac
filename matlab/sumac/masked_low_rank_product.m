%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Lij = masked_low_rank_product(i,j,A,B,opts)

% GPU?
if (isgpuarray(A))
  Lij = gpu_mask_ABt(i,j,A,B);
  return;
end

% FASTEST ON CPU BUT MEMORY INTENSIVE
if (opts.conserve_memory==0)
  Lij = sum(A(i,:).*B(j,:),2);
  return;
end

% SLOWER BUT MORE MEMORY EFFICIENT
Lij = zeros(length(i),1,"single");
for k=1:size(A,2)
  Lij = Lij + A(i,k)*B(j,k);
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Lij = gpu_mask_ABt(i,j,A,B)

r = size(A,2);
Lij = arrayfun(@inner_product,i,j);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% NESTED FUNCTION

function v = inner_product(ii,jj)

v = single(0); 
for k=1:r 
  v = v + A(ii,k)*B(jj,k); 
end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
