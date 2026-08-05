%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [A,B] = salsa_init(S,r,opts)

% USER-INITIALIZED?
[m,n] = size(S);
if (isfield(opts,"init_file"))
  A = single(load(opts.init_file).("A"));
  B = single(load(opts.init_file).("B"));
  assert(isequal(size(A),[m r]));
  assert(isequal(size(B),[n r]));
  return;
end

% START CLOCK
tInit = tic;

% INITIALIZE WITH A>0 AND B<0 SO THAT A*B'<0
rng(opts.seed,"twister");
A = +sqrt(S*rand(n,r));
B = -sqrt(S'*rand(m,r));

% SOLVE: norm(Sij-Lij*scale^2) = 2*normS
[i,j,Sij] = find(S);
Lij = masked_low_rank_product(i,j,A,B,opts);
normL = norm(Lij);
normS = norm(Sij);
dotLS = Lij'*Sij;
scale = sqrt((dotLS+sqrt(dotLS*dotLS+3*normL^2*normS^2)))/normL;

% RESCALE AND REFACTOR
A = scale*A;
B = scale*B;
[A,B] = refactor(A,B);

% REPORT
tsec = toc(tInit);
[iter,fnnz,rmse,jacc] = deal(0,0,1,1);
cost = norm(Sij-Lij*scale^2)/normS;
print_stats(iter,cost,rmse,jacc,fnnz,tsec,cost<Inf);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
