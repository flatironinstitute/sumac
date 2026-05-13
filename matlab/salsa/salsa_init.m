%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% INITIALIZE

function [A,B] = salsa_init(S,d,opts)

% START CLOCK
tInit = tic;

% USER-INITIALIZED?
[m,n] = size(S);
if (isfield(opts,"initfile"))
  A = load(opts.initfile).("A");
  B = load(opts.initfile).("B");
  assert(isequal(size(A),[m d]));
  assert(isequal(size(B),[n d]));
  A = single(A);
  B = single(B);
  if (canUseGPU())
    A = gpuArray(A);
    B = gpuArray(B);
  end
  return;
end

% GPU?
if (canUseGPU())
  S = gpuArray(S);
end

% INITIALIZE WITH A>0 AND B<0 SO THAT A*B'<0
rng(opts.seed,'twister');
T = transpose(S);
A = +sqrt(S*rand(n,d));
B = -sqrt(T*rand(m,d));

% RESCALE TO FIXED INITIAL COST OF 2
[i,j,Sij] = find(S);
normS = norm(Sij);
% Lij = sum(A(i,:).*B(j,:),2);
Lij = zeros(size(Sij));
for k=1:d
  Lij = Lij + A(i,k).*B(j,k);  % LOOP TO SAVE MEMORY
end
% SOLVE: norm(Sij-Lij*scale^2) = 2*normS
normL = norm(Lij);
dotLS = sum(Lij.*Sij);
scale = sqrt((dotLS+sqrt(dotLS*dotLS+3*normL^2*normS^2)))/normL;
A = scale*A;
B = scale*B;

% REFACTOR
[A,B] = refactor(A,B);

% REPORT
[iter,fnnz] = deal(0);
[rmse,jacc] = deal(1);
tsec = toc(tInit);
cost = norm(Sij-Lij*scale^2)/normS;
print_stats(iter,cost,rmse,jacc,fnnz,tsec);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
