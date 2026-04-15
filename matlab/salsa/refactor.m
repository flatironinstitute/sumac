%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% function [Ar,Br] = refactor(A,B)
%
% Computes "refactored" matrices Ar and Br such that
%  (1) Ar*Br' = A*B'
%  (2) Ar and Br have the same singular values

function [Ar,Br] = refactor(A,B)

% QR AND SVD
[Qa,Ra] = qr(A,"econ");
[Qb,Rb] = qr(B,"econ");
[U,S,V] = svd(Ra*Rb');

% REFACTOR
Ar = Qa*U*sqrt(S);
Br = Qb*V*sqrt(S);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%