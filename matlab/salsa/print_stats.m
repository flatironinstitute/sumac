%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PRINT STATS

function print_stats(iter,cost,rmse,jacc,fnnz,tsec)

% STATS
fprintf(1,'   iter: %04d',iter);
fprintf(1,'   cost: %8.6f',cost);
fprintf(1,'   rmse: %8.6f',rmse);
fprintf(1,'   jacc: %8.6f',jacc);
fprintf(1,'   fnnz: %8.5f',fnnz);
fprintf(1,'   tsec: %.2f\n',tsec);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
