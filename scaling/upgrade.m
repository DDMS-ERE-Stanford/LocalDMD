function [Unew,Snew,Vnew] = upgrade(Uold,Sold,Vold,Xnew)
    % given SVD of a (n x k) matrix, incrementally compute
    % SVD of (n x (k+w)) matrix, the new snapshots are given
    % in Xnew, of size (n x w).
    k = size(Vold,1);
    % intermediate matrix
    Wk = [Sold Uold'*Xnew];
    % svd on intermediate matrix
    [Utmp,Stmp,Vtmp] = svd(Wk);
    % return 
    Unew = Uold*Utmp;
    Snew = Stmp;
    Vtmp(1:k,:) = Vold*Vtmp(1:k,:);
    Vnew = Vtmp;
end