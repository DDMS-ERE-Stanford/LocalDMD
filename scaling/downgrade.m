function [Uw,Sw,Vw] = downgrade(Unew,Snew,Vnew,Xold)
    % given SVD of a (n x (k+w)) matrix, incrementally compute
    % SVD of (n x w) matrix, the oldest k snapshots are given
    % in Xold, of size (n x k).
    % [x1 x2 ... xk xk+1 xk+2 ... xk+w] = [Xold Xw]
    % => returns svd([xk+1 xk+2 ... xk+w])
    k = size(Xold,2);
    w = size(Vnew,1)-k;
    tmp = Xold*Vnew(1:k,:);
    % build intermediate Wk
    Wk = Snew-Unew'*tmp;
    % do SVD on Wk
    [Uwk,Swk,Vwk] = svd(Wk);
    % assemble new matrix
    Uw = Unew*Uwk;
    Sw = Swk;
    Vw = Vnew(k+1:k+w,:)*Vwk;
end