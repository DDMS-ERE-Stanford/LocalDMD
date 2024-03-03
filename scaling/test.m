%% Test 1: Incremental SVD
clear; clc; rng("default");
% X_k+w
n = 10;
k = 4;
w = 4;
Xkw = randn(n,k+w);
% slice Xk portion, and Xw portion
Xk = Xkw(:,1:k); Xw = Xkw(:,k+1:k+w);
% try to update Xk with Xw to match Xkw (no SVD)
% enlarge with 0's
Xk2 = zeros(n,k+w); Xk2(:,1:k) = Xk;
% create extension matrix S = [0 I]
S = zeros(w,k+w); S(:,k+1:k+w) = eye(w);
disp(norm(Xk2+Xw*S - Xkw));

%% Test 2: increment in SVD format directly
clear; clc; rng("default");
n = 20;
k = 6;
w = 8;
Xkw = randn(n,k+w);
% slice Xk portion, and Xw portion
Xk = Xkw(:,1:k); Xw = Xkw(:,k+1:k+w);
% first SVD on Xk
[Uk,Sk,Vk] = svd(Xk);
% build intermediate Wk
Wk = [Sk Uk'*Xw];
% do SVD on Wk
[Uwk,Swk,Vwk] = svd(Wk);
% assemble new matrix
Unew = Uk*Uwk;
Snew = Swk;
% to compute Vnew, need to compute diag[Vk Iw] * Vwk in blocks
tmp = blkdiag(Vk,eye(w));
Vnew = tmp*Vwk;
disp(norm(Unew*Snew*Vnew'-Xkw));

%% Test 3: decrement in SVD format directly
clear; clc; rng("default");
n = 200;
k = 6;
w = 8;
Xkw = randn(n,k+w);
% slice Xk portion, and Xw portion
Xk = Xkw(:,1:k); Xw = Xkw(:,k+1:k+w);
% first SVD on Xk+w
[Ukw,Skw,Vkw] = svd(Xkw);
tmp = [Xk, zeros(n,w)];
% build intermediate Wk
Wk = Skw-Ukw'*tmp*Vkw;
% do SVD on Wk
[Uwk,Swk,Vwk] = svd(Wk);
% assemble new matrix
Unew = Ukw*Uwk;
Snew = Swk;
Vnew = Vkw(k+1:k+w,:)*Vwk;
disp(norm(Unew*Snew*Vnew'-Xw));

%% Test 4(a): Testing upgrade function
clear; clc; rng("default");
n = 20;
k = 6;
w = 8;
Xkw = randn(n,k+w);
% slice Xk portion, and Xw portion
Xk = Xkw(:,1:k); Xw = Xkw(:,k+1:k+w);
% first SVD on Xk
[Uk,Sk,Vk] = svd(Xk);
[Unew,Snew,Vnew] = upgrade(Uk,Sk,Vk,Xw);
disp(norm(Unew*Snew*Vnew'-Xkw));

%% Test 4(b): Testing downgrade function
clear; clc; rng("default");
n = 200;
k = 6;
w = 8;
Xkw = randn(n,k+w);
% slice Xk portion, and Xw portion
Xk = Xkw(:,1:k); Xw = Xkw(:,k+1:k+w);
% first SVD on Xk+w
[Ukw,Skw,Vkw] = svd(Xkw);
[Unew,Snew,Vnew] = downgrade(Ukw,Skw,Vkw,Xk);
disp(norm(Unew*Snew*Vnew'-Xw));

%% Test 4(c): Testing shifting directly in SVD format
clear; clc; rng("default");
n = 500;
k = 6;
w = 8;
Xkw = randn(n,k+w);
% slice Xk portion, and Xw portion
Xk = Xkw(:,1:k); Xw = Xkw(:,k+1:k+w);
% first SVD on Xk+w
[Uk,Sk,Vk] = svd(Xk);
[Utmp,Stmp,Vtmp] = upgrade(Uk,Sk,Vk,Xw);
[Unew,Snew,Vnew] = downgrade(Utmp,Stmp,Vtmp,Xk);
disp(norm(Unew*Snew*Vnew'-Xw));

%% Test 5: plot runtime for different window sizes, compare SVD and update

% problem dimension 
N = 6000;
% window size 
w = 200;
% number of snapshots
M = 10000;
% number of windows
num_windows = floor(N/w);
% generate full dataset
X = randn(N,M);
tic
for i = 1:num_windows
    disp(fprintf(">> Direct ... %d", i));
    % slice window i
    Xi = X(:,((i-1)*w+1):(i*w));
    % compute SVD
    [U_Xi,S_Xi,V_Xi] = svd(Xi);
end
direct_svd_time = toc;

%%
% problem dimension 
N = 6000;
% window size 
w = 200;
% number of snapshots
M = 10000;
% number of windows
num_windows = floor(N/w);
% generate full dataset
X = randn(N,M);
tic
% compute initial SVD
[U,S,V] = svd(X(:,1:w));
[U,S,V] = upgrade(U,S,V,X(:,w+1:2*w));
[U,S,V] = downgrade(U,S,V,X(:,1:w));

incremental_time = toc;



