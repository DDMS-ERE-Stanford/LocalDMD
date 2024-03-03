%% Verifying proven upper bound for system
%
%       dx/dt = C(t)*x(t)
% where C(t) is described by the discretized method of characteristics.

clear; clc; rng("default");

dt = 0.01;
tspan = 0:dt:5.0;
nt = length(tspan);
tstart=0; tend=10.0;
dx = 0.02;
xi = -10.0:dx:10.0;
nx = length(xi);

% snapshots from analytic solution
x0 = 0.0;
k = 0.5;
U = 4.0;
UU = U;
omega = 0.5*pi;

p = zeros(nx, nt);
for i = 1:nx
    for j = 1:nt
        xtmp = xi(i);
        ttmp = tspan(j);
        p(i, j) = exp(-k*(xtmp - x0 - 2*(U/omega)*(sin(omega*ttmp/2)^2))^2);
    end
end


visualize = false;
if visualize
    for i = 1:nt
        figure(1);
        plot(xi, p(:, i), "Color", "red", "LineWidth", 1.5)
    end
end

% using a central differencing discretization we have the C(t) matrix is:
% tridiag[a(tn)*dt/2*dx un(i-1) + un(i) + (-a(tn)*dt/2*dx) un(i+1)] where n
% is time step, i is space, a is advection velocity.

% the eigenvalues are given by: https://en.wikipedia.org/wiki/Tridiagonal_matrix
%% Time-shift upper bound
% divide data 
% number of snapshots
X=p;
N=size(X,1);
all_m = 500;
all_actual = zeros(all_m,1);
all_upper = zeros(all_m,1);
for i = 1:all_m
    i
    t = tspan(i);
    m = i;
    Xm = X(:,1:m);
    Ym = X(:,2:m+1);
    % norm
    actual = norm(Ym,2);
    
    % the eigenvalues of C(t) is always tridiag
    Ct = diag(ones(1,N)) + diag((-2.0*sin(pi*t/2)*dt/(2*dx))*ones(1,N-1),1) + diag((2.0*sin(pi*t/2)*dt/(2*dx))*ones(1,N-1),-1);
    gamma = max(abs(eig(Ct)));
    disp(gamma)
    upper = exp((gamma^2)*dt)*norm(Xm,'fro');

    all_actual(i) = actual;
    all_upper(i) = upper;
end
%% save
save("./data/advection1d_Y_bound.mat");
%%
load("./data/advection1d_Y_bound.mat");
%%
figure(1); plot(1:all_m, (all_actual), "Color", "red", "LineWidth", 3.0); 
hold on; 
plot(1:all_m, (all_upper), "--", "Color", "blue", "LineWidth", 3.0)
legend(["True", "Bound"], "FontSize", 16, "Location", "southeast");
xlabel("$r$", "Interpreter", "latex", "FontSize", 18, 'fontweight','bold');
ylabel("$||\mathbf{Y}||_2$", "Interpreter", "latex", "FontSize", 18, 'fontweight','bold');
ax = gca;
ax.FontSize = 18; 
title("1d Advection")








