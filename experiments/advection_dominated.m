%% Advection-dominated problem
% Author: Hongli Zhao
% Date: 04/04/2023
%
% We simulate an advection-dominated example, where we generate random
% normal initial conditions, and advect the mass using dx/dt = c, where 
% c is a constant velocity. Upon simulating the random solutions, we 
% run a plug-in KDE to generate distributions of the particle over time.
% In particular, the density should be a pure advection with Gaussian 
% initial conditions. 
% 
% To reconstruct the solution, we run standard DMD and the so-called 
% orthogonal DMD. 
%
% This MATLAB script is self-contained.
%
clear; clc; rng("default");
%% Generate data
% Run Monte Carlo simulation
num_paths = 5000;
tspan = 0:0.01:10.0;
dt = tspan(2)-tspan(1);
nt = length(tspan);
data = zeros(num_paths,nt);
for i = 1:num_paths
    if mod(i, 5) == 0
        disp(i)
    end
    % initial condition ~ N(0,16)
    x0 = zeros(1,1);
    x0(1) = 4*randn();
    [t,X] = ode45(@advect, tspan, x0);
    % store data
    data(i, :) = X;
end

% KDE density
x1 = data(:,:);
% run KDE at slightly larger domain
x1min = min(min(x1));
x1max = max(max(x1));
x1_std = std(x1(:));
% specify spatial domain
dx = 0.1;
xi = x1min-x1_std:dx:x1max+x1_std;
nx = length(xi);

p = zeros(nx, nt);
for i = 1:nt
    if mod(i, 5) == 0
        disp(i)
    end
    % KDE
    [f, ~] = ksdensity(x1(:, i), xi);
    p(:, i) = f;
end

% ensure density
all_integrals = zeros(nt,1);
for i = 1:nt
    all_integrals(i) = trapz(xi,p(:,i));
end
% renormalize to ensure density
for i = 1:nt
    p(:,i) = p(:,i)./all_integrals(i);
end
%% Standard DMD
% standard DMD fails for advection problem
M = 300;
X_mat = p(:, 1:M); Y_mat = p(:, 2:M+1);
% predictions
X_tilde = zeros(nx, nt);
X_tilde(:, 1) = p(:, 1);
% truncate rank
[U,Sigma,V] = svd(X_mat,'econ');
r = find(diag(Sigma)>1e-3);
Ur = U(:,r); Sigmar = Sigma(r,r); Vr = V(:,r);
A_dmd = Y_mat*Vr*pinv(Sigmar)*Ur';
for i = 2:M
    if mod(i, 5) ==0
        disp(i)
    end
    X_tilde(:, i) = A_dmd*X_tilde(:, i-1);
end

% view predicted values
f = figure(2);
f.Position = [500 400 1000 500];
plot(xi, X_tilde(:,50), "Color", "red", "LineWidth", 2.5);
hold on;
plot(xi, X_mat(:,50), "--", "Color", "black", "LineWidth", 2.5);
hold on;
plot(xi, X_tilde(:,100), "Color", "blue", "LineWidth", 2.5);
hold on;
plot(xi, X_mat(:,100), "--", "Color", "black", "LineWidth", 2.5);
hold on;
plot(xi, X_tilde(:,150), "Color", "green", "LineWidth", 2.5);
hold on;
plot(xi, X_mat(:,150), "--", "Color", "black", "LineWidth", 2.5);
xlim([-30, 30]);
ylim([0.0 0.1])
grid on;
title("Standard DMD", "FontSize", 16);
legend(["$t = 0.5$ (DMD)", ...
    "$t = 0.5$ (Exact)", ...
    "$t = 1.0$ (DMD)", ...
    "$t = 1.0$ (Exact)", ...
    "$t = 1.5$ (DMD)", ...
    "$t = 1.5$ (Exact)"], ...
    "Interpreter", "latex", ...
    "FontSize", 16);

%% Orthogonal DMD
u = sqrt(p);
% number of snapshots
M = 300;
X_mat = u(:, 1:M); Y_mat = u(:, 2:M+1);
n_u = size(u,1);
% standard DMD, solve the Procrustes problem
B = X_mat*Y_mat';
[U,S,V] = svd(B, "econ");
A = V*U'; % A is unitary
% generate predictions for sqrt(p)
u_predicted = zeros(size(u));
u_predicted(:,1) = u(:,1);
for i = 2:M
    u_predicted(:,i) = A*u_predicted(:,i-1);
end
% check norm is preserved
all_ints = zeros(M,1);
for i = 1:M
    integral_i = sum(u_predicted(:,i).^2*dx);
    disp(integral_i)
    all_ints(i) = integral_i;
end
% transform back
p_predicted = u_predicted.^2;


% view predicted values
f = figure(2);
f.Position = [500 400 1000 500];
plot(xi, p_predicted(:,50), "Color", "red", "LineWidth", 2.5);
hold on;
plot(xi, p(:,50), "--", "Color", "black", "LineWidth", 2.5);
hold on;
plot(xi, p_predicted(:,100), "Color", "blue", "LineWidth", 2.5);
hold on;
plot(xi, p(:,100), "--", "Color", "black", "LineWidth", 2.5);
hold on;
plot(xi, p_predicted(:,150), "Color", "green", "LineWidth", 2.5);
hold on;
plot(xi, p(:,150), "--", "Color", "black", "LineWidth", 2.5);
xlim([-30, 30]);
ylim([0.0 0.1])
grid on;
title("Orthogonal DMD", "FontSize", 16);
legend(["$t = 0.5$ (DMD)", ...
    "$t = 0.5$ (Exact)", ...
    "$t = 1.0$ (DMD)", ...
    "$t = 1.0$ (Exact)", ...
    "$t = 1.5$ (DMD)", ...
    "$t = 1.5$ (Exact)"], ...
    "Interpreter", "latex", ...
    "FontSize", 16);

%% helper 
function dxdt = advect(t, x)
    % Implements the right hand side
    dxdt = 10;
end