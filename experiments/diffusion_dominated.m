%% Diffusion-dominated problem
% Author: Hongli Zhao
% Date: 04/22/2023
%
% We simulate random trajectories of Brownian motion with zero mean. 
% After simulation, a plug-in KDE is run at each uniform time grid point
% to obtain a 1D density of the particle. In particular, the density 
% should be the solution of heat equation in 1D. As a linear operator,
% we attempt to reconstruct the solution using DMD.
%
% This MATLAB script is self-contained.
%% Generate data
% Run Monte Carlo simulation
num_paths = 10000;
tspan = 0:0.01:10.0;
dt = tspan(2)-tspan(1);
nt = length(tspan);
data = zeros(num_paths,nt);
for i = 1:num_paths
    if mod(i, 5) == 0
        disp(i)
    end
    % initial condition ~ N(0, 1)
    x0 = randn();
    % create Brownian motion
    process = bm(0.0, 2.5, "StartState", x0);
    [X, T] = simulate(process, nt-1, "DeltaTime", dt);
    % store data
    data(i, :) = X;
end

x1 = data(:,:);
% run KDE at slightly larger domain
x1min = min(min(x1));
x1max = max(max(x1));
x1_std = std(x1(:));
% specify spatial domain
dx = 0.1;
xi = x1min-0.5*x1_std:dx:x1max+0.5*x1_std;
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

% visualize
for i = 1:nt
    figure(1);
    plot(xi, p(:,i), "LineWidth", 1.2, "Color", "blue")
    ylim([0 0.5])
end

%% Standard DMD
% number of snapshots
M = 150;
X_mat = p(:, 1:M); Y_mat = p(:, 2:M+1);
% predictions
X_tilde = zeros(nx, M);
X_tilde(:, 1) = p(:, 1);
% truncate rank
[U,Sigma,V] = svd(X_mat,'econ');
r = find(diag(Sigma)>1e-3);
Ur = U(:,r); Sigmar = Sigma(r,r); Vr = V(:,r);
A_dmd = Y_mat*Vr*pinv(Sigmar)*Ur';

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
ylim([-0.4 0.4])
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

%% Lagrange DMD with estimated advection
u = p;
mean_u = zeros(nt,1);
argmax_u = zeros(nt,1);
[~, max_u_idx] = max(u);
for i = 1:nt
    mean_u(i) = trapz(xi, xi(:).*u(:,i));
    argmax_u(i) = xi(max_u_idx(i));
end
% central differencing velocity estimates (as function of t)
v_from_mean = (mean_u(2:end)-mean_u(1:end-1))./dt;
v_from_argmax = (argmax_u(2:end)-argmax_u(1:end-1))./dt;

% fit a spline to velocity estimate to get a function form
v_spline_from_mean = csaps(tspan(1:end-1), ...
    v_from_mean, 0.01,tspan(1:end-1));
v_spline_from_argmax = csaps(tspan(1:end-1), ...
    v_from_argmax, 0.01,tspan(1:end-1));

% Lagrangian grid does not depend on solution u, compute separately
LagrangianGrid_data = zeros(nx, nt-1);
LagrangianGrid_data(:, 1) = xi;
for i = 1:nt-2
    % velocity is constant in space
    v_i = v_spline_from_mean(i).*ones(nx,1);
    v_ip1 = v_spline_from_mean(i+1).*ones(nx,1);
    LagrangianGrid_data(:,i+1) = LagrangianGrid_data(:,i)+(dt/2).*(v_i+v_ip1);
end

% u is simulated in Eulerian grid, need to interpolate to Lagrangian grid
% at each time step
u_interpolated = zeros(nx, nt-1);
u_interpolated(:, 1) = u(:,1); % initial solution is defined on Eularian grid
for i = 1:nt-2
    u_interpolated(:,i+1) = interp1(xi, ...
        p(:,i+1), LagrangianGrid_data(:,i));
end

% replace NaN with zeros
u_interpolated(isnan(u_interpolated)) = 0.0;

% build DMD on Lagrangian data, first concatenate
R = [u_interpolated; LagrangianGrid_data];

% take snapshots
M = 600;
X_mat = R(:, 1:M); Y_mat = R(:, 2:M+1);

% predictions
X_tilde = zeros(size(R,1), M);
X_tilde(:, 1) = R(:, 1);
% truncate rank
[U,Sigma,V] = svd(X_mat,'econ');
r = find(diag(Sigma)>1e-3);
%r = 1:size(Sigma,1);
%r = 1:100;
Ur = U(:,r); Sigmar = Sigma(r,r); Vr = V(:,r);
A_dmd = Y_mat*Vr*pinv(Sigmar)*Ur';

%% Interpolation
for i = 2:M
    if mod(i, 5) ==0
        disp(i)
    end
    X_tilde(:, i) = A_dmd*X_tilde(:, i-1);
end

% predictions
for i = 2:nt
    if mod(i, 5) ==0
        disp(i)
    end
    X_tilde(:, i) = A_dmd*X_tilde(:, i-1);
end

% project predicted Lagrangian solution back to Eulerian grid
u_predicted_lagrangian = X_tilde(1:nx,1:M);
u_predicted = zeros(nx, M);
u_predicted(:, 1) = u_predicted_lagrangian(:,1); % initial solution is defined on Eularian grid
for i = 1:M-1
    u_predicted(:,i+1) = interp1(LagrangianGrid_data(:,i), ...
        u_predicted_lagrangian(:,i+1), xi);
end
% replace NaN with zeros
u_predicted(isnan(u_predicted)) = 0.0;

% Visualize solution
f = figure(3);
f.Position = [500 400 1000 500];
plot(xi, u_predicted(:,50), "Color", "red", "LineWidth", 2.5);
hold on;
plot(xi, p(:,50), "--", "Color", "black", "LineWidth", 2.5);
hold on;
plot(xi, u_predicted(:,300), "Color", "blue", "LineWidth", 2.5);
hold on;
plot(xi, p(:,300), "--", "Color", "black", "LineWidth", 2.5);
hold on;
plot(xi, u_predicted(:,600), "Color", "blue", "LineWidth", 2.5);
hold on;
plot(xi, p(:,600), "--", "Color", "black", "LineWidth", 2.5);
xlim([-30, 30]);
ylim([0 0.4])
grid on;
title("Lagrangian DMD", "FontSize", 16);
legend(["$t = 0.5$ (DMD)", ...
    "$t = 0.5$ (Exact)", ...
    "$t = 3.0$ (DMD)", ...
    "$t = 3.0$ (Exact)", ...
    "$t = 6.0$ (DMD)", ...
    "$t = 6.0$ (Exact)"], ...
    "Interpreter", "latex", ...
    "FontSize", 16);


