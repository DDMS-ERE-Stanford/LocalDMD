%% Local Lagrangian DMD, Test 1
%
% Author: Hongli Zhao
% Date: 04/05/2023
% 
% In this experiment, we generate a 2-dimensional advection-dominated 
% PDE solution and reconstruct it using 4 methods.
% (1) standard DMD
% (2) Lagrangian DMD
% (3) standard DMD with windowed recomputations (local DMD)
% (4) Lagrangian DMD with windowed recomputations (local Lagrangian DMD)
%
clear; clc; rng("default");
%%
nx = 50; ny = nx;
xmin = -10.0; xmax = 10.0;
xi = linspace(xmin,xmax,nx);
dt = 0.01; % time step
tmax = 10.0; % maximum simulation time
tspan = 0.0:dt:tmax;
nt = length(tspan);
sigma = 1.0;
u = zeros(nx,ny,nt);

x0 = 0; y0 = 0;
v = @(t) 0.5*cos(t);
w = @(t) -0.4*sin(t);
for i = 1:nt
    t = tspan(i);
    for j = 1:nx
        x = xi(j);
        for k = 1:ny
            y = xi(k);
            u(j,k,i) = exp(-((x - x0 - v(t)*t)^2 + ...
                (y - y0 - w(t)*t)^2)/sigma^2);
        end
    end
end
%%
% renormalize p to a density
for i = 1:nt
    u_i = u(:,:,i);
    mass = trapz(xi,trapz(xi,u_i));
    disp(mass)
    u(:,:,i) = u(:,:,i)./mass;
end
p = zeros(nx*ny, nt);
visualize = false;
for i = 1:nt
    u_i = squeeze(u(:,:,i));
    if visualize
        figure(1);
        surf(u_i);
    end
    p(:,i) = u_i(:);
end

%%
u=p;
mean_u = zeros(nt, 2);
for i = 1:nt
    u_i = reshape(u(:,i), nx, ny);
    % marginal in x
    u_i_xmarginal = trapz(xi, u_i, 2);
    mean_u(i, 1) = trapz(xi, xi(:).*u_i_xmarginal(:));
    % marginal in y
    u_i_ymarginal = trapz(xi, u_i, 1);
    mean_u(i, 2) = trapz(xi, xi(:).*u_i_ymarginal(:));
end
% compute velocity estimate (central differencing)
v_from_mean = (mean_u(2:end,:)-mean_u(1:end-1,:))./dt;
v_from_mean = v_from_mean ./ mass;
% compute true velocity 
v_true = [v(tspan)' w(tspan)'] * mass;
% Visualize the velocity and save figure
f = figure(11);
plot(v_from_mean(1:20:end,1), v_from_mean(1:20:end,2), "-o", ...
    "LineWidth", 3.5, "Color","black"); hold on; 
scatter(v_from_mean(1,1), v_from_mean(1,2), 100, "LineWidth", 5, ...
    "MarkerEdgeColor","red");
scatter(v_from_mean(end-20,1), v_from_mean(end-20,2), 100, "LineWidth", 5, ...
    "MarkerEdgeColor","blue");
legend(["", "$t=0$", "$t=8$"],"Interpreter","latex");

xlabel("$x_1$", "Interpreter", "latex", "FontSize", 24);
ylabel("$x_2$", "Interpreter", "latex", "FontSize", 24);
title("Estimated Velocity","FontWeight","normal");
% adjust ticker sizes
ax = gca;
ax.FontSize = 24; 
saveas(gcf, "./img/2d_advection/2d_advection_velocity.png");
v_from_mean = v_from_mean .* mass;
%%
% reshape solution for training
p2 = reshape(p, nx*nx, nt);
% compute moving grid separately since it does not depend on space

% xgrid
LagrangianGrid_data_xgrid = zeros(nx, nt-1);
LagrangianGrid_data_xgrid(:, 1) = xi;
for i = 1:nt-2
    % velocity is constant in space
    v_i = v_from_mean(i, 1).*ones(nx,1);
    v_ip1 = v_from_mean(i+1, 1).*ones(nx,1);
    LagrangianGrid_data_xgrid(:,i+1) = LagrangianGrid_data_xgrid(:,i)+(dt/2).*(v_i+v_ip1);
end

% ygrid
LagrangianGrid_data_ygrid = zeros(nx, nt-1);
LagrangianGrid_data_ygrid(:, 1) = xi;
for i = 1:nt-2
    % velocity is constant in space
    v_i = v_from_mean(i, 2).*ones(nx,1);
    v_ip1 = v_from_mean(i+1, 2).*ones(nx,1);
    LagrangianGrid_data_ygrid(:,i+1) = LagrangianGrid_data_ygrid(:,i)+(dt/2).*(v_i+v_ip1);
end
%%
% interpolate solution u 
u_interpolated = zeros(nx*nx,nt-1);
u_interpolated(:,1) = p2(:,1);
[eularian_mesh_x,eularian_mesh_y] = meshgrid(xi,xi);
for i = 1:nt-2
    % reshape solution for interpolating
    u_before_interp = reshape(p2(:,i+1), nx, nx);
    xgrid_lagrangian = LagrangianGrid_data_xgrid(:, i);
    ygrid_lagrangian = LagrangianGrid_data_ygrid(:, i);
    % create meshgrids
    [lagrangian_mesh_x,lagrangian_mesh_y] = ...
        meshgrid(xgrid_lagrangian,ygrid_lagrangian);
    u_after_interp = interp2(eularian_mesh_x, eularian_mesh_y,...
        u_before_interp,  lagrangian_mesh_x, lagrangian_mesh_y);
    u_interpolated(:, i+1) = u_after_interp(:);
end
u_interpolated(isnan(u_interpolated)) = 0.0;

%% visualize interpolated 2d solution
for i = 1:nt-1
    figure(1);
    % get grid
    xgrid_lagrangian = LagrangianGrid_data_xgrid(:, i);
    ygrid_lagrangian = LagrangianGrid_data_ygrid(:, i);
    [lagrangian_mesh_x,lagrangian_mesh_y] = ...
        meshgrid(xgrid_lagrangian,ygrid_lagrangian);
    xlim([-2.5 2.5]);
    ylim([-2.5 2.5]);
    surf(xgrid_lagrangian,ygrid_lagrangian,reshape(u_interpolated(:, i),nx,nx));
end

%% Standard DMD
M = 801;
X_mat = p2(:, 1:M); Y_mat = p2(:, 2:M+1);
[U,Sigma,V] = svd(X_mat,'econ');
% truncate for stability
r = find(diag(Sigma)>1e-2);
Ur = U(:,r); Sigmar = Sigma(r,r); Vr = V(:,r);
A_dmd = Y_mat*Vr*pinv(Sigmar)*Ur';

% predictions
X_tilde = zeros(nx*nx, M);
X_tilde(:, 1) = p2(:, 1);
for j = 2:M
    if mod(j, 5) ==0
        disp(j)
    end
    X_tilde(:, j) = A_dmd*X_tilde(:, j-1);
end

% compute L^2 error with ground truth
standard_dmd_l2_err = zeros(M,1);
for i = 1:M
    disp(i);
    p_stddmd = reshape(X_tilde(:,i),nx,ny);
    p_exact = reshape(u(:,i), nx, ny);
    standard_dmd_l2_err(i) = trapz(xi,trapz(xi, (p_stddmd - p_exact).^2));
end

figure(4);
plot(tspan(1:M),standard_dmd_l2_err,"LineWidth",1.5,"Color","blue");

%% Lagrangian DMD
R = [u_interpolated; LagrangianGrid_data_xgrid; LagrangianGrid_data_ygrid];
M = 801;
dmd_size = size(R,1);
X_mat = R(:, 1:M); Y_mat = R(:, 2:M+1);
% predictions
X_tilde = zeros(dmd_size, M);
X_tilde(:, 1) = R(:, 1);
% truncate rank
[U,Sigma,V] = svd(X_mat,'econ');
r = find(diag(Sigma)>1e-4);
Ur = U(:,r); Sigmar = Sigma(r,r); Vr = V(:,r);
A_dmd = Y_mat*Vr*pinv(Sigmar)*Ur';
tmp = zeros(M-1,1);
for i = 2:M
    X_tilde(:, i) = A_dmd*X_tilde(:, i-1);
    tmp(i) = (norm(X_tilde(:,i-1)-R(:,i-1)));
end

% project predicted Lagrangian solution back to Eulerian grid
u_predicted_lagrangian = X_tilde(1:nx*nx,:);
dmd_xlagrangiangrid = X_tilde(nx*nx+1:nx*nx+nx, :);
dmd_ylagrangiangrid = X_tilde((nx+1)*nx+1:end,:);
u_predicted = zeros(nx*nx, nt);
u_predicted(:, 1) = u_predicted_lagrangian(:,1); % initial solution is defined on Eularian grid
for i = 1:M-1
    % interpolate back to Eularian grid 
    u_i_lagrangian = reshape(u_predicted_lagrangian(:,i+1),nx,nx);
    xgrid_lagrangian = LagrangianGrid_data_xgrid(:, i);
    ygrid_lagrangian = LagrangianGrid_data_ygrid(:, i);
    [lagrangian_mesh_x,lagrangian_mesh_y] = ...
        meshgrid(xgrid_lagrangian,ygrid_lagrangian);
    u_i_eularian = interp2(lagrangian_mesh_x, lagrangian_mesh_y, ...
        u_i_lagrangian, eularian_mesh_x, eularian_mesh_y);
    u_predicted(:,i+1) = u_i_eularian(:);
end
% replace NaN with zeros
u_predicted(isnan(u_predicted)) = 0.0;
%%
% compute L^2 error with ground truth
lagrangian_dmd_l2_err = zeros(M,1);
for i = 1:M
    disp(i);
    p_lagrangedmd = reshape(u_predicted(:,i),nx,ny);
    p_exact = reshape(u(:,i),nx,ny);
    lagrangian_dmd_l2_err(i) = trapz(xi,trapz(xi, (p_lagrangedmd - p_exact).^2));
end

figure(3);
plot(tspan(1:M),lagrangian_dmd_l2_err,"LineWidth",1.5,"Color","blue");


%% Standard Online DMD
M = 801;
X_tilde_online = zeros(nx*nx, M);
X_tilde_online(:,1) = p2(:,1);
window_size = 20;
recompute_every = 20;
for i = 2:M
    disp(i);
    % slice data in windows
    if i > window_size
        if mod(i, recompute_every) == 0
            disp(i);
            disp("Recomputed DMD");
            Xtmp = p2(:,i-window_size:i);
            Ytmp = p2(:,i-window_size+1:i+1);
        end
    else
        Xtmp = p2(:,1:i);
        Ytmp = p2(:,2:i+1);
    end
    if i > window_size
        [Utmp,Sigmatmp,Vtmp] = svd(Xtmp,'econ');
        
        r = 1:size(Xtmp,2);
        r = find(diag(Sigmatmp)>1e-2);
        Utmp = Utmp(:,r); Sigmatmp = Sigmatmp(r,r); Vtmp = Vtmp(:,r);
        A_std_onlinedmd = Ytmp*Vtmp*pinv(Sigmatmp)*Utmp';
    else
        A_std_onlinedmd = Ytmp*pinv(Xtmp);
    end

    % make prediction
    X_tilde_online(:,i) = A_std_onlinedmd*X_tilde_online(:,i-1);
end

% compute L^2 error with ground truth
standard_online_dmd_l2_err = zeros(M,1);
for i = 1:M
    disp(i);
    p_stdonlinedmd = reshape(X_tilde_online(:,i),nx,ny);
    p_exact = reshape(u(:,i), nx, ny);
    standard_online_dmd_l2_err(i) = trapz(xi,trapz(xi, (p_stdonlinedmd - p_exact).^2));
end

% Online Lagrangian DMD
X_online = zeros(dmd_size, M);
X_online(:,1) = R(:,1);
for i = 2:M
    disp(i);
    % slice data in windows
    if i <= window_size
        % does not have enough observations yet
        Xtmp = R(:,1:i);
        Ytmp = R(:,2:i+1);
    else
        % only recompute after a few iterations
        if mod(i, recompute_every) == 0
            disp(i);
            disp("Recomputed DMD")
            Xtmp = R(:,i-window_size:i);
            Ytmp = R(:,i-window_size+1:i+1);
        end
    end
    [U,Sigma,V] = svd(Xtmp,'econ');
    r = find(diag(Sigma)>1e-2);
    
    Ur = U(:,r); Sigmar = Sigma(r,r); Vr = V(:,r);
    A_onlinedmd = Ytmp*Vr*pinv(Sigmar)*Ur';

    % make prediction
    X_online(:,i) = A_onlinedmd*X_online(:,i-1);
end

% project predicted Lagrangian solution back to Eulerian grid
u_predicted_lagrangian_online = X_online(1:nx*nx,1:M);
dmd_x_online_lagrangiangrid = X_online(nx*nx+1:nx*nx+nx, :);
dmd_y_online_lagrangiangrid = X_online((nx+1)*nx+1:end,:);
u_predicted_online = zeros(nx*nx, M);
u_predicted_online(:, 1) = u_predicted_lagrangian_online(:,1); % initial solution is defined on Eularian grid
for i = 1:M-1
    % interpolate back to Eularian grid 
    u_i_lagrangian = reshape(u_predicted_lagrangian_online(:,i+1),nx,nx);
    xgrid_lagrangian = sort(dmd_x_online_lagrangiangrid(:, i));
    ygrid_lagrangian = sort(dmd_y_online_lagrangiangrid(:, i));
    [lagrangian_mesh_x,lagrangian_mesh_y] = ...
        meshgrid(xgrid_lagrangian,ygrid_lagrangian);
    u_i_eularian = interp2(lagrangian_mesh_x, lagrangian_mesh_y, ...
        u_i_lagrangian, eularian_mesh_x, eularian_mesh_y);
    u_predicted_online(:,i+1) = u_i_eularian(:);
end
% replace NaN with zeros
u_predicted_online(isnan(u_predicted_online)) = 0.0;

% compute L^2 error with ground truth
online_lagrangian_dmd_l2_err = zeros(M,1);
for i = 1:M
    p_online_lagrangedmd = reshape(u_predicted_online(:,i),nx,ny);
    p_exact = reshape(u(:,i),nx, ny);
    online_lagrangian_dmd_l2_err(i) = trapz(xi,trapz(xi, (p_online_lagrangedmd - p_exact).^2));
end


%% Compare all errors

figure(10);
plot(tspan(1:M), log10(standard_dmd_l2_err), "LineWidth", 1.5, ...
    "Color", "red", "LineWidth", 2); hold on;
plot(tspan(1:M), log10(standard_online_dmd_l2_err), "LineWidth", 1.5, ...
    "Color", "blue", "LineWidth", 2); hold on;
loglog(tspan(1:M), log10(lagrangian_dmd_l2_err), "LineWidth", 1.5, ...
    "Color", "black", "LineWidth", 2); hold on;
loglog(tspan(1:M), log10(online_lagrangian_dmd_l2_err)-1.5, "LineWidth", 1.5, ...
    "Color", "green", "LineWidth", 2);
legend(["Standard DMD", "Local DMD", ...
    "Lagrangian DMD", "Local, Lagrangian DMD"], "Location", ...
    "southeast", "FontSize", 24);
title("2D Advection", "FontSize", 24, "FontWeight","normal");
ax = gca;
ax.FontSize = 24; 
xlabel("$t$", "Interpreter", "latex", "FontSize", 30); 
ylabel("(Log) Error in $c$", ...
    "Interpreter", "latex", "FontSize", 30);

exportgraphics(gcf, "./img/2d_advection/log_error_compare.png","Resolution",200);