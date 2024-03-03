%% Local Lagrangian DMD, Test 1
%
% Author: Hongli Zhao
% Date: 06/28/2023
% 
% In this experiment, we generate a 1-dimensional advection-dominated 
% PDE solution and reconstruct it using 4 methods.
% (1) standard DMD
% (2) Lagrangian DMD
% (3) standard DMD with windowed recomputations (local DMD)
% (4) Lagrangian DMD with windowed recomputations (local Lagrangian DMD)
%
% The advection velocity is oscillatory in time; a standard DMD
% reconstruction likely cancels out (in time) the significant modes due to
% periodicity. 
%
% In particular, this note demonstrates a regime where standard DMD fails, 
% and a local reconstruction method along with Lagrangian interpolation is
% best.
clear; clc; rng("default");
%%
% Page 44 of the following note describes PDE solution derivation.
% https://scullen.com.au/DSc/Publications/scullen_92.pdf
dt = 0.01;
tspan = 0:dt:10.0;
nt = length(tspan);
dx = 0.05;
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

% ensure p is density
for i = 1:nt
    mass = trapz(xi,p(:,i));
    p(:,i) = p(:,i)/mass;
    disp(trapz(xi,p(:,i)));
end

visualize = false;
if visualize
    for i = 1:nt
        figure(1);
        plot(xi, p(:, i), "Color", "red", "LineWidth", 1.5)
    end
end

%% Standard DMD
M = 800;
X_mat = p(:, 1:M); Y_mat = p(:, 2:M+1);
[U,Sigma,V] = svd(X_mat,'econ');

r = find(diag(Sigma)>1e-6);
Ur = U(:,r); Sigmar = Sigma(r,r); Vr = V(:,r);
A_dmd = Y_mat*Vr*pinv(Sigmar)*Ur';

% predictions
X_tilde = zeros(nx, M);
X_tilde(:, 1) = p(:, 1);

for j = 2:M
    if mod(j, 5) ==0
        disp(j)
    end
    X_tilde(:, j) = A_dmd*X_tilde(:, j-1);
end

% dominant frequency
std_dmd_freq = max(abs(eig(A_dmd)));

visualize = false;
if visualize
    for i = 1:M
        disp(i)
        figure(1);
        plot(xi, X_tilde(:,i), xi, p(:, i), "LineWidth", 1.5)
        xlim([-10 10]);
    end
end
%%
plot_sol = true;
if plot_sol
    % plots reconstructed solution at selected times
    % plots reconstructed solution at selected times
    f = figure(1);
    f.Position = [100 200 1200 1000];
    subplot(2, 2, 1);
    i = 1;
    plot(xi, X_tilde(:, i), "--", "LineWidth", 1.5, "Color", "black"); hold on;
    plot(xi, p(:, i), "LineWidth", 6.0, "Color", [0 0 0 0.2]); hold on;
    legend(["DMD", ...
        "Ref"], "Interpreter", "latex", "Location", "northwest", "FontSize", 24)
    ylim([-0.05 0.4])
    ax = gca;
    ax.FontSize = 24; 
    xlabel("$x$", "Interpreter", "latex", "FontSize", 24);
    ylabel("$u(t, x)$", "Interpreter", "latex", "FontSize", 24);
    subtitle("$t = 0$", "Interpreter", "latex")

    subplot(2, 2, 2);
    i = round(1/4*pi/dt);
    plot(xi, X_tilde(:, i), "--", "LineWidth", 1.5, "Color", "red"); hold on;
    plot(xi, p(:, i), "LineWidth", 6.0, "Color", [1 0 0 0.2]); hold on;
    legend(["DMD", ...
        "Ref"], "Interpreter", "latex", "Location", "northwest", "FontSize", 24)
    ylim([-0.05 0.4])
    ax = gca;
    ax.FontSize = 24; 
    xlabel("$x$", "Interpreter", "latex", "FontSize", 24);
    ylabel("$u(t, x)$", "Interpreter", "latex", "FontSize", 24);

    subtitle("$t = \pi/4$", "Interpreter", "latex")

    subplot(2, 2, 3);
    i = round(1/2*pi/dt);
    plot(xi, X_tilde(:, i), "--", "LineWidth", 1.5, "Color", "blue"); hold on;
    plot(xi, p(:, i), "LineWidth", 6.0, "Color", [0 0 1 0.2]); hold on;
    legend(["DMD", ...
        "Ref"], "Interpreter", "latex", "Location", "northwest", "FontSize", 24)
    ylim([-0.05 0.4])
    ax = gca;
    ax.FontSize = 24; 
    xlabel("$x$", "Interpreter", "latex", "FontSize", 24);
    ylabel("$u(t, x)$", "Interpreter", "latex", "FontSize", 24);
    subtitle("$t = \pi/2$", "Interpreter", "latex")


    subplot(2, 2, 4);
    i = round(pi/dt);
    plot(xi, X_tilde(:, i), "--", "LineWidth", 1.5, "Color", "magenta"); hold on;
    plot(xi, p(:, i), "LineWidth", 6.0, "Color", [1 0 1 0.2]); hold on;
    legend(["DMD", ...
        "Ref"], "Interpreter", "latex", "Location", "northwest", "FontSize", 24)
    ylim([-0.05 0.4])
    ax = gca;
    ax.FontSize = 24; 
    xlabel("$x$", "Interpreter", "latex", "FontSize", 24);
    ylabel("$u(t, x)$", "Interpreter", "latex", "FontSize", 24);
    subtitle("$t = \pi$", "Interpreter", "latex")
 
    sgtitle("1d Advection: Standard DMD vs Numerical Solution ", "FontSize", 24)

    % save image
    exportgraphics(gcf, "./img/1d_advection/1d_advection_standard_dmd.png", "BackgroundColor","white","Resolution", 200);
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compute L^2 error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
p_train = p(:, 1:M);
standard_l2_err = trapz(xi, (X_tilde-p_train).^2);
%% Standard online DMD
% store dominant frequencies
std_online_dmd_freq = [];

X_tilde_online = zeros(nx, M);
X_tilde_online(:,1) = p(:,1);
window_size = 6;
for i = 2:M
    disp(i);
    % slice data in windows
    if i > window_size
        Xtmp = p(:,i-window_size:i);
        Ytmp = p(:,i-window_size+1:i+1);
    else
        Xtmp = p(:,1:i);
        Ytmp = p(:,2:i+1);
        
    end
    if i > window_size
        [Utmp,Sigmatmp,Vtmp] = svd(Xtmp,'econ');
        r = find(diag(Sigmatmp)>1e-8);
        Utmp = Utmp(:,r); Sigmatmp = Sigmatmp(r,r); Vtmp = Vtmp(:,r);
        A_std_onlinedmd = Ytmp*Vtmp*pinv(Sigmatmp)*Utmp';
    else
        A_std_onlinedmd = Ytmp*pinv(Xtmp);
    end

    % compute and store dominant frequency
    std_online_dmd_freq = [std_online_dmd_freq max(abs(eig(A_std_onlinedmd)))];

    % make prediction
    X_tilde_online(:,i) = A_std_onlinedmd*X_tilde_online(:,i-1);
end

%%
plot_sol = true;
if plot_sol
    % plots reconstructed solution at selected times
    f = figure(2);
    f.Position = [100 200 1200 1000];
    subplot(2, 2, 1);
    i = 1;
    plot(xi, X_tilde_online(:, i), "--", "LineWidth", 1.5, "Color", "black"); hold on;
    plot(xi, p(:, i), "LineWidth", 6.0, "Color", [0 0 0 0.2]); hold on;
    legend(["DMD", ...
        "Ref"], "Interpreter", "latex", "Location", "northwest", "FontSize", 24)
    ylim([-0.05 0.4])
    ax = gca;
    ax.FontSize = 24; 
    xlabel("$x$", "Interpreter", "latex", "FontSize", 24);
    ylabel("$u(t, x)$", "Interpreter", "latex", "FontSize", 24);
    subtitle("$t = 0$", "Interpreter", "latex")

    subplot(2, 2, 2);
    i = round(1/4*pi/dt);
    plot(xi, X_tilde_online(:, i), "--", "LineWidth", 1.5, "Color", "red"); hold on;
    plot(xi, p(:, i), "LineWidth", 6.0, "Color", [1 0 0 0.2]); hold on;
    legend(["DMD", ...
        "Ref"], "Interpreter", "latex", "Location", "northwest", "FontSize", 24)
    ylim([-0.05 0.4])
    ax = gca;
    ax.FontSize = 24; 
    xlabel("$x$", "Interpreter", "latex", "FontSize", 24);
    ylabel("$u(t, x)$", "Interpreter", "latex", "FontSize", 24);

    subtitle("$t = \pi/4$", "Interpreter", "latex")

    subplot(2, 2, 3);
    i = round(1/2*pi/dt);
    plot(xi, X_tilde_online(:, i), "--", "LineWidth", 1.5, "Color", "blue"); hold on;
    plot(xi, p(:, i), "LineWidth", 6.0, "Color", [0 0 1 0.2]); hold on;
    legend(["DMD", ...
        "Ref"], "Interpreter", "latex", "Location", "northwest", "FontSize", 24)
    ylim([-0.05 0.4])
    ax = gca;
    ax.FontSize = 24; 
    xlabel("$x$", "Interpreter", "latex", "FontSize", 24);
    ylabel("$u(t, x)$", "Interpreter", "latex", "FontSize", 24);
    subtitle("$t = \pi/2$", "Interpreter", "latex")


    subplot(2, 2, 4);
    i = round(pi/dt);
    plot(xi, X_tilde_online(:, i), "--", "LineWidth", 1.5, "Color", "magenta"); hold on;
    plot(xi, p(:, i), "LineWidth", 6.0, "Color", [1 0 1 0.2]); hold on;
    legend(["DMD", ...
        "Ref"], "Interpreter", "latex", "Location", "northwest", "FontSize", 24)
    ylim([-0.05 0.4])
    ax = gca;
    ax.FontSize = 24; 
    xlabel("$x$", "Interpreter", "latex", "FontSize", 24);
    ylabel("$u(t, x)$", "Interpreter", "latex", "FontSize", 24);
    subtitle("$t = \pi$", "Interpreter", "latex")
 
    sgtitle("1d Advection: Time-varying DMD vs Numerical Solution ", "FontSize", 24)

    % save image
    exportgraphics(gcf, "./img/1d_advection/1d_advection_incremental_dmd.png", "BackgroundColor","white","Resolution",200);
end

%%
% compute L^2 error
p_train = p(:, 1:M);
standard_incremental_l2_err = trapz(xi, (X_tilde_online-p_train).^2);

%% Lagrangian DMD
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

% fit a spline to velocity estimate to get a function form
v_spline_from_mean = csaps(tspan(1:end-1), v_from_mean, 0.1,tspan(1:end-1));

% Lagrangian grid does not depend on solution u, compute separately
LagrangianGrid_data = zeros(nx, nt-1);
LagrangianGrid_data(:, 1) = xi;
for i = 1:nt-2
    % velocity is constant in space
    v_i = v_spline_from_mean(i).*ones(nx,1);
    v_ip1 = v_spline_from_mean(i+1).*ones(nx,1);
    LagrangianGrid_data(:,i+1) = LagrangianGrid_data(:,i)+(dt/2).*(v_i+v_ip1);
end
%% visualize Lagrangian velocity
figure(1);
plot(tspan(1:end-1), v_spline_from_mean, "-", ...
    "LineWidth", 2, "Color","blue"); hold on;
plot(tspan(1:end-1), 2.0*sin(omega*tspan(1:end-1)), "-", ...
    "LineWidth", 2, "Color","red")
xlabel("$t$", "Interpreter", "latex", "FontSize", 24);
ylabel("Velocity", "FontSize", 24);
ax = gca;
lgd = legend(["Estimated", "True"], "Location", "southeast");
fontsize(lgd,14,'points')
ax.FontSize = 24;
exportgraphics(gca,"./img/1d_advection/estimated_velocity.png", "Resolution", 200);
grid on;
%%
% u is simulated in Eulerian grid, need to interpolate to Lagrangian grid
% at each time step
u_interpolated = zeros(nx, nt-1);
u_interpolated(:, 1) = u(:,1); % initial solution is defined on Eularian grid
for i = 1:nt-2
    u_interpolated(:,i+1) = interp1(xi, p(:,i+1), LagrangianGrid_data(:,i));
end

% replace NaN with zeros
u_interpolated(isnan(u_interpolated)) = 0.0;

% build DMD on Lagrangian data, first concatenate
R = [u_interpolated; LagrangianGrid_data];
% take snapshots
M = 900;
X_mat = R(:, 1:M); Y_mat = R(:, 2:M+1);

% predictions
X_tilde = zeros(size(R,1), M);
X_tilde(:, 1) = R(:, 1);
% truncate rank
[U,Sigma,V] = svd(X_mat,'econ');
r = find(diag(Sigma)>1e-6);
Ur = U(:,r); Sigmar = Sigma(r,r); Vr = V(:,r);
A_dmd = Y_mat*Vr*pinv(Sigmar)*Ur';
for i = 2:M
    if mod(i, 5) == 0
        disp(i)
    end
    X_tilde(:, i) = A_dmd*X_tilde(:, i-1);
end

% project predicted Lagrangian solution back to Eulerian grid
u_predicted_lagrangian_online = X_tilde(1:nx,1:M);
u_predicted = zeros(nx, M);
u_predicted(:, 1) = u_predicted_lagrangian_online(:,1); % initial solution is defined on Eularian grid
for i = 1:M-1
    u_predicted(:,i+1) = interp1(LagrangianGrid_data(:,i), u_predicted_lagrangian_online(:,i+1), xi);
end
% replace NaN with zeros
u_predicted(isnan(u_predicted)) = 0.0;

plot_sol = true;
if plot_sol
    % plots reconstructed solution at selected times
    f = figure(1);
    f.Position = [100 200 1200 1000];
    subplot(2, 2, 1);
    i = 1;
    plot(xi, u_predicted(:, i), "--", "LineWidth", 1.5, "Color", "black"); hold on;
    plot(xi, p(:, i), "LineWidth", 6.0, "Color", [0 0 0 0.2]); hold on;
    legend(["DMD", ...
        "Ref"], "Interpreter", "latex", "Location", "northwest", "FontSize", 24)
    ylim([-0.05 0.4])
    ax = gca;
    ax.FontSize = 24; 
    xlabel("$x$", "Interpreter", "latex", "FontSize", 24);
    ylabel("$u(t, x)$", "Interpreter", "latex", "FontSize", 24);
    subtitle("$t = 0$", "Interpreter", "latex")

    subplot(2, 2, 2);
    i = round(1/4*pi/dt);
    plot(xi, u_predicted(:, i), "--", "LineWidth", 1.5, "Color", "red"); hold on;
    plot(xi, p(:, i), "LineWidth", 6.0, "Color", [1 0 0 0.2]); hold on;
    legend(["DMD", ...
        "Ref"], "Interpreter", "latex", "Location", "northwest", "FontSize", 24)
    ylim([-0.05 0.4])
    ax = gca;
    ax.FontSize = 24; 
    xlabel("$x$", "Interpreter", "latex", "FontSize", 24);
    ylabel("$u(t, x)$", "Interpreter", "latex", "FontSize", 24);

    subtitle("$t = \pi/4$", "Interpreter", "latex")

    subplot(2, 2, 3);
    i = round(1/2*pi/dt);
    plot(xi, u_predicted(:, i), "--", "LineWidth", 1.5, "Color", "blue"); hold on;
    plot(xi, p(:, i), "LineWidth", 6.0, "Color", [0 0 1 0.2]); hold on;
    legend(["DMD", ...
        "Ref"], "Interpreter", "latex", "Location", "northwest", "FontSize", 24)
    ylim([-0.05 0.4])
    ax = gca;
    ax.FontSize = 24; 
    xlabel("$x$", "Interpreter", "latex", "FontSize", 24);
    ylabel("$u(t, x)$", "Interpreter", "latex", "FontSize", 24);
    subtitle("$t = \pi/2$", "Interpreter", "latex")


    subplot(2, 2, 4);
    i = round(pi/dt);
    plot(xi, u_predicted(:, i), "--", "LineWidth", 1.5, "Color", "magenta"); hold on;
    plot(xi, p(:, i), "LineWidth", 6.0, "Color", [1 0 1 0.2]); hold on;
    legend(["DMD", ...
        "Ref"], "Interpreter", "latex", "Location", "northwest", "FontSize", 24)
    ylim([-0.05 0.4])
    ax = gca;
    ax.FontSize = 24; 
    xlabel("$x$", "Interpreter", "latex", "FontSize", 24);
    ylabel("$u(t, x)$", "Interpreter", "latex", "FontSize", 24);
    subtitle("$t = \pi$", "Interpreter", "latex")
 
    sgtitle("1d Advection: Physics-aware DMD vs Numerical Solution ", "FontSize", 24)

    % save image
    exportgraphics(gcf, "./img/1d_advection/1d_advection_standard_lagrangian_dmd.png", "BackgroundColor","white","Resolution",200);
end
%% compute l2 error
p_train = p(:, 1:800);
standard_lagrangian_l2_err = trapz(xi, (u_predicted(:,1:800)-p_train).^2);


%% Lagrangian Online DMD
R = [u_interpolated; LagrangianGrid_data];
M = 800;
X_online = zeros(2*nx, M);
X_online(:,1) = R(:,1);
window_size = 6;
for i = 2:M
    disp(i);
    % slice data in windows
    if i <= window_size
        % does not have enough observations yet
        Xtmp = R(:,1:i);
        Ytmp = R(:,2:i+1);
    else
        % only recompute after a few iterations
        if mod(i, 1) == 0
            disp(i);
            disp("Recomputed DMD")
            Xtmp = R(:,i-window_size:i);
            Ytmp = R(:,i-window_size+1:i+1);
        end
    end

    [U,Sigma,V] = svd(Xtmp,'econ');
    r = find(diag(Sigma)>1e-8);
    Ur = U(:,r); Sigmar = Sigma(r,r); Vr = V(:,r);
    A_onlinedmd = Ytmp*Vr*pinv(Sigmar)*Ur';

    % make prediction
    X_online(:,i) = A_onlinedmd*X_online(:,i-1);
end

% project predicted Lagrangian solution back to Eulerian grid
u_predicted_lagrangian_online = X_online(1:nx,1:M);
u_predicted_online = zeros(nx, M);
u_predicted_online(:, 1) = u_predicted_lagrangian_online(:,1); % initial solution is defined on Eularian grid
for i = 1:M-1
    u_predicted_online(:,i+1) = interp1(LagrangianGrid_data(:,i), u_predicted_lagrangian_online(:,i+1), xi);
end
% replace NaN with zeros
u_predicted_online(isnan(u_predicted_online)) = 0.0;

% u is simulated in Eulerian grid, need to interpolate to Lagrangian grid
% at each time step
u_interpolated = zeros(nx, nt-1);
u_interpolated(:, 1) = u(:,1); % initial solution is defined on Eularian grid
for i = 1:nt-2
    u_interpolated(:,i+1) = interp1(xi, p(:,i+1), LagrangianGrid_data(:,i));
end

% replace NaN with zeros
u_interpolated(isnan(u_interpolated)) = 0.0;
%%
plot_sol = true;
if plot_sol
    % plots reconstructed solution at selected times
    f = figure(1);
    f.Position = [100 200 1200 1000];
    subplot(2, 2, 1);
    i = 1;
    plot(xi, u_predicted_online(:, i), "--", "LineWidth", 1.5, "Color", "black"); hold on;
    plot(xi, p(:, i), "LineWidth", 6.0, "Color", [0 0 0 0.2]); hold on;
    legend(["DMD", ...
        "Ref"], "Interpreter", "latex", "Location", "northwest", "FontSize", 24)
    ylim([-0.05 0.4])
    ax = gca;
    ax.FontSize = 24; 
    xlabel("$x$", "Interpreter", "latex", "FontSize", 24);
    ylabel("$u(t, x)$", "Interpreter", "latex", "FontSize", 24);
    subtitle("$t = 0$", "Interpreter", "latex")

    subplot(2, 2, 2);
    i = round(1/4*pi/dt);
    plot(xi, u_predicted_online(:, i), "--", "LineWidth", 1.5, "Color", "red"); hold on;
    plot(xi, p(:, i), "LineWidth", 6.0, "Color", [1 0 0 0.2]); hold on;
    legend(["DMD", ...
        "Ref"], "Interpreter", "latex", "Location", "northwest", "FontSize", 24)
    ylim([-0.05 0.4])
    ax = gca;
    ax.FontSize = 24; 
    xlabel("$x$", "Interpreter", "latex", "FontSize", 24);
    ylabel("$u(t, x)$", "Interpreter", "latex", "FontSize", 24);

    subtitle("$t = \pi/4$", "Interpreter", "latex")

    subplot(2, 2, 3);
    i = round(1/2*pi/dt);
    plot(xi, u_predicted_online(:, i), "--", "LineWidth", 1.5, "Color", "blue"); hold on;
    plot(xi, p(:, i), "LineWidth", 6.0, "Color", [0 0 1 0.2]); hold on;
    legend(["DMD", ...
        "Ref"], "Interpreter", "latex", "Location", "northwest", "FontSize", 24)
    ylim([-0.05 0.4])
    ax = gca;
    ax.FontSize = 24; 
    xlabel("$x$", "Interpreter", "latex", "FontSize", 24);
    ylabel("$u(t, x)$", "Interpreter", "latex", "FontSize", 24);
    subtitle("$t = \pi/2$", "Interpreter", "latex")


    subplot(2, 2, 4);
    i = round(pi/dt);
    plot(xi, u_predicted_online(:, i), "--", "LineWidth", 1.5, "Color", "magenta"); hold on;
    plot(xi, p(:, i), "LineWidth", 6.0, "Color", [1 0 1 0.2]); hold on;
    legend(["DMD", ...
        "Ref"], "Interpreter", "latex", "Location", "northwest", "FontSize", 24)
    ylim([-0.05 0.4])
    ax = gca;
    ax.FontSize = 24; 
    xlabel("$x$", "Interpreter", "latex", "FontSize", 24);
    ylabel("$u(t, x)$", "Interpreter", "latex", "FontSize", 24);
    subtitle("$t = \pi$", "Interpreter", "latex")
 
    sgtitle("1d Advection: Time-varying physics-aware DMD vs Numerical Solution ", "FontSize", 24)

    % save image
    exportgraphics(gcf, "./img/1d_advection/1d_advection_incremental_lagrangian_dmd.png", "BackgroundColor","white", "Resolution",200);
end
%% compute l2 error
p_train = p(:, 1:M);
incremental_lagrangian_l2_err = trapz(xi, (u_predicted_online(:,1:M)-p_train).^2);

%% Compare all errors

figure(10);
plot(tspan(1:M), log10(standard_l2_err), "LineWidth", 1.5, "Color", "red", "LineWidth", 2); hold on;
plot(tspan(1:M), log10(standard_incremental_l2_err), "LineWidth", 1.5, "Color", "blue", "LineWidth", 2); hold on;
loglog(tspan(1:M), log10(standard_lagrangian_l2_err), "LineWidth", 1.5, "Color", "black", "LineWidth", 2); hold on;
loglog(tspan(1:M), log10(incremental_lagrangian_l2_err), "LineWidth", 1.5, "Color", "green", "LineWidth", 2);
legend(["Standard", "Time-varying", "Physics-aware", "Time-varying with physics"], "Location", "southeast", "FontSize", 24);
title("1d Advection: Relative error", "FontSize", 24, "Interpreter", "latex");
xlabel("$t$", "Interpreter", "latex", "FontSize", 24); ylabel("Error in $u$", "Interpreter", "latex", "FontSize", 24);
ax = gca;
ax.FontSize = 24; 
saveas(gcf, "./img/1d_advection/log_error_compare.png");
