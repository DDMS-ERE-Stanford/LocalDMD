%% Verifying proven upper bound for system
%
%       dx/dt = C(t)*x(t)
%  where C(t) = [0, 1+a*t; -1-a*t, 0]

clear; clc; rng("default");


% simulate data for the time dependent linear system
x0 = [1,0];
dt = 1e-3;
tend = 20.0;
tspan = 0:dt:tend+dt;
% parameter for strength of linear dynamical system 
eps = 0.1;
nt = length(tspan);
wrapper = @(t, x) linear_time(t, x, eps);
[t,U] = ode45(wrapper, tspan, x0);
U = U';
% Time-shift upper bound
% divide data 
% number of snapshots
M = 1000+1;
all_upper = zeros(M,1);
all_Xnorm = zeros(M,1);
% number of snapshots to delete
k = 1;
m = M-k+1;
all_errors = zeros(M,1);
all_upper_no_physics = zeros(M,1);
all_upper_with_physics = zeros(M,1);

% actual Y norm
all_actual_y_norms = zeros(M,1);
for i = 1:m-1
    % subslice matrices
    Xi = U(:,1:i);
    Yi = U(:,2:i+1);
   
    % k more columns
    Ximore = U(:,1:i+k);
    Yimore = U(:,2:i+k+1);

    u = Ximore(:,end);
    v = Yimore(:,end);
    % -----------
    % system specific bound
    % the eigenvalues of C(t) is always \pm i*(1+a*t), thus the maximum 2 norm
    % over the full interval will be sqrt(1+T)
    gamma = sqrt(1+tend);
    upper = exp((gamma^2)*dt)*(norm(Xi,'fro')+(sqrt(2)*i/gamma^2));
    % save matrix shift bounds
    all_upper(i) = upper;
    all_Xnorm(i) = norm(Xi,2);

    % operator error due to deletion
    % compute operators
    A = Yi*pinv(Xi);
    A_more = Yimore*pinv(Ximore);
    all_errors(i) = norm(A-A_more);
    % compute upper bound (no physics)
    u_norm = norm(u,2);
    v_norm = norm(v,2);
    X_inv_norm = norm(pinv(Xi),2);

    Y_norm = norm(Yi,2);
    all_actual_y_norms(i) = Y_norm;
    %c = 1/(u'*u-u'*Xi*pinv(Xi)*u);
    c = 1/(u'*u);
    const1 = (c^2)*(u_norm^2)*(1+(u_norm/X_inv_norm)^2);
    const2 = (Y_norm^2+v_norm^2);
    const3 = (v_norm*X_inv_norm)^2;
    all_upper_no_physics(i) = sqrt(const1*const2+const3);

    % compute physics-informed upper bound
    Y_norm_physics = all_upper(i)^2;
    % modified const2, others do not depend on Ynorm
    const2 = (Y_norm_physics^2+v_norm^2);
    all_upper_with_physics(i) = sqrt(const1*const2+const3);
    
end



figure(1);
plot_skip=21;
plot(1:plot_skip:M, log10(all_Xnorm(1:plot_skip:end)), ...
    "--", "Color", "green", "LineWidth", 5.0); 

hold on;
plot(1:plot_skip:M, log10(all_actual_y_norms(1:plot_skip:end)), ...
    "-.", "Color", "#EDB120", "LineWidth", 5.0); 

hold on;
plot(1:plot_skip:M, log10(all_upper(1:plot_skip:end)), ...
    "-", "Color", "black", "LineWidth", 5.0); 



%legend(["$||\mathbf{X}||$", "$||\mathbf{Y}||$", ...
 %   "$||\mathbf{Y}||$ bound"], "FontSize", 16, ...
  %  "Location", "southeast", "Interpreter", "latex");
legend({'$\|\mathbf{X}\|$', '$\|\mathbf{Y}\|$', ...
    '$\|\mathbf{Y}\|$ bound'}, ...
       'Interpreter', 'latex', ...
       'Orientation', 'horizontal', ...
       'Location', 'south',"FontSize", 24);
legend boxoff;

ax = gca;
ax.FontSize = 20; 
title("Effect of time shift",'Interpreter','latex');
ytickformat('10^{%0d}');
xlabel("$N_{sn}$", "Interpreter", "latex", 'fontweight','bold');
ylabel("$\|\cdot\|_2$", "Interpreter", "latex", 'fontweight','bold');

xticks([50 250 500 750 1000])
exportgraphics(gcf, "./fig/time_shift_bound.png","Resolution",200);


figure(2);
plot(1:plot_skip:M-1, ...
    log10(all_upper_with_physics(1:plot_skip:end-1)), "-", ...
    "Color", "black", "LineWidth", 5.0); 
hold on;
plot(1:plot_skip:M-1, ...
    log10(all_upper_no_physics(1:plot_skip:end-1)), "-.", ...
    "Color", "blue", "LineWidth", 5.0); 
hold on;
plot(1:plot_skip:M-1, ...
    log10(all_errors(1:plot_skip:end-1)), '--',...
    "LineWidth", 5.0,'Color','green');
lgd = legend(["Error bound", "Error bound (physics)", ...
    "Actual perturbation"]);
lgd.Position = [0.45, 0.40, 0.4, 0.05];
legend boxoff;

ytickformat('10^{%0d}');
xlabel("$N_{sn}$", ...
    "Interpreter", "latex", 'fontweight','bold');
ylabel("$\|\cdot\|_2$", "Interpreter", "latex", ...
    'fontweight','bold');
ax = gca;
ax.FontSize = 20; 
title("Operator perturbation $\|\mathbf{K}-\mathbf{K}_{N_{sn}}\|_2$",'Interpreter','latex');
exportgraphics(gcf, "./fig/physics_informed_perturb.png","Resolution",200);



%% Helper
% simple time dependent linear system
function dxdt = linear_time(t, x, eps)
    dxdt = zeros(2, 1);
    dxdt(1) = (1+eps*t)*x(2);
    dxdt(2) = (-1-eps*t)*x(1);
    dxdt = dxdt + f(t,eps);
end

function feval = f(t, eps)
    feval = zeros(2,1);
    feval(1)=-sin(eps*t);
    feval(2)=cos(eps*t);
end