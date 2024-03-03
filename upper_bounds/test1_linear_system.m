%% Verifying proven upper bound for system
%
%       dx/dt = C(t)*x(t)
%  where C(t) = [0, 1+a*t; -1-a*t, 0]

clear; clc; rng("default");

% simulate data for the time dependent linear system
x0 = [1,0];
dt = 1e-3;
tend = 10.0;
tspan = 0:dt:tend+dt;
nt = length(tspan);
[t,U] = ode45(@linear_time, tspan, x0);
U = U';
%% Time-shift upper bound
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

    Y_norm = norm(Yi);
    %c = 1/(u'*u-u'*Xi*pinv(Xi)*u);
    c = 1/(u'*u);
    const1 = (c^2)*(u_norm^2)*(1+(u_norm/X_inv_norm)^2);
    const2 = (Y_norm^2+v_norm^2);
    const3 = (v_norm*X_inv_norm)^2;
    all_upper_no_physics(i) = sqrt(const1*const2+const3);

    % compute physics-informed upper bound
    Y_norm_physics = all_upper(i);
    % modified const2, others do not depend on Ynorm
    const2 = (Y_norm_physics^2+v_norm^2);
    all_upper_with_physics(i) = sqrt(const1*const2+const3);
    
end


%%
figure(1);
plot_skip=20;
plot(1:plot_skip:M, log10(all_Xnorm(1:plot_skip:end)), ...
    "-*", "Color", "green", "LineWidth", 3.0); 
hold on;
plot(1:plot_skip:M, log10(all_upper(1:plot_skip:end)), ...
    "-*", "Color", "blue", "LineWidth", 3.0); 

legend(["$||\mathbf{X}||$", "Bound on $||\mathbf{Y}||$"], "FontSize", 24, ...
    "Location", "southeast", "Interpreter", "latex");
xlabel("$r$", "Interpreter", "latex", "FontSize", ...
    24, 'fontweight','bold');
ylabel("(Log) Matrix 2-Norm", "Interpreter", "latex", "FontSize", ...
    24, 'fontweight','bold');
ax = gca;
ax.FontSize = 24; 
title("Effect of Time Shift")
exportgraphics(gcf, "./fig/time_shift_bound.png","Resolution",200);


figure(2);
plot(1:M-1, log10(all_upper_with_physics(1:end-1)), "--", ...
    "Color", "blue", "LineWidth", 3.0); 
hold on;
plot(1:M-1, log10(all_upper_no_physics(1:end-1)), "--", ...
    "Color", "green", "LineWidth", 3.0); 
hold on;
plot(1:M-1, log10(all_errors(1:end-1)),"LineWidth", 3.0);
legend(["Error bound (linear system)", "Error bound", ...
    "Actual perturbation"]);
xlabel("$r$", "Interpreter", "latex", "FontSize", ...
    24, 'fontweight','bold');
ylabel("(Log) Matrix 2-Norm Error", "Interpreter", "latex", "FontSize", ...
    24, 'fontweight','bold');
ax = gca;
ax.FontSize = 24; 
title("Updated DMD perturbation")
exportgraphics(gcf, "./fig/physics_informed_perturb.png","Resolution",200);



%% Helper
% simple time dependent linear system
function dxdt = linear_time(t, x)
    dxdt = zeros(2, 1);
    dxdt(1) = (1+0.1*t)*x(2);
    dxdt(2) = (-1-0.1*t)*x(1);
    dxdt = dxdt + f(t);
end

function feval = f(t)
    feval = zeros(2,1);
    feval(1)=-sin(0.1*t);
    feval(2)=cos(0.1*t);
end