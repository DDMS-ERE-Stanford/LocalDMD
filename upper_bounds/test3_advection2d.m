%% Verifying proven upper bound for system
%
%       dx/dt = C(t)*x(t)
% where C(t) is described by the discretized method of characteristics.

clear; clc; rng("default");

nx = 50; ny = nx;
xmin = -10.0; xmax = 10.0;
xi = linspace(xmin,xmax,nx);
dx=xi(2)-xi(1);
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
            u(j,k,i) = exp(-((x - x0 - v(t)*t)^2 + (y - y0 - w(t)*t)^2)/sigma^2);
        end
    end
end
%%
X = zeros(nx*ny,nt);
for i=1:nt
    i
    tmp=u(:,:,i);
    X(:,i)=tmp(:);
end

%% Time-shift upper bound
% divide data 
% number of snapshots
N=size(X,1);
all_m = 500;
all_actual = zeros(all_m,1);
all_upper = zeros(all_m,1);
% diffusion coef
D = 0.001;
for i = 1:all_m
    i
    t = tspan(i);
    m = i;
    Xm = X(:,1:m);
    Ym = X(:,2:m+1);
    % norm
    actual = norm(Ym,2);
    
    % the eigenvalues of C(t) is always tridiag
    aij = 1-(4.0*dt/(dx^2));
    aim1j = dt*(D/(dx^2))+(v(t)/(2*dx));
    aip1j = dt*(D/(dx^2))-(v(t)/(2*dx));
    aijm1 = dt*(D/(dx^2))+(w(t)/(2*dx));
    aijp1 = dt*(D/(dx^2))-(w(t)/(2*dx));
    Ct = diag(aij*ones(1,N)) + diag(aip1j*ones(1,N-1),1) + diag(aim1j*ones(1,N-1),-1) + ...
        diag(aijp1*ones(1,N-nx),nx) + diag(aijm1*ones(1,N-nx),-nx);
    gamma = max(abs(eig(Ct)));
    disp(gamma)
    upper = exp((gamma^2)*dt)*norm(Xm,'fro');

    all_actual(i) = actual;
    all_upper(i) = upper;
    disp(actual)
    disp(upper)
end
%% save
save("./data/advection2d_Y_bound.mat");
%%
figure(1); plot(1:all_m, (all_actual), "Color", "red", "LineWidth", 3.0); 
hold on; 
plot(1:all_m, (all_upper), "--", "Color", "blue", "LineWidth", 3.0)
legend(["True", "Bound"], "FontSize", 16, "Location", "southeast");
xlabel("$r$", "Interpreter", "latex", "FontSize", 18, 'fontweight','bold');
ylabel("$||\mathbf{Y}||_2$", "Interpreter", "latex", "FontSize", 18, 'fontweight','bold');
ax = gca;
ax.FontSize = 18; 
title("2d Advection")








