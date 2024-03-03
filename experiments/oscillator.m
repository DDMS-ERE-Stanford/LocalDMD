% Testing online updates to DMD matrices.
% Author: Hongli Zhao
% Date: 04/30/2023
%
%
%
% This note constructs a 2D oscillator example with time-varying frequency.
% This is a typical case where DMD fails, as it does not include temporal
% nonstationarity. In this case, a locally-updated DMD variant should be 
% better suited.
%
%
%
clear; clc; rng("default");
%%
% simulate data for the time dependent linear system
x0 = [1,0];
dt = 0.01;
tspan = 0:dt:10;
nt = length(tspan);
[t,X] = ode45(@linear_time, tspan, x0);
X = X';

% standard DMD should fail to capture changing frequency of the oscillator
% number of snapshots
M = 1000;
Xdata = X(:,1:M);
Ydata = X(:,2:M+1);
% compute DMD operator
Admd = Ydata*pinv(Xdata);
X_stddmd = zeros(2, M);
X_stddmd(:,1) = x0;
for i = 2:M
    X_stddmd(:,i) = Admd*X_stddmd(:,i-1);
end
X_stddmd = X_stddmd';
X = X';
figure(1);
plot(tspan(1:M), X_stddmd(1:M, 1), "LineWidth", 1.5, "LineStyle", "--"); 
hold on; plot(tspan(1:M), X(1:M,1), "LineWidth", 1.5, "LineStyle", "-");
legend(["DMD", "Truth"]); grid on;
title("Standard DMD (Time-varying Linear System)");
xlabel("$t$", "Interpreter", "latex"); ylabel("$x_1$", "Interpreter", ...
    "latex");

figure(2);
plot(tspan(1:M), X_stddmd(1:M, 2), "LineWidth", 1.5, "LineStyle", "--"); 
hold on; plot(tspan(1:M), X(1:M,2), "LineWidth", 1.5, "LineStyle", "-");
legend(["DMD", "Truth"]); grid on;
title("Standard DMD (Time-varying Linear System)");
xlabel("$t$", "Interpreter", "latex"); ylabel("$x_2$", "Interpreter", ...
    "latex");

%% Locally updated reconstruction
% online updated DMD (naive, recomputes the standard DMD at each time step)
X = X';
all_A_dmd = zeros(M-1, 2, 2);
for i = 2:M
    % slice data in windows
    if i > 20
        Xtmp = X(:,i-20:i);
        Ytmp = X(:,i-19:i+1);
    else
        Xtmp = X(:,1:i);
        Ytmp = X(:,2:i+1);
    end
    all_A_dmd(i-1, :, :) = Ytmp*pinv(Xtmp);
end


% make predictions
X_online = zeros(2, M);
X_online(:,1) = x0;
for i = 2:M
    Atmp = squeeze(all_A_dmd(i-1,:,:));
    X_online(:,i) = Atmp*X_online(:,i-1);
end
X_online = X_online';
X = X';


figure(3);
plot(tspan(1:M), X_online(1:M, 1), "LineWidth", 1.5, "LineStyle", "--"); 
hold on; plot(tspan(1:M), X(1:M,1), "LineWidth", 1.5, "LineStyle", "-");
legend(["DMD", "Truth"]); grid on;
title("Online DMD (Time-varying Linear System) Window size 10");
xlabel("$t$", "Interpreter", "latex"); ylabel("$x_1$", "Interpreter", ...
    "latex");

figure(4);
plot(tspan(1:M), X_online(1:M, 2), "LineWidth", 1.5, "LineStyle", "--"); 
hold on; plot(tspan(1:M), X(1:M,2), "LineWidth", 1.5, "LineStyle", "-");
legend(["DMD", "Truth"]); grid on;
title("Online DMD (Time-varying Linear System) Window size 10");
xlabel("$t$", "Interpreter", "latex"); ylabel("$x_2$", "Interpreter", ...
    "latex");

%% Helper functions

% simple time dependent linear system
function dxdt = linear_time(t, x)
    dxdt = zeros(2, 1);
    dxdt(1) = (1+1.0*t)*x(2);
    dxdt(2) = (-1-1.0*t)*x(1);
end