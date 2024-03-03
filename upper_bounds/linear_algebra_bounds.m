% Testing pure linear algebra upper bounds

%% Test 1: adding 1 more observation
clear; clc; rng("default");
% state dimension
N = 1000;
num_trials = 10;
% number of snapshots
M = 1001;
k = 1;
m = M-k-1;
all_errors_over_trials = zeros(num_trials, m);
all_upper_over_trials = zeros(num_trials, m);
for kk = 1:num_trials
    disp(kk)
    % draw random data
    all_data = randn(N,M);
    all_errors = zeros(m,1);
    all_upper = zeros(m,1);
    for i = 1:m
        i
        X = all_data(:,1:i);
        Y = all_data(:,2:i+1);
        X_more = all_data(:,1:i+k);
        Y_more = all_data(:,2:i+1+k);
        u = X_more(:,end);
        v = Y_more(:,end);
        % compute operators
        A = Y*pinv(X);
        A_more = Y_more*pinv(X_more);
        all_errors(i) = norm(A-A_more);
        % compute upper bound
        
        assert(rank(X)==size(X,2)); % must be full column rank
        
        u_norm = norm(u,2);
        v_norm = norm(v,2);
        X_inv_norm = norm(pinv(X),2);
        X_norm = norm(X);
        Y_norm = norm(Y);
        c = 1/(u_norm^2-u'*X*pinv(X)*u);
        const1 = (c^2)*(u_norm^2)*(1+(u_norm/X_inv_norm)^2);
        const2 = (Y_norm^2+v_norm^2);
        const3 = (v_norm*X_inv_norm)^2;
        all_upper(i) = sqrt(const1*const2+const3); 
    end
    all_errors_over_trials(kk,:) = all_errors;
    all_upper_over_trials(kk,:) = all_upper;
end
%% save 
save("./data/linear_algebra_bounds.mat", "-v7.3");
%%
load("./data/linear_algebra_bounds.mat");
%%
figure(1);
mean_errors = mean(all_errors_over_trials,1);
mean_uppers = mean(all_upper_over_trials,1);
std_errors = std(all_errors_over_trials,1);
std_uppers = std(all_upper_over_trials,1);
% plus minus 2 standard deviations
errors_high = mean_errors+2.0*std_errors;
errors_low = mean_errors-2.0*std_errors;
uppers_high = mean_uppers+2.0*std_uppers;
uppers_low = mean_uppers-2.0*std_uppers;

tmp = 1:1:m;
plot(tmp, log10(mean_errors), "LineWidth", 3.0, "Color", "red"); hold on;
plot(tmp, log10(mean_uppers), "LineWidth", 3.0, "Color", "blue"); hold on;

% plot upper and lower bands
% plot(1:m, log10(errors_high), "LineWidth", 1.5, "Color", "black"); hold on;
% plot(1:m, log10(errors_low), "LineWidth", 1.5, "Color", "black"); hold on;
% plot(1:m, log10(uppers_high), "LineWidth", 1.5, "Color", "black"); hold on;
% plot(1:m, log10(uppers_low), "LineWidth", 1.5, "Color", "black"); hold on;

legend(["Error", "Bound"], "Interpreter", "latex", "FontSize", 18, 'Position',[0.797 0.119 0.05 0.1]);
xlabel("Number of Snapshots", "Interpreter", "latex", "FontSize", 18);
ylabel("Operator 2-norm Error", "Interpreter", "latex", "FontSize", 18);

% tick sizes
ax = gca;
ax.FontSize = 16; 
xticks([10, 100, 500, 1000])
xticklabels({'10', '10^2', '5\times 10^2', '10^3'});
set(gca, 'XDir','reverse');
yticks([0, 1, 2, 3, 4, 5, 6]);
yticklabels({'0', '10^1', '10^2', '10^3', '10^4', '10^5', '10^6'});
% save figure
exportgraphics(gcf, "./fig/column_deletion.png", "Resolution", 300);





%