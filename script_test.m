%% Test Script
% Seong Jae Hwang (ICCV 2015)
%% Initializations
% optimization and other parameters
opts.rho1 = 0.01;           % Wolfe condition c1 - step length (Armijo)
opts.rho2 = 0.99;           % Wolfe condition c2 - curvature
opts.grad_epsilon = 1e-7;   % terminate when gradient change less than epsilon
opts.print_interval = 100;  % print outputs interval
opts.max_iter = 1000;       % maximum iteration (better to just rely on this than grad_epsilon)
opts.k = 10;                % submatrix size i (Eq.4, i rows of submatrix V)

% framework parameters
p = 5;                      % PCA rank p
r = 1;                      % tensor rank r
lambda = 0;               % regularization parameter lambda

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Input Arguments %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Primary matrix M. (Eq.3) Replace with your choice
M = rand(50,50); M = (M*M') / 2;

% Secondary matrix D. (Eq.3) Replace with your choice
D = rand(50,50); D = (D*D') / 2; 
% Recommend preconditioning (in this case, an identity marix)
D = D + 1e-3*eye(size(D,1));
D = eye(size(M,1));

% Supplementary vector alpha. (Eq.3) Replace with your choice
alpha = rand(30,1);             % notice size(alpha,1) < size(M,1)
% indices of M which correspond to the supplementary information alpha
alpha_idx = 1:length(alpha);    % arbitrarily chosen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Initial eigenvectors and function handles initialization
% feasible initial eigenvectors V0
[E, psd] = chol(D);
E_inv = inv(E);
V0 = E_inv(:, end-p+1:end);

% objective function handle. Set this to -f for max objective function (PCA)
f = @(C, V) -trace(V'*C*V) + lambda*sum(abs(alpha - V(alpha_idx,1)));

% gradient function handle. Set this to -g for max objective function (PCA)
grad = @(C, V, K) -partial_trace_grad(C, V, K, alpha, alpha_idx, lambda);

%% Solver
fprintf(1, '------------------ Solving R-GEP ------------------\n');
fprintf(1, 'PCA rank: %d, Tensor rank: %d, lambda: %d\n', p, r, lambda);
Vout = RGEP(M, f, grad, D, V0, opts);
fprintf(1, '-------------- Finished Solving R-GEP -------------\n');

%% Compare against eig
% Matlab eig function
[V_eig, D_eig] = eig(M);
D_eig = diag(D_eig(end-p+1:end,end-p+1:end));
V_eig = V_eig(:,end-p+1:end);

f_rgep = f(M, Vout);    % rgep objective value
f_eig = f(M, V_eig);    % eig objective value
diff = abs(f_rgep - f_eig);

fprintf(1, '-------------- Comparison: RGEP vs. eig -------------- \n');
fprintf(1, 'RGEP f(V): %f\neig f(V): %f\ndiff: %f\n', f_rgep, f_eig, diff);









