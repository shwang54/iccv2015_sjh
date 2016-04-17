function V = RGEP(M, f, grad, D, V0, opts)
% V = RGEP(M, f, grad, B, V0, opts)
%
% Solver for Projection Free Generalzed Eigenvalue Problem with a nonsmooth
% Regularizer (SJH, ICCV 2015)
%
% Input:
%   M: Primary matrix M
%   f: objective function handle
%   grad: gradient function handle
%   D: Secondary matrix D
%   V0: initial eigenvectors
%   opts: parameter struct (with default values)
%       opts.rho1 = 0.01;           % Wolfe condition c1 - step length (Armijo)
%       opts.rho2 = 0.99;           % Wolfe condition c2 - curvature
%       opts.grad_epsilon = 1e-7;   % terminate when gradient change less than epsilon
%       opts.print_interval = 100;  % print outputs interval
%       opts.max_iter = 20000;      % maximum iteration
%       opts.k = 30;                % submatrix size i (Eq.4, i rows of submatrix V)
%
% Output:
%   V: solved eigenvectors
% 
% REFERENCES
%   Seong Jae Hwang (ICCV 2015)
%   http://pages.cs.wisc.edu/~sjh
%
%   Seong Jae Hwang
%   sjh@cs.wisc.edu
%   University of Wisconsin, Madison
%--------------------------------------------------------------------------

    % Interpret opts argument
    if ~exist('opts', 'var')
        opts = struct();
        defaults.roh1 = 0.01;           % Wolfe condition c1 - step length (Armijo)
        defaults.roh2 = 0.99;           % Wolfe condition c2 - curvature
        defaults.grad_epsilon = 1e-7;   % terminate when gradient change less than epsilon
        defaults.print_interval = 100;  % print outputs interval
        defaults.max_iter = 20000;      % maximum iteration
        defaults.k = 30;                % submatrix size i (Eq.4, i rows of submatrix V)
        opts = optdefaults(opts, defaults);
    end
    
    [n, p] = size(V0);
    
    M0 = M;
    
    k = opts.k;
    V = V0;
    
    % Geodesic curves on (generalized) stiefel manifold
    function X = Y(W, M, X0, tau)
        tWM = (tau/2)*W*M;
        m = size(tWM, 1);
        Q = (eye(m) + tWM) \ (eye(m) - tWM);
        X = Q*X0;
    end

    % Line search objective function
    function incobj = line_search(K, Kc, ind, dep, R, U)
        Vinc = zeros(n, p);
        Vinc(Kc, :) = V(Kc, :);
        Vinc(K, :) = D(K, K) \ (D(Kc, K)' * V(Kc, :));
        Vinc(K, ind) = U - Vinc(K, ind);
        Vinc(K, dep) = U * R - Vinc(K, dep);
        incobj = f(M, Vinc);
    end
    %% %%%%%%%%%%%%%%%%%%%% Iterations %%%%%%%%%%%%%%%%%%%%%%%%%%%
    dF = inf;
    iter = 0;
    K0 = randperm(n);
    time = tic;
    
    % theoretically we should check |dF|, but it is not smooth so better to
    % just let it run for many iterations
    while 1 % abs(dF) > opts.grad_epsilon
       
        % permute indices and select sumatrix of size k without replacement
        if length(K0) <= k
            K = K0(1:end);
            Kc = setdiff(1:n, K);
            K0 = randperm(n);
        else
            K = K0(1:k);
            Kc = setdiff(1:n, K);
            K0(1:k) = [];
        end
        
        % Uall is the matrix whos orthogonality wrt B(K,K) we must preserve
        VKoff = D(K, K) \ (D(Kc, K)' * V(Kc, :));
        Uall = V(K, :) + VKoff;
        
        % Singularity correction
        % Get linearly independent columns of Uall
        [R, ind] = rref(Uall, 1e-8);    % reduced row echelon form
        dep = setdiff(1:p, ind);
        R = R(ind, dep);
        U = Uall(:, ind);
        
        % Determine descent direction
        G = grad(M, V, K);
        G = G(:, ind) + G(:, dep) * R';
        
        % skew symmetric matrix and Cayley transform line search
        W = G*U' - U*G';
        fl = @(U) line_search(K, Kc, ind, dep, R, U);
        
        obj = fl(U);
        dF = -1/2 * norm(W, 'fro')^2;   % gradient magnitude
        
        
        tau = 5;    % initial step size tau
        
        % line search under strong Wolfe condition
        while 1
            if fl(Y(W, D(K, K), U, tau)) <= obj + opts.rho1 * tau * dF
                G2 = grad(M, V, K);
                G2 = G2(:, ind) + G2(:, dep) * R';
                W22 = G2*U' - U*G2';
                
                if abs(fl(Y(W22, D(K, K), U, tau))) >= opts.rho2 * dF
                    break;
                end
            end
            tau = tau/2;

            if tau < 1e-11
                break
            end
        end
        
        if tau < 1e-11
            continue
        end
        
        % Take step
        U = Y(W, D(K, K), U, tau);
        V(K, ind) = U;
        V(K, dep) = U * R;
        V(K, :) = V(K, :) - VKoff;
        iter = iter + 1;
        
        % print out
        if opts.print_interval > 0 && (mod(iter, opts.print_interval) == 0 || opts.print_interval == 1)
            obj = f(M, V);
            
            fprintf(1, 'Iter %3d:: obj: %.4e, tau: %.2e, |dF|: %.4e  runtime: %f sec\n', ...
                iter, obj, tau, abs(dF), toc(time));
        end
        
        if iter >= opts.max_iter
            break
        end
    end
    
    obj = f(M0, V); % final objective value
  
    fprintf(1, 'Finished. Final iteration:\nIter %3d:: obj: %.4e, tau: %.2e, |dF|: %.4e  runtime: %f sec\n', ...
                iter, obj, tau, abs(dF), toc(time));
end