function grad = partial_trace_grad(M, V, K, alpha, alpha_idx, lambda)
% partial gradient of a trace function

n = size(M, 1);
Kc = setdiff(1:n, K);
grad = M(K, K) * V(K, :) + M(Kc, K)' * V(Kc, :);
grad = 2*grad;

V_alpha = zeros(size(V,1), 1);
V_alpha(alpha_idx) = alpha;
V_temp = V;
V_temp(~alpha_idx,:) = 0;
grad(:,1) = grad(:,1) - lambda*sign(V_alpha(K,1) - V_temp(K,1));