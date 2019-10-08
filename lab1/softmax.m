function activation = softmax(x)
% softmax activation function
% x vector input of K numbers
K = length(x);
activation = zeros(K,1);
norm = sum(exp(-x));
for k = 1:K
activation(k) = exp(-x(k))./norm;
end

