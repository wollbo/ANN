function [activation, derivative] = sigmoid(x)
% sigmoid activation function
activation = 1./(1+exp(-x));
derivative = activation.*(1-activation);
end

