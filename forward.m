function yOut = forward(X,W,V)
%FORWARD naive forward algorithm for two layer perceptron
% X matrix: N rows (inputs) M columns (observations) input
% W, V needs proper initialization w.r.t. X and bias terms

nData = size(X,2);
bias = ones(1,nData);

hIn = W*[X;bias];
hOut = sigmoid(hIn);
yIn = V*hOut;
yOut = sigmoid(yIn);

% Note: W,V could be combined in a larger matrix/tensor

end

