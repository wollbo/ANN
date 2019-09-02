function Y = forward(X,W,V)
%FORWARD naive forward algorithm for two layer perceptron
% X matrix: N rows (inputs) M columns (observations) input
% W, V needs proper initialization w.r.t. X and bias terms

nData = size(X,2);
bias = ones(1,nData);

H = sigmoid(W*[X;bias]);
Y = sigmoid(V*H);

% Note: H, O could be combined in a larger matrix/tensor 

end

