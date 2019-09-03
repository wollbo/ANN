function delta = backward(targets,weightMatrix,output,activation)
% Input: Target, Output
% Output: delta maxtrix, each column represents a layer
depth = size(weightMatrix,3);
delta = zeros(size(output,1),size(output,2),depth);
[~,dY] = sigmoid(activation(:,end));
delta(:,:,end) = (output - targets) .* dY;
for k = 1:depth-1
    delta(:,:,end-k) = backwardGeneral(activation(:,end-k),weightMatrix(:,:,end-k),delta(:,:,end-k+1));
end

