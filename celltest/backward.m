function delta = backward(targets,weightMatrix,output,activation)
% Input: Target, Output
% Output: delta maxtrix, each column represents a layer
depth = size(weightMatrix,2);
delta = cell(size(activation));
for i = 1:length(delta)
    delta{i} = zeros(size(activation{i}));
end
[~,dY] = sigmoid(activation{end});
delta{end} = (output - targets) .* dY;
for k = 1:depth-1
    size(delta{end})
    size(weightMatrix{end})
    size(delta{end})
    delta{end-k} = backwardGeneral(activation{end-k},weightMatrix{end-k},delta{end-k+1});
end

end
