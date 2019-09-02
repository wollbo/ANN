function delta = backward(targets,Y,V,H)
% Input: Target, Output
% Output: delta maxtrix, each column represents a layer
[~,dY] = sigmoid(Y);
[~,dH] = sigmoid(H);

delta_o = (Y - targets) .* dY;
delta_h = (V * delta_o) .* dH;
delta_h = delta_h(1:end-1,:);

delta = [delta_h,delta_o];


end

