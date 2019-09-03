function weightMatrixDelta = update(weightMatrix,eta,alpha,delta,output,weightMatrixDelta)
% weightMatrix update for one layer at a time
% contains the weights for all layers
% update over sum of errors for this layer
depth = size(weightMatrix,3);
for k = 1:depth
    weightMatrixDelta(:,:,k) = -eta*(alpha*weightMatrixDelta(:,:,k)-(1-alpha)*delta*output'); %-eta*delta*output'; simple case
end

end