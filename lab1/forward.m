function [activation,output] = forward(weightMatrix,input)
%FORWARD general forward algorithm for multi layer perceptron
nData = size(input,2); % number of observations
%bias = ones(1,nData); should be added outside this function
%input = [input;bias];
depth = size(weightMatrix,3);
activation = zeros(size(input,1),nData,depth);
output = activation;
for k = 1:depth
    [activation(:,:,k), output(:,:,k)] = forwardGeneral(weightMatrix(:,:,k),input);
    input = output(:,:,k);
end

end

