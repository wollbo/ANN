%% main
clear all
close all

%Generate Data
mean = [1 2;5 7]%;-2 -5];
sigma = [1 0.8;0.5 2]%; 2 6];
datapoints = 100;

[data target] = generateData(datapoints,mean,sigma);

%scatter(data(:,1),data(:,2))
%%

data = data'
target = target'
%%

nodes = 2;
layers = 2;
weightMatrix = 0.01*rand(nodes,2,layers);
weightMatrixDelta = zeros(size(weightMatrix));
alpha = 0.9;
eta = 0.1;

for k = 1:100000
%Forward Pass
    [a,y] = forward(weightMatrix,data);
%Error Backpropagation
    delta = backward(target,weightMatrix,y(:,:,end),a);
%Weight update
    weightMatrixDelta = update(weightMatrix,eta,alpha,delta,y,weightMatrixDelta);
    weightMatrix = weightMatrix - weightMatrixDelta
end
