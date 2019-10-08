%% main
clear all
close all

%Generate Data
mu = [1 2;5 7];%;-2 -5];
sigma = [1 0.8;0.5 2];%; 2 6];
datapoints = 50;

[data target] = generateData(datapoints,mu,sigma);

%scatter(data(:,1),data(:,2))
%%

X = data';
t = target';
t(t==-1) = 0; % for sigmoid
nData = length(X);
X = [X;ones(1,nData)];
%%

%layers = 2;
nodes = 1;
inputs = 2;
outputs = 1;

W = 0.01*randn(nodes,inputs+1); %+1 for bias term
V = 0.01*randn(outputs,nodes);

dw = zeros(size(W));
dv = zeros(size(V));
eta = 0.001;
alpha = 0; %0.9
%f = plot(W(1), W(2), '*')

%%

for k = 1:100
[a1, z1] = forwardGeneral(W,X);
%z1 = [z1;ones(1,nData)]; ??? add bias term?
[a2, z2] = forwardGeneral(V,z1);
%
%backward
[~,dY] = sigmoid(a2);
delta2 = (z2 - t) .* dY;
delta1 = backwardGeneral(a2,V,delta2);
delta1 = delta1(1:nodes,:); % unclear if necessary

%
%update

dw = updateGeneral(dw,eta,alpha,delta1,X);
dv = updateGeneral(dv,eta,alpha,delta2,z1);
W = W+dw;
V = V+dv;

%drawnow
%pause(0.1)

end

guess = round(z2);

