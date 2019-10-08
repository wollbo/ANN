%% init weight matrix and input

layers = 1;
nodes = 2;
inputs = 2;
outputs = 1;
nData = 100;

W = randn(inputs,nodes+1); %+1 for bias term
V = randn(outputs,nodes);

X = randn(2,nData);
X = [X;ones(1,nData)];

t = randn(1,nData);

dw = zeros(size(W));
dv = zeros(size(V));
eta = 0.0001;
alpha = 0.9;
%%
%forward

[a1, z1] = forwardGeneral(W,X);
[a2, z2] = forwardGeneral(V,z1);

%%
%backward
[~,dY] = sigmoid(a2(:,end));
delta2 = (z2 - t) .* dY;
delta1 = backwardGeneral(a2,V,delta2);
delta1 = delta1(1:end-1,:); % unclear if necessary

%%
%update

dw = updateGeneral(dw,eta,alpha,delta1,X);
dv = updateGeneral(dv,eta,alpha,delta2,z1);
W = W+dw;
V = V+dv;


