%% uppg1

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
%t(t==-1) = 0; % for sigmoid
nData = length(X);
X = [X;ones(1,nData)];
nodes = 1;
inputs = 2;
W = 0.01*randn(nodes,inputs+1);
eta = 0.0001;
outputs = 1;
alpha = 0.9;
epochs = 10000;
guess = zeros(epochs,length(t));
error = zeros(epochs,1);
%%
% Perceptron learning rule
for k = 1:epochs
dW = -eta*(W*X-t)*X';
W = W+dW;
guess(k,:) = sign(W*X);
error(k) = mean((guess(k,:)-t).^2);
end

 plot(error)
%% 
% Delta learning rule

 dw = zeros(size(W));
 
 for k = 1:epochs
 hin = W * X;
 hout = [2 ./ (1+exp(-hin)) - 1 ];
 
 delta_o = (hout - t) .* ((1 + hout) .* (1 - hout)) * 0.5;
 delta_o = delta_o(1:nodes, :);
 
 dw = (dw .* alpha) - (delta_o * X') .* (1-alpha);
 W = W + dw .* eta;
 
guess(k,:) = sign(hout);
error(k) = mean((guess(k,:)-t).^2);
 
 end
 
plot(error)