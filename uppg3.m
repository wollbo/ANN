%% uppg3
% 

%Generate Data
mu = [1 2;5 7];%;-2 -5];
sigma = [1 0.8;0.5 2];%; 2 6];
datapoints = 50;

[data target] = generateData(datapoints,mu,sigma);

%%

X = data';
t = target';
%t(t==-1) = 0; % for sigmoid
nData = length(X);
%X = [X;ones(1,nData)];
nodes = 1;
inputs = 2;
W = 0.01*randn(nodes,inputs);
eta = 0.0001;
outputs = 1;
alpha = 0.9;

%%

dw = zeros(size(W));
y = zeros(size(t));

 
 for k = 1:10000
     for j = 1:length(X)
        hin = W * X(:,j);
        hout = [2 ./ (1+exp(-hin)) - 1 ];
        y(j) = hout;
 
        delta_o = (hout - t(j)) .* ((1 + hout) .* (1 - hout)) * 0.5;
        %delta_o = delta_o(1:nodes, :);
 
        dw = (dw .* alpha) - (delta_o * X(:,j)') .* (1-alpha);
        W = W + dw .* eta;
        
     end
 end
 
guess = sign(hout);
error = mean(y-t);