%% uppg4
% non linearly separable data

%Generate non linearly separable data
close all
clear all
mu = [0 3;4 0];%;-2 -5];
sigma = [1 1;2 2];%; 2 6];
%mu = [1 2;5 7];%;-2 -5];
%sigma = [1 0.8;0.5 2];%; 2 6];
datapoints = 50;

[data, target] = generateData(datapoints,mu,sigma);

dataSplit1 = data(target==1,:);
dataSplit2 = data(target==-1,:);

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
epochs = 100;
guess = zeros(epochs,length(t));
error = zeros(epochs,1);


%%
scatter(dataSplit1(:,1),dataSplit1(:,2))
hold on
scatter(dataSplit2(:,1),dataSplit2(:,2))
hold off

%% Compare perceptron learning with delta learning in this case:

for k = 1:epochs
dW = -eta*(W*X-t)*X';
W = W+dW;
guess(k,:) = sign(W*X);
error(k) = mean((guess(k,:)-t).^2);
end

plot(error)

[~,I] = min(error)
bestGuess = guess(I,:);
%% Delta learning

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


 

