%% uppg1

clear all
close all

%Generate Data
mu = [1 1;3.5 3.5];%;-2 -5];
sigma = [1 1;1 1];%; 2 6];
datapoints = 100;

[data1 target1] = generateData(datapoints,mu(1,:),sigma(1,:),1);
[data2 target2] =generateData(datapoints,mu(2,:),sigma(2,:),[-1]);

data = [data1;data2];
target = [target1;target2];

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
W_saved = W;
eta = 0.0001;
outputs = 1;
alpha = 0.9;
epochs = 1000;
guess = zeros(epochs,length(t));
error = zeros(epochs,1);
%%
% Perceptron learning rule
for k = 1:epochs
dW = -eta*(sign(W*X)-t)*X';
W = W+dW;
guess(k,:) = sign(W*X);
error(k) = mean((guess(k,:)-t).^2);
end
W_perc = W;

plot(error)
 hold on
%% 
% Delta learning rule
W = W_saved;
dw = zeros(size(W));
error = zeros(epochs,1);
 
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
 
W_delta = W; 
plot(error)
title('Learning Curves')
legend('Perceptron','Delta Learning Rule')
xlabel('Epochs')
ylabel('MSE')

figure()

x_dec = [min(data(:,1))-0.5 max(data(:,2))+0.75];
y_dec_perc = - (x_dec*(W_perc(1)/W_perc(2)) + (W_perc(3)/W_perc(2)));
y_dec_delta = - (x_dec*(W_delta(1)/W_delta(2)) + (W_delta(3)/W_delta(2)));
% plot(x_dec,y_dec_perc)
% plot(x_dec,y_dec_delta)
axis([min(data(:,1))-0.5 max(data(:,1)+0.75) min(data(:,2))-0.25 max(data(:,2))+0.75])
hold on
plot(x_dec,y_dec_perc,'-.','Linewidth',2)
plot(x_dec,y_dec_delta,'-.','Linewidth',2)
scatter(data1(:,1),data1(:,2))
scatter(data2(:,1),data2(:,2))
title('Data distribution and decision boundaries')
xlabel('x1')
ylabel('x2')
legend('Perceptron decision boundary', 'Delta Learning rule decision boundary', 'Data Cluster 1', 'Data Cluster 2', 'location', 'northwest')