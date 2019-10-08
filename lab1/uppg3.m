%% uppg3
% delta rule without bias
clear all
close all
%Generate Data
mu = [1 2;5 7];%;-2 -5];
sigma = [1 0.8;0.5 2];%; 2 6];
datapoints = 50;

[data1 target1] = generateData(datapoints,mu(1,:),sigma(1,:),1);
[data2 target2] =generateData(datapoints,mu(2,:),sigma(2,:),[-1]);

data = [data1;data2];
target = [target1;target2];

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
epochs  = 10000;

%%

dw = zeros(size(W));
y = zeros(size(t));

 
 for k = 1:epochs
        hin = W * X;
        hout = [2 ./ (1+exp(-hin)) - 1 ];
        y = hout;
 
        delta_o = (hout - t) .* ((1 + hout) .* (1 - hout)) * 0.5;
        %delta_o = delta_o(1:nodes, :);
 
        dw = (dw .* alpha) - (delta_o * X') .* (1-alpha);
        W = W + dw .* eta;
        
        guess(k,:) = sign(hout);
        error(k) = mean((guess(k,:)-t).^2);

 end
 
plot(error)
axis([0 length(error) 0 max(error)])
title('Learning Curve')
legend('Delta Learning Rule without bias')
xlabel('Epochs')
ylabel('MSE')
figure()
x_dec = [min(data(:,1))-0.5 max(data(:,2))+0.75];
y_dec_delta = - (x_dec*(W(1)/W(2)));
plot(x_dec,y_dec_delta)
hold on
scatter(data1(:,1),data1(:,2))
scatter(data2(:,1),data2(:,2))
axis([min(data(:,1))-0.5 max(data(:,1)+0.75) min(data(:,2))-0.25 max(data(:,2))+0.75])
xlabel('x1')
ylabel('x2')

legend('Delta Learning rule decision boundary wihout bias', 'Data Cluster 1', 'Data Cluster 2', 'location', 'northwest')