%% uppg2
% sequential vs batch learning for the delta rule.
clear all
close all

%Generate Data
mu = [1 1;8 8];%;-2 -5];
sigma = [1 0.8;0.5 2];%; 2 6];
datapoints = 100;

[data target] = generateData(datapoints,mu,sigma,[1 -1]);

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
epochs =300;
%%

dw = zeros(size(W));
y = zeros(size(t));
%error = zeros(epochs,length(X));
%error_temp = 0;

 for k = 1:epochs
     for j = 1:length(X)
        hin = W * X(:,j);
        hout = [2 ./ (1+exp(-hin)) - 1 ];
        y(j) = hout;
        
        delta_o = (hout - t(j)) .* ((1 + hout) .* (1 - hout)) * 0.5;
        delta_o = delta_o(1:nodes, :);
 
        dw = (dw .* alpha) - (delta_o * X(:,j)') .* (1-alpha);
        W = W + dw .* eta;
        
        guess(k,j) = sign(hout);
        error_temp = (guess(k,1:j)-t(1:j)).^2;
        error(k,j) = mean(error_temp);      
     end
     error_epochs(k) = mean((guess(k,:)-t).^2);
 end
 
% for i = 1:length(error(:))
%     error_OL(i) = mean(error(1:i));
% end

f1 = figure('Name','figures/uppgbatchonline')

error = error';
error = error(:);
plot((1:epochs*length(X))/length(X),error)
% plot((1:epochs*length(X))/length(X),error_OL)
hold on
plot(1:epochs,error_epochs,'-.','linewidth',2)
title('Learning Curves')
xlabel('Epochs')
ylabel('MSE')
legend('Online Learning','Batch Learning')
x_dec = [min(data(:,1))-0.5 max(data(:,2))+0.75];
y_dec_delta = - (x_dec*(W(1)/W(2)) + (W(3)/W(2)));