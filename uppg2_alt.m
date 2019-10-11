%% uppg2
% sequential vs batch learning for the delta rule.
clear all
close all

%GAMMAL Generate Data
% mu = [1 0.5;-1 0];%;-2 -5];
% sigma = [0.5 0.5;0.5 0.5];%; 2 6];
% datapoints = 100;
% 
% [data target] = generateData(datapoints,mu,sigma,[1 -1]);
% 
% X = data';
% t = target';
%HIT

n = 100;
mA = [ 1.0, 0.5]*1.3; sigmaA = 0.5;
mB = [-1.0, 0.0]*1.3; sigmaB = 0.5;
classA(1,:) = randn(1,n) .* sigmaA + mA(1);
classA(2,:) = randn(1,n) .* sigmaA + mA(2);
classB(1,:) = randn(1,n) .* sigmaB + mB(1);
classB(2,:) = randn(1,n) .* sigmaB + mB(2);

data1 = classA';
data2 = classB';

data = [classA';classB'];
target = [ones(length(classA),1);-ones(length(classB),1)];

X = data';
t = target';

%t(t==-1) = 0; % for sigoid
nData = length(X);
X = [X;ones(1,nData)];
nodes = 1;
inputs = 2;
W = 0.1*randn(nodes,inputs+1);
% W = 1*randn(nodes,inputs+1);
W_OL = W;
W_B = W;
eta = 0.001;
outputs = 1;
alpha = 0.9;
epochs =20;


dw = zeros(size(W_OL));
y = zeros(size(t));
%error = zeros(epochs,length(X));
%error_temp = 0;

 for k = 1:epochs
     for j = 1:length(X)
        dw = -eta*(W_OL*X(:,j)-t(j))*X(:,j)';
        W_OL = W_OL + dw;

        hout= W_OL*X;
        guess_OL = sign(hout);
        error_OL_temp(k,:) = ((guess_OL-t).^2);
     end
 end
% error_OL_temp = error_OL_temp';
error_OL = mean(error_OL_temp');
%%
dw = zeros(size(W_B));
y = zeros(size(t)); 
 
for k = 1:epochs
%     hin = W_B * X;
%     hout = [2 ./ (1+exp(-hin)) - 1 ];
%     y = hout;
% 
%     delta_o = (hout - t) .* ((1 + hout) .* (1 - hout)) * 0.5;
%     delta_o = delta_o(1:nodes, :);
% 
%     dw = (dw .* alpha) - (delta_o * X') .* (1-alpha);
%     W_B = W_B + dw .* eta;
%     
%     guess_B(k,:) = sign(hout);
%     error_B(k) = mean((guess_B(k,:)-t).^2);
 
%     hout = W_B * X;
    dw = -eta*(W_B*X-t)*X';
    W_B = W_B + dw;
    
    guess_B(k,:) = sign(W_B*X);
    error_B(k) = mean((guess_B(k,:)-t).^2);
    
end
        



f1 = figure('Name','figures/uppgbatchonlineS_alt')

% error_OL = mean(error_OL');

% error_OL_temp = error_OL_temp;
% plot((1:epochs*length(X))/length(X),error_OL_temp(:))

plot(1:epochs,error_OL, 'linewidth',2)

hold on
plot(1:epochs,error_B,'-.','linewidth',2)
title('Learning Curves')
xlabel('Epochs')
ylabel('MSE')
legend('Online Learning','Batch Learning')
x_dec = [min(data(:,1))-0.5 max(data(:,2))+0.75];
y_dec_delta = - (x_dec*(W(1)/W(2)) + (W(3)/W(2)));