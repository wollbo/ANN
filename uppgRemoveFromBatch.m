%% uppg1

clear all
close all

%Generate Data
mu = [1 0.3;0 -0.1];%;-2 -5];
sigma = [0.2 0.2;0.3 0.3];%; 2 6];

% mu = [1 1;5 5];%;-2 -5];
% sigma = [1 1;1 1];%; 2 6];

datapoints = 100;

[data1 target1] = generateData(datapoints,mu(1,:),sigma(1,:),1);
[data2 target2] =generateData(datapoints,mu(2,:),sigma(2,:),[-1]);

data = [data1;data2];
target = [target1;target2];

data_25 = [data1(1:75,:);data2(1:75,:)];
target_25 = [target1(1:75);target2(1:75)];

data_50A = [data1(1:50,:);data2];
target_50A = [target1(1:75);target2(1:75)];

data_50B = [data1;data2(1:50,:)];
target_50B = [target1(1:75);target2(1:75)];

data_less_ind = find(data1(:,2)<0);
data_less = data1(data_less_ind,:);
t_less = target1(data_less_ind);
data_less(1:ceil(length(data_less)*0.2),:) = [];
t_less(1:ceil(length(data_less)*0.2)) = [];


data_more_ind = find(data1(:,2)>0);
data_more = data1(data_more_ind,:);
t_more = target1(data_more_ind);
data_more(1:ceil(length(data_more)*0.8),:) = [];
t_more(1:ceil(length(data_more)*0.8)) = [];


data_2080A = [data1(1:75,:);data2(1:75,:)];
target_2080A = [target1(1:75);target2(1:75)];


%scatter(data(:,1),data(:,2))
%%

X = data';
t = target';
nData = length(X);
X = [X;ones(1,nData)];

X_25 = data_25';
t_25 = target_25';
nData = length(X_25);
X_25 = [X_25;ones(1,nData)];

X_50A = data_50A';
t_50A = target_50A';
nData = length(X_50A);
X_50A = [X_50A;ones(1,nData)];

X_50B = data_50B';
t_50B = target_50B';
nData = length(X_50B);
X_50B = [X_50B;ones(1,nData)];

X_2080A = data_2080A';
t_2080A = target_2080A';
nData = length(X_2080A);
X_2080A = [X_2080A;ones(1,nData)];

nodes = 1;
inputs = 2;

W = 0.01*randn(nodes,inputs+1);
W_25 = W;
W_50A = W;
W_50B = W;
W_2080A = W;
eta = 0.0001;
outputs = 1;
alpha = 0.9;
epochs = 1000;

guess = zeros(epochs,length(t));
guess_25 = zeros(epochs,length(t_25));
guess_50A = zeros(epochs,length(t_50A));
guess_50B = zeros(epochs,length(t_50B));
guess_2080A = zeros(epochs,length(t_2080A));

error = zeros(epochs,1);
error_25 = error;
error_50A = error;
error_50B = error;
error_2080A = error;



%% 
% All data
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
 
W_all = W; 
%%
% 25% from each class
dw = zeros(size(W_25));
error = zeros(epochs,1);
 
 for k = 1:epochs
 hin = W_25 * X_25;
 hout = [2 ./ (1+exp(-hin)) - 1 ];
 
 delta_o = (hout - t_25) .* ((1 + hout) .* (1 - hout)) * 0.5;
 delta_o = delta_o(1:nodes, :);
 
 dw = (dw .* alpha) - (delta_o * X_25') .* (1-alpha);
 W_25 = W_25 + dw .* eta;
 
guess_25(k,:) = sign(hout);
error_25(k) = mean(mean((guess_25(k,:)-t_25).^2));

 end
  

%%
% 50% from class A
dw = zeros(size(W_50A));
error = zeros(epochs,1);
 
 for k = 1:epochs
 hin = W_50A * X_50A;
 hout = [2 ./ (1+exp(-hin)) - 1 ];
 
 delta_o = (hout - t_50A) .* ((1 + hout) .* (1 - hout)) * 0.5;
 delta_o = delta_o(1:nodes, :);
 
 dw = (dw .* alpha) - (delta_o * X_50A') .* (1-alpha);
 W_50A = W_50A + dw .* eta;
 
guess_50A(k,:) = sign(hout);
error_50A(k) = mean((guess_50A(k,:)-t_50A).^2);
 
 end

 %%
% 50% from class B
dw = zeros(size(W_50B));
error = zeros(epochs,1);
 
 for k = 1:epochs
 hin = W_50B * X_50B;
 hout = [2 ./ (1+exp(-hin)) - 1 ];
 
 delta_o = (hout - t_50B) .* ((1 + hout) .* (1 - hout)) * 0.5;
 delta_o = delta_o(1:nodes, :);
 
 dw = (dw .* alpha) - (delta_o * X_50B') .* (1-alpha);
 W_50B = W_50B + dw .* eta;
 
guess_50B(k,:) = sign(hout);
error_50B(k) = mean((guess_50B(k,:)-t_50B).^2);
 
 end

 %%
% 20% of A<0 and 80% of A> 0 removed
dw = zeros(size(W_2080A));
error = zeros(epochs,1);
 
 for k = 1:epochs
 hin = W_2080A * X_2080A;
 hout = [2 ./ (1+exp(-hin)) - 1 ];
 
 delta_o = (hout - t_2080A) .* ((1 + hout) .* (1 - hout)) * 0.5;
 delta_o = delta_o(1:nodes, :);
 
 dw = (dw .* alpha) - (delta_o *X_2080A') .* (1-alpha);
 W_2080A = W_2080A + dw .* eta;
 
guess_2080A(k,:) = sign(hout);
error_2080A(k) = mean((guess_2080A(k,:)-t_2080A).^2);
 
 end


%%
f2 = figure('Name','figures/uppgBatchRemoval');

x_dec = [min(data(:,1))-0.5 max(data(:,2))+0.75];
y_dec = - (x_dec*(W(1)/W(2)) + (W(3)/W(2)));
plot(x_dec,y_dec,'-.','Linewidth',2); hold on;

x_dec_25 = [min(data_25(:,1))-0.5 max(data_25(:,2))+0.75];
y_dec_25 = - (x_dec_25*(W_25(1)/W_25(2)) + (W_25(3)/W_25(2)));
plot(x_dec_25,y_dec_25,'-.','Linewidth',2)

x_dec_50A = [min(data_50A(:,1))-0.5 max(data_50A(:,2))+0.75];
y_dec_50A = - (x_dec_50A*(W_50A(1)/W_50A(2)) + (W_50A(3)/W_50A(2)));
plot(x_dec_50A,y_dec_50A,'-.','Linewidth',2)

x_dec_50B = [min(data_50B(:,1))-0.5 max(data_50B(:,2))+0.75];
y_dec_50B = - (x_dec_50B*(W_50B(1)/W_50B(2)) + (W_50B(3)/W_50B(2)));
plot(x_dec_50B,y_dec_50B,'-.','Linewidth',2)

x_dec_2080A = [min(data_2080A(:,1))-0.5 max(data_2080A(:,2))+0.75];
y_dec_2080A = - (x_dec_2080A*(W_2080A(1)/W_2080A(2)) + (W_2080A(3)/W_2080A(2)));
plot(x_dec_2080A,y_dec_2080A,'--','Linewidth',2)


axis([min(data_25(:,1))-0.5 max(data_25(:,1)+0.75) min(data_25(:,2))-0.25 max(data_25(:,2))+0.75])

scatter(data1(:,1),data1(:,2))
scatter(data2(:,1),data2(:,2))
title('Data distribution and decision boundaries')
xlabel('x1')
ylabel('x2')
% plot(0,0,'*','linewidth',4)
% legend('Perceptron decision boundary', 'Delta Learning rule decision boundary', 'Data Cluster 1', 'Data Cluster 2', 'location', 'northwest')
legend('Full dataset','25% removed from both datasets', '50% removed from dataset A','50% removed from dataset B','20% of A<0 and 80% A>0 removed', 'location', 'northeast')

error_rate = sum(guess(end,:) ~= t)/length(t)
error_rate_25 = sum(guess_25(end,:) ~= t_25)/length(t_25)
error_rate_50A = sum(guess_50A(end,:) ~= t_50A)/length(t_50A)
error_rate_50B = sum(guess_50B(end,:) ~= t_50B)/length(t_50B)
error_rate_2080A = sum(guess_2080A(end,:) ~= t_2080A)/length(t_2080A)
