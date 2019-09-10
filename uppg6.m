%% 3.2.1 Classification and Regression

clear all
close all
mu = [1 0.3;-1 0.3;0 -0.1];%;-2 -5];
sigma = [0.2 0.2;0.2 0.2;0.3 0.3];%; 2 6];
%mu = [1 2;5 7];%;-2 -5];
%sigma = [1 0.8;0.5 2];%; 2 6];
datapoints = 50;


[data1, target1] = generateData(datapoints,[mu(1,:); mu(3,:)],[sigma(1,:); sigma(3,:)]);
[data2, target2] = generateData(datapoints,[mu(2,:); mu(3,:)],[sigma(1,:); sigma(3,:)]);

data = [data1; data2];
target = [target1; target2];
data1 = data1';
data2 = data2';
target1 = target1';
target2 = target2';
data = [data1 data2];
target = [target1 target2];

%%

nodes1 = 10;
nodes2 = 1;
inputs = 2;
W = 0.01*randn(nodes1,inputs+1);
V = 0.01*randn(nodes2,nodes1);
eta = 0.01;
outputs = 1;
alpha = 0.9;
epochs = 1000;
X = data;
nData = length(X);
X = [X;ones(1,nData)];
t = target;

%%

dataSplit1 = data(:,target==1); % bara för viualisering
dataSplit2 = data(:,target==-1);
scatter(dataSplit1(1,:),dataSplit1(2,:))
hold on
scatter(dataSplit2(1,:),dataSplit2(2,:))
hold off

%% Remove:
% • random 25% from each class

data = [data1(:,1:floor(length(data1)*0.75)) data2(:,1:floor(length(data2)*0.75))];
test_data = [data1(:,ceil(length(data1)*0.75):end) data2(:,ceil(length(data2)*0.75):end)];

target = [target1(1:floor(length(data1)*0.75)) target2(1:floor(length(data2)*0.75))];
test_target = [target1(ceil(length(data1)*0.75):end) target2(ceil(length(data2)*0.75):end)];

X_train = data;
X_test = test_data;
t_train = target;
t_test = test_target;

nData_train = length(X_train);
nData_test = length(X_test);

X_train = [X_train;ones(1,nData_train)];
X_test = [X_test;ones(1,nData_test)];

X = X_train;
t = t_train;

%%
dw = zeros(size(W));
dv = zeros(size(V));
 
for k = 1:epochs
    [a1,z1] = forwardGeneral(W,X);
    [a2,z2] = forwardGeneral(V,z1);
    
    [~,dY] = sigmoid2(a2); 
    delta2 = (z2-t).*dY;
    delta1 = backwardGeneral(a2,V,delta2);

    dw = updateGeneral(dw,eta,alpha,delta1,X);
    dv = updateGeneral(dv,eta,alpha,delta2,z1);

    dw = (dw .* alpha) - (delta1 * X') .* (1-alpha);
    dv = (dv .* alpha) - (delta2 * z1') .* (1-alpha);
    W = W + dw .* eta;
    V = V + dv .* eta;

    guess(k,:) = sign(z2);
    error(k) = mean((guess(k,:)-t).^2);
    rate(k) = 1-length(target(target==guess(k,:)))/length(target);
    % add tError as (W*data-targets)

end
plot(rate) % do for 3/4 different datasets
hold on

%%

f1 = figure('Name', 'figures/32error')
hold on
plot(rate)
ylabel('Error Rate')
xlabel('Epoch')

%%
legend('N=2','N=5','N=10');
