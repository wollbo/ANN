%% 3.2 Classification and Regression

clear all
close all
mu = [1 0.3;-1 0.3;0 -0.1];%;-2 -5];
sigma = [0.2 0.2;0.2 0.2;0.3 0.3];%; 2 6];
%mu = [1 2;5 7];%;-2 -5];
%sigma = [1 0.8;0.5 2];%; 2 6];
datapoints = 500;


[data1, target1] = generateData(datapoints,[mu(1,:); mu(3,:)],[sigma(1,:); sigma(3,:)]);
[data2, target2] = generateData(datapoints,[mu(2,:); mu(3,:)],[sigma(1,:); sigma(3,:)]);

data = [data1; data2];
target = [target1; target2];

nodes1 = 20;
nodes2 = 1;
inputs = 2;
W = 0.001*randn(nodes1,inputs+1);
V = 0.001*randn(nodes2,nodes1);
eta = 0.0001;
outputs = 1;
alpha = 0.9;
data1 = data1';
data2 = data2';
target1 = target1';
target2 = target2';
data = [data1 data2];
target = [target1 target2];
epochs = 2000;
X = data;
nData = length(X);
X = [X;ones(1,nData)];
t = target;

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
    % add tError as (W*data-targets)

end
plot(error)