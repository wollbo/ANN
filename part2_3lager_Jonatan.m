%% generate data
clear all
close all

tau = 25;
N = 1500+tau;
beta = 0.2;
gamma = 0.1;
n = 10;
x = zeros(N,1);
x(tau) = 1.5;

for i = tau+1:N-1
    x(i+1) = x(i)+beta*x(i-tau)/(1+x(i-tau).^n)-gamma*x(i);
end

% x = x-1;

f_mean = mean(x);
x = x - f_mean;

f_tt = abs(max(x)-min(x));
x = 2*x/f_tt;
%%

t = 301:1500;
train = 1:900;
valid = 901:1000;
test = 1001:length(t);


input = [x(t-20) x(t-15) x(t-10) x(t-5) x(t)];
output = x(t+5);

% can be accessed trough input(train,:), input(valid,:), input(test,:)


%% two layer network

nodes1 = 3;
nodes2 = 8;
nodesOut = 1;
inputs = 5;
W1 = 1*rand(nodes1,inputs+1);
W2 = 1*rand(nodes2,nodes1+1);
V = 1*rand(nodesOut,nodes2+1);
X = [input(train,:) ones(length(input(train)),1)]';
t = output(train)';
epochs = 1500;
X_valid = [input(valid,:) ones(length(input(valid)),1)]';
t_valid = output(valid)';

dw1 = zeros(size(W1));
dw2 = zeros(size(W2));
dv = zeros(size(V));
alpha = 0.9;
eta = 0.00005;
T = 20; % minimal amount of epochs

%%
for k = 1:epochs
    [a1,z1] = forwardGeneral(W1,X); 
    z1 = [z1;ones(1,length(z1))];
    
    [a2,z2] = forwardGeneral(W2,z1); 
    z2 = [z2;ones(1,length(z2))];
    
    [a3,z3] = forwardGeneral(V,z2);
    
%     [~,dY] = sigmoid2(a3);
    dY = a3;
    
    
    
    delta3 = (z3-t).*dY;
    delta2 = backwardGeneral(a2,V,delta3);
    delta1 = backwardGeneral(a1,W2,delta2);

    dw1 = updateGeneral(dw1,alpha,delta1,X);
    dw2 = updateGeneral(dw2,alpha,delta2,z1);
    dv = updateGeneral(dv,alpha,delta3,z2);

    dw1 = (dw1 .* alpha) - (delta1 * X') .* (1-alpha);
    dw2 = (dw2 .* alpha) - (delta2 * z1') .* (1-alpha);
    dv = (dv .* alpha) - (delta3 * z2') .* (1-alpha);
    
    W1 = W1 + eta*dw1;
    W2 = W2 + eta*dw2;
    V = V + eta*dv;
    
    plot(z3)
    hold on
    plot(t)
    hold off
    drawnow
    
 
    
    % Early stopping test, i.e. test on validation set to detect
    % overfitting
    
    
    [b1,u1] = forwardGeneral(W1,X_valid); 
    u1 = [u1;ones(1,length(u1))];
    [b2,u2] = forwardGeneral(W2,u1);
    u2 = [u2;ones(1,length(u2))];
    [b3,u3] = forwardGeneral(V,u2);
    error(k) = mean((norm(b3-t_valid)).^2);
    
    
    
    if (k > T && error(k)>error(k-1) && error(k)>error(k-2) && error(k)>error(k-3) && error(k)>error(k-4) && error(k)>error(k-5))
        break % might not find global minimum but always a low value
    end
 
end
disp('done')
%%
figure()
[~,iMin] = min(error);
minVec = NaN(length(error),1);
minVec(iMin) = error(iMin);
plot(error)
hold on
plot(minVec, '*')
hold off

figure()

X = [input(test,:) ones(length(input(test)),1)]';

[a1,z1] = forwardGeneral(W1,X); 
z1 = [z1;ones(1,length(z1))];
    
[a2,z2] = forwardGeneral(W2,z1); 
z2 = [z2;ones(1,length(z2))];
   
[a3,z3] = forwardGeneral(V,z2);

plot(a3*f_tt+f_mean)
hold on
plot(output(test)*f_tt+f_mean)

MSE = mean((output(test)-a3').^2)

legend('Estimated function','Real function')