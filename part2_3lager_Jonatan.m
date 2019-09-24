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




t = 301:1500;
train = 1:900;
valid = 901:1000;
test = 1001:length(t);


input = [x(t-20) x(t-15) x(t-10) x(t-5) x(t)];
output = x(t+5);
%%
% Scaling of data
% f_mean = mean(x);
% x = x - f_mean;

% f_var = sum((x-f_mean).^2)/length(x);
% x = (x-f_mean)/sqrt(f_var);

% x = x + randn(length(x),1)*0.5+10

% can be accessed trough input(train,:), input(valid,:), input(test,:)


%% three layer network

nodes1 = 8;
nodes2 = 8;
nodesOut = 1;
inputs = 5;
W1 = 0.1*rand(nodes1,inputs+1);
W2 = 0.1*rand(nodes2,nodes1+1);
V = 0.1*rand(nodesOut,nodes2+1);
X = [input(train,:) ones(length(input(train)),1)]';
t = output(train)';
epochs = 100000;
X_valid = [input(valid,:) ones(length(input(valid)),1)]';
t_valid = output(valid)';

dw1 = zeros(size(W1));
dw2 = zeros(size(W2));
dv = zeros(size(V));
alpha = 0.9;
eta = 0.0001;
T = 20; % minimal amount of epochs
n_epochs = 0;

%%
for k = 1:epochs
    [a1,z1] = forwardGeneral(W1,X); 
    z1 = [z1;ones(1,length(z1))];
    
    [a2,z2] = forwardGeneral(W2,z1);
%     [a2,z2] = forwardGeneral(W2,z1+ randn(length(z1),nodes1+1)'*0.001); 
    z2 = [z2;ones(1,length(z2))];
    
    [a3,z3] = forwardGeneral(V,z2);
%     [a3,z3] = forwardGeneral(V,z2 + randn(length(z2),nodes2+1)'*0.001);
    
%     [~,dY] = sigmoid2(a3);
    dY = 1;
    
    
    
    delta3 = (a3-t).*dY;
    delta2 = backwardGeneral(a2,V,delta3);
    delta1 = backwardGeneral(a1,W2,delta2);

%     dw1 = updateGeneral(dw1,alpha,delta1,X);
%     dw2 = updateGeneral(dw2,alpha,delta2,z1);
%     dv = updateGeneral(dv,alpha,delta3,z2);

    dw1 = (dw1 .* alpha) - (delta1 * X') .* (1-alpha);
    dw2 = (dw2 .* alpha) - (delta2 * z1') .* (1-alpha);
    dv = (dv .* alpha) - (delta3 * z2') .* (1-alpha);
    
    W1 = W1 + eta*dw1;
    W2 = W2 + eta*dw2;
    V = V + eta*dv;
    
%     plot(a3)
%     hold on
%     plot(t)
%     hold off
%     drawnow
    
 
    
    % Early stopping test, i.e. test on validation set to detect
    % overfitting
    
    
    [b1,u1] = forwardGeneral(W1,X_valid); 
    u1 = [u1;ones(1,length(u1))];
    [b2,u2] = forwardGeneral(W2,u1);
    u2 = [u2;ones(1,length(u2))];
    [b3,u3] = forwardGeneral(V,u2);
    error_valid(k) = mean(((b3-t_valid)).^2);
    error_train(k) = mean(((a3-t)).^2);
    
    
    
    if ~(k > T && error_valid(k)>error_valid(k-1) && error_valid(k)>error_valid(k-2) && error_valid(k)>error_valid(k-3) && error_valid(k)>error_valid(k-4) && error_valid(k)>error_valid(k-5))
%         break % might not find global minimum but always a low value
        W1_opt = W1;
        W2_opt = W2;
        V_opt = V;
        n_epochs = n_epochs + 1;
    end
 
end
disp('done')
%%
figure()
n_graph = min(n_epochs*5,length(error_valid));
[~,iMin] = min(error_valid);
minVec = NaN(length(error_valid),1);
minVec(iMin) = error_valid(iMin);
semilogy([1:n_graph],error_train(1:n_graph));
hold on
semilogy([1:n_graph],error_valid(1:n_graph));
plot(minVec, '*')
axis([1 n_graph 0 max(max(error_valid(1:n_graph)),max(error_train(1:n_graph)))])
hold off
xlabel('Number of epochs')
ylabel('MSE')
legend('Error curve - Training','Error curve - Validation','Optimal weights')
figure()

X = [input(test,:) ones(length(input(test)),1)]';

[a1,z1] = forwardGeneral(W1_opt,X); 
z1 = [z1;ones(1,length(z1))];
    
[a2,z2] = forwardGeneral(W2_opt,z1); 
z2 = [z2;ones(1,length(z2))];
   
[a3,z3] = forwardGeneral(V_opt,z2);

plot(a3)
hold on
plot(output(test))

MSE = mean((output(test)-a3').^2)

xlabel('t')
legend('Estimated function','Real function')