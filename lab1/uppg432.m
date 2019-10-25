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
sigma = 0.18; % add noise
x = x + sigma*randn(size(x));


%%

t = 301:1500;
train = 1:900;
valid = 901:1000;
test = 1001:length(t);


input = [x(t-20) x(t-15) x(t-10) x(t-5) x(t)]; 
output = x(t+5);


% can be accessed trough input(train,:), input(valid,:), input(test,:)

%% three layer network

nodes1 = 5;
nodes2 = 5;
nodesOut = 1;
inputs = 5;
epochs = 1500;
X_valid = [input(valid,:) ones(length(input(valid)),1)]';
t_valid = output(valid)';
lambda = 10e-4; % ger inte b�ttre prestanda �n lambda = 0 : kolla om weight decay �r r�tt


alpha = 0.9;
eta = 10e-4; % p� gr�nsen, konvergerar ej f�r eta = 10e-3. learning rate verkar vara avg�rande faktor f�r prestanda
T = 20; % minimal amount of epochs

%% Tre lager
eta = 3*10e-4; % p� gr�nsen, konvergerar ej f�r eta = 10e-3. learning rate verkar vara avg�rande faktor f�r prestanda
for m = 1:100
    W1 = 0.1*rand(nodes1,inputs+1);
    W2 = 0.1*rand(nodes2,nodes1+1);
    V = 0.1*rand(nodesOut,nodes2+1);
    X = [input(train,:) ones(length(input(train)),1)]';
    t = output(train)';
    dw1 = zeros(size(W1));
    dw2 = zeros(size(W2));
    dv = zeros(size(V));
    n_epochs = 0;
for k = 1:epochs
    [a1,z1] = forwardGeneral(W1,X); 
    z1 = [z1;ones(1,length(z1))];
    
    [a2,z2] = forwardGeneral(W2,z1); 
    z2 = [z2;ones(1,length(z2))];
    
    [a3,z3] = forwardGeneral(V,z2);
    
    dY = a3;
    
    weights = cell(3,1);
    weights{1} = W1;
    weights{2} = W2;
    weights{3} = V;

    delta3 = (weightDecayError(a3,t,lambda,weights)).*dY;
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
    
    error_train(k) = mean(((a3-t)).^2);
    
    [b1,u1] = forwardGeneral(W1,X_valid); 
    u1 = [u1;ones(1,length(u1))];
    [b2,u2] = forwardGeneral(W2,u1);
    u2 = [u2;ones(1,length(u2))];
    [b3,u3] = forwardGeneral(V,u2);
    error_valid(k) = mean((b3-t_valid).^2);
    
    
    
    if ~(k > T && error_valid(k)>error_valid(k-1) && error_valid(k)>error_valid(k-2) && error_valid(k)>error_valid(k-3) && error_valid(k)>error_valid(k-4) && error_valid(k)>error_valid(k-5))
        %break % might not find global minimum but always a low value
        W1_opt = W1;
        W2_opt = W2;
        V_opt = V;
        n_epochs = n_epochs + 1;
    end
 
end
X = [input(test,:) ones(length(input(test)),1)]';
[a1,z1] = forwardGeneral(W1_opt,X); 
z1 = [z1;ones(1,length(z1))];
[a2,z2] = forwardGeneral(W2_opt,z1);
z2 = [z2;ones(1,length(z2))];
[a3,z3] = forwardGeneral(V_opt,z2);
MSE(m) = mean(((output(test)-a3').^2));
end
disp('done')
mm3 = mean(MSE)
mvar3 = var(MSE)

%% Two layer network
eta = 0.005;
for m = 1:100
    W = 0.1*rand(nodes1,inputs+1);
    V = 0.1*rand(nodesOut,nodes1+1);
    X = [input(train,:) ones(length(input(train)),1)]';
    t = output(train)';
    dw = zeros(size(W));
    dv = zeros(size(V));
    n_epochs = 0;
for k = 1:epochs
    [a1,z1] = forwardGeneral(W,X); 
    z1 = [z1;ones(1,length(z1))];
    
    [a2,z2] = forwardGeneral(V,z1); 
    dY = a2;
    
    weights = cell(3,1);
    weights{1} = W;
    weights{2} = V;

    delta2 = (weightDecayError(a2,t,lambda,weights)).*dY;
    delta1 = backwardGeneral(a2,V,delta2);

    dw = updateGeneral(dw,alpha,delta1,X);
    dv = updateGeneral(dv,alpha,delta2,z1);

    dw = (dw .* alpha) - (delta1 * X') .* (1-alpha);
    dv = (dv .* alpha) - (delta2 * z1') .* (1-alpha);
    
    W = W + eta*dw;
    V = V + eta*dv;
    
    error_train(k) = mean(((a2-t)).^2);
    
    [b1,u1] = forwardGeneral(W,X_valid); 
    u1 = [u1;ones(1,length(u1))];
    [b2,u2] = forwardGeneral(V,u1);
    error_valid(k) = mean((b2-t_valid).^2);
    
    
    
    if ~(k > T && error_valid(k)>error_valid(k-1) && error_valid(k)>error_valid(k-2) && error_valid(k)>error_valid(k-3) && error_valid(k)>error_valid(k-4) && error_valid(k)>error_valid(k-5))
        %break % might not find global minimum but always a low value
        W_opt = W;
        V_opt = V;
        n_epochs = n_epochs + 1;
    end
 
end
X = [input(test,:) ones(length(input(test)),1)]';
[a1,z1] = forwardGeneral(W_opt,X); 
z1 = [z1;ones(1,length(z1))];
[a2,z2] = forwardGeneral(V_opt,z1);

MSE(m) = mean(((output(test)-a2').^2));
end
disp('done')
mm2 = mean(MSE)
mvar2 = var(MSE)
%%
n_graph = min(n_epochs*5,length(error_valid));

[~,iMin] = min(error_valid);
minVec = NaN(length(error_valid),1);
minVec(iMin) = error_valid(iMin);
plot([1:n_graph],error_train(1:n_graph))
hold on
plot([1:n_graph],error_valid(1:n_graph))
plot(minVec, '*')
axis([1 n_graph 0 max(max(error_valid(1:n_graph)),max(error_train(1:n_graph)))])
hold off
xlabel('Number of epochs')
ylabel('MSE')
legend('Error on training set','Error on validation set','Optimal weights')

f2=figure('Name','figures/uppg432sigma018')

X = [input(test,:) ones(length(input(test)),1)]';

[a1,z1] = forwardGeneral(W1,X); 
z1 = [z1;ones(1,length(z1))];
    
[a2,z2] = forwardGeneral(W2,z1); 
z2 = [z2;ones(1,length(z2))];
   
[a3,z3] = forwardGeneral(V,z2);

plot(a3)
hold on
plot(output(test))

MSE = mean((output(test)-a3').^2)

legend('Estimated function','Real function')

title([char('sigma = ') num2str(sigma)]);