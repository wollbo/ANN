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


% % x = normalize(x)
% %normalisering - medelvärde
% 
% f_mean = mean(x);
% % x = x - f_mean;
% 
% %normalisering - varians
% 
% f_var = sum((x-f_mean).^2)/length(x);
% x = (x-f_mean)/sqrt(f_var);
% 
% %Noise
% % noise = randn(length(x),1)*0.01;
% % x = x + noise;




t = 301:1500;
train = 1:900;
valid = 901:1000;
test = 1001:length(t);


input = [x(t-20) x(t-15) x(t-10) x(t-5) x(t)];
output = x(t+5);
%%
% Scaling of variables
%normalisering - medelvärde

f_mean = mean(x);
% x = x - f_mean;

%normalisering - varians

f_var = sum((x-f_mean).^2)/length(x);
x = (x-f_mean)/sqrt(f_var);

%Noise
% noise = randn(length(x),1)*0.01;
% x = x + noise;

% can be accessed trough input(train,:), input(valid,:), input(test,:)


%% two layer network

nodes1 = 8;
nodes2 = 1;
inputs = 5;
W = 0.1*rand(nodes1,inputs+1);
V = 0.1*rand(nodes2,nodes1+1);
X = [input(train,:) ones(length(input(train)),1)]';
t = output(train)';
epochs = 1000;
X_valid = [input(valid,:) ones(length(input(valid)),1)]';
t_valid = output(valid)';

dw = zeros(size(W));
dv = zeros(size(V));
alpha = 0.9;
eta = 0.001;
T = 20; % minimal amount of epochs
n_epochs = 0;



%%
for k = 1:epochs
    [a1,z1] = forwardGeneral(W,X); 
    z1 = [z1;ones(1,length(z1))];
%     [a2,z2] = forwardGeneral(V,z1);
    [a2,z2] = forwardGeneral(V,z1 + randn(length(z1),nodes2)'*0.001);
    
%     [~,dY] = sigmoid2(a2);
    dY = a2;

    delta2 = (a2-t).*dY;
    delta1 = backwardGeneral(a1,V,delta2);

%     dw = updateGeneral(dw,alpha,delta1,X);
%     dv = updateGeneral(dv,alpha,delta2,z1);

    dw = (dw .* alpha) - (delta1 * X') .* (1-alpha);
    dv = (dv .* alpha) - (delta2 * z1') .* (1-alpha);
    
    W = W + eta*dw;
    V = V + eta*dv;
    
%     plot(a2)
%     hold on
%     plot(t)
%     hold off
%     drawnow
    
    % Error on training set
    
    
    
%     [b1,u1] = forwardGeneral(W,X); 
%     u1 = [u1;ones(1,length(u1))];
%     [b2,u2] = forwardGeneral(V,u1);
    error_train(k) = mean(((a2-t)).^2);
    
    % Early stopping test, i.e. test on validation set to detect
    % overfitting
    [b1,u1] = forwardGeneral(W,X_valid); 
    u1 = [u1;ones(1,length(u1))];
    [b2_valid,u2] = forwardGeneral(V,u1);
    error_valid(k) = mean(((b2_valid-t_valid)).^2);
 
    if ~(k > T && error_valid(k)>error_valid(k-1) && error_valid(k)>error_valid(k-2) && error_valid(k)>error_valid(k-3) && error_valid(k)>error_valid(k-4) && error_valid(k)>error_valid(k-5))
        W_opt = W;
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
plot([1:n_graph],error_train(1:n_graph))
hold on
plot([1:n_graph],error_valid(1:n_graph))
plot(minVec, '*')
axis([1 n_graph 0 max(max(error_valid(1:n_graph)),max(error_train(1:n_graph)))])
hold off
xlabel('Number of epochs')
ylabel('MSE')
legend('Error on training set','Error on validation set','Optimal weights')

figure()

X = [input(test,:) ones(length(input(test)),1)]';
[a1,z1] = forwardGeneral(W_opt,X); 
z1 = [z1;ones(1,length(z1))];
[a2,z2] = forwardGeneral(V_opt,z1);
% plot(a2*f_var+f_mean)
plot(a2)
hold on
% plot(output(test)*f_var+f_mean)
plot((output(test)))
xlabel('t')

MSE = mean(((output(test)-a2').^2))
legend('Estimated function','Real function')



