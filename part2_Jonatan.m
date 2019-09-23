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


% x = normalize(x)
%%
%normalisering - medelvärde

f_mean = mean(x);
% x = x - f_mean;

%normalisering - varians

f_var = sum((x-f_mean).^2)/length(x);
x = (x-f_mean)/sqrt(f_var);

%%
%Noise
% noise = randn(length(x),1)/100;
% x = x + noise;



%%

t = 301:1500;
train = 1:900;
valid = 901:1000;
test = 1001:length(t);


input = [x(t-20) x(t-15) x(t-10) x(t-5) x(t)];
output = x(t+5);

% can be accessed trough input(train,:), input(valid,:), input(test,:)


%% two layer network

nodes1 = 5;
nodes2 = 1;
inputs = 5;
W = 1*rand(nodes1,inputs+1);
V = 1*rand(nodes2,nodes1+1);
X = [input(train,:) ones(length(input(train)),1)]';
t = output(train)';
epochs = 1000;
X_valid = [input(valid,:) ones(length(input(valid)),1)]';
t_valid = output(valid)';

dw = zeros(size(W));
dv = zeros(size(V));
alpha = 0.9;
eta = 0.0001;
T = 2; % minimal amount of epochs



%%
for k = 1:epochs
    [a1,z1] = forwardGeneral(W,X); 
    z1 = [z1;ones(1,length(z1))];
    [a2,z2] = forwardGeneral(V,z1 + randn(length(z1),6)'*0);
    
%     [~,dY] = sigmoid2(a2);
    dY = a2;
    delta2 = (z2-t).*dY;
    delta1 = backwardGeneral(a1,V,delta2);

    dw = updateGeneral(dw,alpha,delta1,X);
    dv = updateGeneral(dv,alpha,delta2,z1);

    dw = (dw .* alpha) - (delta1 * X') .* (1-alpha);
    dv = (dv .* alpha) - (delta2 * z1') .* (1-alpha);
    
    W = W + eta*dw;
    V = V + eta*dv;
    
    plot(z2)
    hold on
    plot(t)
    hold off
    drawnow
    
    % Early stopping test, i.e. test on validation set to detect
    % overfitting
    [b1,u1] = forwardGeneral(W,X_valid); 
    u1 = [u1;ones(1,length(u1))];
    [b2,u2] = forwardGeneral(V,u1);
    error(k) = mean((norm(b2-t_valid)).^2);
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
[a1,z1] = forwardGeneral(W,X); 
z1 = [z1;ones(1,length(z1))];
[a2,z2] = forwardGeneral(V,z1);
% plot(a2*f_var+f_mean)
plot((a2*sqrt(f_var))+sqrt(f_mean))
hold on
% plot(output(test)*f_var+f_mean)
plot((output(test)*sqrt(f_var)+f_mean))

MSE = mean((output(test)-a2').^2)
legend('Estimated function','Real function')



