%% generate data
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

%%

t = 301:1500;
input = [x(t-20) x(t-15) x(t-10) x(t-5) x(t)];
output = x(t+5);

%%

nodes1 = 5;
nodes2 = 1;
inputs = 5;
W = 1*rand(nodes1,inputs+1);
V = 1*rand(nodes2,nodes1+1);
X = [input ones(length(input),1)]';
t = output';
epochs = 10;

dw = zeros(size(W));
dv = zeros(size(V));
alpha = 0.9;
eta = 0.1;

%%
for k = 1:epochs
    [a1,z1] = forwardGeneral(W,X); 
    z1 = [z1;ones(1,length(z1))];
    [a2,z2] = forwardGeneral(V,z1);
    
    [~,dY] = sigmoid2(a2);
    delta2 = (z2-t).*dY;
    delta1 = backwardGeneral(a1,V,delta2);

    dw = updateGeneral(dw,alpha,delta1,X);
    dv = updateGeneral(dv,alpha,delta2,z1);

    dw = (dw .* alpha) - (delta1 * X') .* (1-alpha);
    dv = (dv .* alpha) - (delta2 * z1') .* (1-alpha);
    
    W = W + eta*dw;
    V = V + eta*dv;
 
end