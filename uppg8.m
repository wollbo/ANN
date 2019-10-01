%% uppg 3.2.3 function approximation
clear all
close all
x=[-5:0.5:5]';
y=[-5:0.5:5]';
z=exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5;
%mesh(x, y, z);
% z = z*10+2;

ndata = length(x)*length(y);

targets = reshape(z, 1, ndata);
[xx, yy] = meshgrid (x, y);
patterns = [reshape(xx, 1, ndata); reshape(yy, 1, ndata)];

% add nSamp


nodes1 = 25;
nodes2 = 1;
inputs = 2; 
eta = 0.001;
outputs = 1;
alpha = 0.9; %0.9
epochs = 5000;
X = patterns;


% X = X./max(max(X)); %normalization
% normalize(X)

nData = length(X);
X = [X;ones(1,nData)];
t = targets;
M = 100;


%%
for m = 1:100
for l = 0:4
nSamp = 1-0.2*l; %0.8 to 0.2
perm = randperm(round(length(targets)));
perm = perm(1:round(nSamp*length(perm)));
sX = X(:,perm);
st = t(perm);
%sX = X;
%st = t;
W = 0.1*rand(nodes1,inputs+1);
V = 0.1*rand(nodes2,nodes1+1);
dw = zeros(size(W));
dv = zeros(size(V));


for k = 1:epochs
    [a1,z1] = forwardGeneral(W,sX);
    z1 = [z1;ones(1,length(z1))];
    [a2,z2] = forwardGeneral(V,z1);
    
    
    [~,dY] = sigmoid2(a2);
    dY = 1;
    delta2 = (a2-st).*dY;
    delta1 = backwardGeneral(a1,V,delta2);
    

    dw = (dw .* alpha) - (delta1 * sX') .* (1-alpha);
    dv = (dv .* alpha) - (delta2 * z1') .* (1-alpha);
    
    
    
    W = W + eta*dw;
    V = V + eta*dv;

    %guess(k,:,:) = a2;
    %error(k) = mean((a2-st).^2);
end

[a1,z1] = forwardGeneral(W,X);
z1 = [z1;ones(1,length(z1))];
[a2,z2] = forwardGeneral(V,z1);
out = a2;
zz = reshape(out, length(x), length(y));
MSE(l+1,m) = mean(mean((z-zz).^2));
%merror = 1-error/max(error);
end
end

%%
f1 = figure('Name','figures/functionApproximation')
subplot(1,2,1)
mesh(x,y,zz);
title('Estimated function')
axis([-5 5 -5 5 -2 2]);
subplot(1,2,2)
mesh(x,y,abs(z-zz));
axis([-5 5 -5 5 -2 2]);
title('Absolute error')

% f2 = figure('Name','figures/functionapproxMSE')
% plot(error)
% title('Learning curve of the function approximator')
% ylabel('MSE')
% xlabel('Epoch')


