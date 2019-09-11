%% uppg 3.2.3 function approximation

x=[-5:0.5:5]';
y=[-5:0.5:5]';
z=exp(-x.*x*0.1) * exp(-y.*y*0.1)' - 0.5;
mesh(x, y, z);

ndata = length(x)*length(y);

targets = reshape(z, 1, ndata);
[xx, yy] = meshgrid (x, y);
patterns = [reshape(xx, 1, ndata); reshape(yy, 1, ndata)];

nodes1 = 10;
nodes2 = 1;
inputs = 2;
W = 0.001*randn(nodes1,inputs+1);
V = 0.001*randn(nodes2,nodes1+1); % här är det skumt, bias terms??
eta = 0.5;
outputs = 1;
alpha = 0.9;
epochs = 1000;
X = patterns;
nData = length(X);
X = [X;ones(1,nData)];
t = targets;

dw = zeros(size(W));
dv = zeros(size(V));

%%

for k = 1:1000
    [a1,z1] = forwardGeneral(W,X);
    z1 = [z1;ones(1,length(z1))];
    [a2,z2] = forwardGeneral(V,z1); % a2 konstant??
    % probably a bias problem ...
    
    [~,dY] = sigmoid2(a2); 
    delta2 = (z2-t).*dY;
    delta1 = backwardGeneral(a2,V,delta2);
    
    delta1 = delta1(1:end-1,:);

    dw = updateGeneral(dw,eta,alpha,delta1,X);
    dv = updateGeneral(dv,eta,alpha,delta2,z1);

    dw = (dw .* alpha) - (delta1 * X') .* (1-alpha);
    dv = (dv .* alpha) - (delta2 * z1') .* (1-alpha);
    
    
    
    W = W + dw .* eta;
    V = V + dv .* eta;

    %guess(k,:,:) = z2;
    %error(k) = mean((guess(k,:)-t).^2);
    %rate(k) = 1-length(target(target==guess(k,:)))/length(target);
    % add tError as (W*data-targets)

%     out = z2;
%     zz = reshape(out, length(x), length(y));
%     mesh(x,y,zz);
%     axis([-5 5 -5 5 -0.7 0.7]);
%     drawnow
    
end

%%

