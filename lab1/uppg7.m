%% 3.2.2 Autoencoder

clear all
close all

data = -1*ones(8);
data = data+2*eye(8);

target = data;

nodes1 = 4;
nodes2 = 8;
inputs = 8;
W = 1*randn(nodes1,inputs+1);
V = 1*randn(nodes2,nodes1+1);
eta = 0.1;
outputs = 8;
alpha = 0.9;
epochs = 10000;
X = data;
nData = length(X);
X = [X;ones(1,nData)];
t = target;

dw = zeros(size(W));
dv = zeros(size(V));

%%

for k = 1:epochs
    [a1,z1] = forwardGeneral(W,X);
    z1 = [z1;ones(1,length(z1))];
    [a2,z2] = forwardGeneral(V,z1);
    
    [~,dY] = sigmoid2(a2); 
    delta2 = (z2-t).*dY;
    delta1 = backwardGeneral(a1,V,delta2);
    
    %delta1 = delta1(1:end-1,:);

    dw = updateGeneral(dw,alpha,delta1,X);
    dv = updateGeneral(dv,alpha,delta2,z1);

    dw = (dw .* alpha) - (delta1 * X') .* (1-alpha);
    dv = (dv .* alpha) - (delta2 * z1') .* (1-alpha);
    W = W + dw .* eta;
    V = V + dv .* eta;

    %guess(k,:,:) = z2;
    %error(k) = mean((guess(k,:)-t).^2);
    %rate(k) = 1-length(target(target==guess(k,:)))/length(target);
    % add tError as (W*data-targets)

end
%%
% X = -ones(size(X));
% X = zeros(size(X));
% X(8,8) = 1;
 f = figure('Name','figures/uppgEncoder4');

[a1,z1] = forwardGeneral(W,X);
z1 = [z1;ones(1,length(z1))];
[a2,z2] = forwardGeneral(V,z1);
a2 = (sigmoid2(a2));

% z1 = z1(1:length(),:)

subplot(3,1,1)
% sgtitle('n hidden neurons')
imagesc((data-min(min(data)))/(max(max(data))-min(min(data))))
title('Input data')
% colorbar
subplot(3,1,2)
imagesc((W-min(min(W)))/(max(max(W))-min(min(W))))
title('Weight Matrix')
% colorbar
subplot(3,1,3)
% image((a2-min(min(a2)))/(max(max(a2))-min(min(a2)))*50)
imagesc((a2+1)/2)
title('Output data')
% colorbar

MSE = mean(mean((data-a2).^2))