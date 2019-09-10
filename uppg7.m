%% 3.2.2 Autoencoder

data = -1*ones(8);
data = data+2*eye(8);

target = data;

nodes1 = 3;
nodes2 = 8;
inputs = 8;
W = 0.01*randn(nodes1,inputs+1);
V = 0.01*randn(nodes2,nodes1);
eta = 0.01;
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

    %guess(k,:,:) = z2;
    %error(k) = mean((guess(k,:)-t).^2);
    %rate(k) = 1-length(target(target==guess(k,:)))/length(target);
    % add tError as (W*data-targets)

end

% Maps everything to (-7+1)/8 = -0.75 i.e. the mean

