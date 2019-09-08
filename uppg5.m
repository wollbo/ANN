%% uppg 5

% enl. labbpeket

ndata = 100;
mA = [ 1.0, 0.3]; sigmaA = 0.2;
mB = [ 0.0, -0.1]; sigmaB = 0.3;
classA(1,:) = [ randn(1,round(0.5*ndata)) .* sigmaA - mA(1), randn(1,round(0.5*ndata)) .* sigmaA + mA(1)]; % creates data from two distributions in A
classA(2,:) = randn(1,ndata) .* sigmaA + mA(2);
classB(1,:) = randn(1,ndata) .* sigmaB + mB(1);
classB(2,:) = randn(1,ndata) .* sigmaB + mB(2);

scatter(classA(1,:),classA(2,:))
hold on
scatter(classB(1,:),classB(2,:))
hold off

%%

%klok implementation
mu = [1 0.3;-1 0.3;0 -0.1];%;-2 -5];
sigma = [0.2 0.2;0.2 0.2;0.3 0.3];%; 2 6];
%mu = [1 2;5 7];%;-2 -5];
%sigma = [1 0.8;0.5 2];%; 2 6];
datapoints = 50;


[data1, target1] = generateData(datapoints,[mu(1,:); mu(3,:)],[sigma(1,:); sigma(3,:)]);
[data2, target2] = generateData(datapoints,[mu(2,:); mu(3,:)],[sigma(1,:); sigma(3,:)]);

data = [data1; data2];
target = [target1; target2];
%%
dataSplit1 = data(target==1,:); % bara för viualisering
dataSplit2 = data(target==-1,:);
scatter(dataSplit1(:,1),dataSplit1(:,2))
hold on
scatter(dataSplit2(:,1),dataSplit2(:,2))
hold off

%% Olika testfall: 

% Then apply the Delta learning rule in batch mode to this new dataset as well
% as to different versions of the subsampled data, i.e. before training please remove 25% of data samples (for two classes with n=100 samples each, remove 50
% samples) according to the following scenarios:
% • random 25% from each class
% • random 50% from classA
% • random 50% from classB
% • 20% from a subset of classA for which classA(1,:)<0 and 80% from a
% subset of classA for which classA(1,:)>0

nodes = 1;
inputs = 2;
W = 0.01*randn(nodes,inputs+1);
eta = 0.0001;
outputs = 1;
alpha = 0.9;
data1 = data1'
data2 = data2'
target1 = target1'
target2 = target2'
data = [data1 data2];
target = [target1 target2];

%% Remove:
% • random 25% from each class

data = [data1(:,datapoints*0.75) data2(:,datapoints*0.75)];

X = data';
t = target';
%t(t==-1) = 0; % for sigmoid
nData = length(X);
X = [X;ones(1,nData)];
epochs = 100;
guess = zeros(epochs,length(t));
error = zeros(epochs,1);

%% Remove: 50% from classA

tDataA = data(:,target==1);
tDataB = data(:,target==-1);
targetA = target(target==1);
targetB = target(target==-1);

tDataA = tDataA(:,1:datapoints);
targetA = targetA(:,1:datapoints);

[shuffled,stargets] = shuffle(tDataA,tDataB,targetA,targetB);

X = shuffled;
t = stargets;
nData = length(X);
X = [X;ones(1,nData)];
epochs = 100;
guess = zeros(epochs,length(t));
error = zeros(epochs,1);


%% Remove: 50% from classB

tDataA = data(:,target==1);
tDataB = data(:,target==-1);
targetA = target(target==1);
targetB = target(target==-1);

tDataB = tDataB(:,1:datapoints);
targetB = targetB(:,1:datapoints);

[shuffled,stargets] = shuffle(tDataA,tDataB,targetA,targetB);

X = shuffled;
t = stargets;
nData = length(X);
X = [X;ones(1,nData)];
epochs = 100;
guess = zeros(epochs,length(t));
error = zeros(epochs,1);

%% Remove: 20% from classA1 80% from classA2

tDataA1 = data1(:,target1==1);
tDataA2 = data2(:,target2==1);
tDataB = data(:,target==-1);
targetA1 = target1(target1==1);
targetA2 = target2(target2==1);
targetB = target(target==-1);

tDataA1 = tDataA1(:,1:datapoints*0.8);
tDataA2 = tDataA2(:,1:datapoints*0.2);
targetA1 = target1(:,1:datapoints*0.8);
targetA2 = target2(:,1:datapoints*0.2);

tDataA = [tDataA1, tDataA2];
targetA = [targetA1, targetA2];

[shuffled,stargets] = shuffle(tDataA,tDataB,targetA,targetB);

X = shuffled;
t = stargets;
nData = length(X);
X = [X;ones(1,nData)];
epochs = 100;
guess = zeros(epochs,length(t));
error = zeros(epochs,1);


%% OBS testet skall även inkludera alla X-värden när man utvärderar error

dw = zeros(size(W));
 
for k = 1:epochs
    hin = W * X;
    hout = [2 ./ (1+exp(-hin)) - 1 ];

    delta_o = (hout - t) .* ((1 + hout) .* (1 - hout)) * 0.5;
    delta_o = delta_o(1:nodes, :);

    dw = (dw .* alpha) - (delta_o * X') .* (1-alpha);
    W = W + dw .* eta;

    guess(k,:) = sign(hout);
    error(k) = mean((guess(k,:)-t).^2);

end
 
plot(error)