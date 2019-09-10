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
datapoints = 101;


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



data = [data1(:,1:floor(length(data1)*0.75)) data2(:,1:floor(length(data2)*0.75))];
test_data = [data1(:,ceil(length(data1)*0.75):end) data2(:,ceil(length(data2)*0.75):end)];

target = [target1(1:floor(length(data1)*0.75)) target2(1:floor(length(data2)*0.75))];
test_target = [target1(ceil(length(data1)*0.75):end) target2(ceil(length(data2)*0.75):end)];

X_train = data;
X_test = test_data;
t_train = target;
t_test = test_target;
%t(t==-1) = 0; % for sigmoid
nData_train = length(X_train);
nData_test = length(X_test);

X_train = [X_train;ones(1,nData_train)];
X_test = [X_test;ones(1,nData_test)];

epochs = 10000;
guess_train = zeros(epochs,length(t_train));
error_train = zeros(epochs,1);
error_test = error_train;

%% Remove: 50% from classA

tDataA = data(:,target==1);
tDataB = data(:,target==-1);
targetA = target(target==1);
targetB = target(target==-1);

tDataA = tDataA(:,1:datapoints);
targetA = targetA(1:datapoints);

[shuffled,stargets] = shuffle(tDataA,tDataB,targetA,targetB);

X_train = shuffled;
t_train = stargets;
nData_train = length(X_train);
X_train = [X_train;ones(1,nData_train)];
epochs = 100;
guess_train = zeros(epochs,length(t_train));
error_train = zeros(epochs,1);


%% Remove: 50% from classB

tDataA = data(:,target==1);
tDataB = data(:,target==-1);
targetA = target(target==1);
targetB = target(target==-1);

tDataB = tDataB(:,1:datapoints);
targetB = targetB(:,1:datapoints);

[shuffled,stargets] = shuffle(tDataA,tDataB,targetA,targetB);

X_train = shuffled;
t_train = stargets;
nData_train = length(X_train);
X_train = [X_train;ones(1,nData_train)];
epochs = 100;
guess_train = zeros(epochs,length(t_train));
error_train = zeros(epochs,1);

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

X_train = shuffled;
t_train = stargets;
nData_train = length(X_train);
X_train = [X_train;ones(1,nData_train)];
epochs = 100;
guess_train = zeros(epochs,length(t_train));
error_train = zeros(epochs,1);


%% OBS testet skall även inkludera alla X-värden när man utvärderar error

dw = zeros(size(W));
 
for k = 1:epochs
    hin_train = W * X_train;
    %hin_test = W * X_test; %For 50% Test
    
    hout_train = [2 ./ (1+exp(-hin_train)) - 1 ];
    %hout_test = [2 ./ (1+exp(-hin_test)) - 1 ]; %For 50% Test

    delta_o = (hout_train - t_train) .* ((1 + hout_train) .* (1 - hout_train)) * 0.5;
    delta_o = delta_o(1:nodes, :);

    dw = (dw .* alpha) - (delta_o * X_train') .* (1-alpha);
    W = W + dw .* eta;

    guess_train(k,:) = sign(hout_train);
    %guess_test(k,:) = sign(hout_test); %For 50% Test
    
    error_train(k) = mean((guess_train(k,:)-t_train).^2);
    %error_test(k) = mean((guess_test(k,:)-t_test).^2); %For 50% Test

    % add tError as (W*data-targets)

end
 
plot(error_train)
hold on
plot(error_test)
legend('Training Error MSE','Testing Error MSE')