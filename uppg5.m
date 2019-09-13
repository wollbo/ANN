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
close all
clear all

%klok implementation
mu = [-2 -3;-4 -1.3;-1 3;1 1];%;-2 -5];
sigma = [0.7 0.4 ;0.3 1;0.6 0.3;0.5 0.8];%; 2 6];
%mu = [1 2;5 7];%;-2 -5];
%sigma = [1 0.8;0.5 2];%; 2 6];
datapoints = 200;


[data1, target1] = generateData(datapoints,[mu(1,:); mu(4,:)],[sigma(1,:); sigma(4,:)],[-1 1]);
[data2, target2] = generateData(datapoints,[mu(3,:); mu(2,:)],[sigma(3,:); sigma(2,:)],[-1 1]);

data = [data1; data2];
target = [target1; target2];

%[data target] = generateData(datapoints,mu,sigma,[-1 -1 1 1]);

% scatter(data1(:,1),data1(:,2));
% hold on
% scatter(data2(:,1),data2(:,2));

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
%W = 0.011*randn(nodes,inputs+1);
W = [1 1 0];
eta = 0.01;
outputs = 1;
alpha = 0.9;
data1 = data1';
data2 = data2';
target1 = target1';
target2 = target2';
data = [data1 data2];
target = [target1 target2];

%% Remove:
% • random 25% from each class

split_per = 0.75;

data = [data1(:,1:floor(length(data1)*split_per)) data2(:,1:floor(length(data2)*split_per))];
test_data = [data1(:,ceil(length(data1)*split_per):end) data2(:,ceil(length(data2)*split_per):end)];

target = [target1(1:floor(length(data1)*split_per)) target2(1:floor(length(data2)*split_per))];
test_target = [target1(ceil(length(data1)*split_per):end) target2(ceil(length(data2)*split_per):end)];

X_train = data;
X_test = test_data;
t_train = target;
t_test = test_target;
%t(t==-1) = 0; % for sigmoid
nData_train = length(X_train);
nData_test = length(X_test);

X_train = [X_train;ones(1,nData_train)];
X_test = [X_test;ones(1,nData_test)];

epochs = 100;
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
x_dec = [min(X_train(1,:,:))-0.5 max(X_train(1,:,:))+0.75];
%hold off
 
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
    err = find((guess_train(k,:)-t_train));
    %error_test(k) = mean((guess_test(k,:)-t_test).^2); %For 50% Test

    % add tError as (W*data-targets)
    
    y_dec = - (x_dec*(W(1)/W(2)) + (W(3)/W(2)));
    plot(x_dec,y_dec,'-.k','Linewidth',2)
    
    hold on
    scatter(X_train(1,find(t_train-1)),X_train(2,find(t_train-1)));
    scatter(X_train(1,find(t_train+1)),X_train(2,find(t_train+1)));
    scatter(X_train(1,err),X_train(2,err),'kx');
    
    legend('Decision Boundary','Category 1','Catergory 2','Errors','location','northwest')
    
    hold off
    
    axis([min(X_train(1,:,:))-0.5 max(X_train(1,:,:)+0.75) min(X_train(2,:,:))-0.25 max(X_train(2,:,:))+0.75])
    %disp(error_train(k))
    %disp(guess_train(k,:)-t_train)
    
    drawnow

end

%x_dec = min(X_train(1,:,:)):0.001:max(X_train(1,:,:));
%y_dec = x_dec*(W(1)/W(2)) - (W(3)/W(2));
%plot(x_dec,y_dec,'-.k','Linewidth',2)
%legend('Category 1','Catergory 2','Decision Boundary')



%plot(error_train)
%hold on
%plot(error_test)
%legend('Training Error MSE','Testing Error MSE')



