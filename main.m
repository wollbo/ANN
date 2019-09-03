%% main
clear all
close all

%Generate Data
mean = [1 2;5 7;-2 -5];
sigma = [1 0.8;0.5 2; 2 6];
datapoints = 100;

[data target] = generateData(datapoints,mean,sigma);

scatter(data(:,1),data(:,2),target.^3)

%Forward Pass

%Error Backpropagation

%Weight update