function [labeledVector] = labelData(dataMatrix)
dataSize = size(dataMatrix);
nLables = dataSize(3);

labeledVector = [];
labels = [-1 1]

for i = 1:nLables
    tempData = dataMatrix(:,:,i);
    label = ones(dataSize(1),1)*labels(i);
    labeledVector = [labeledVector;tempData label];
end
end