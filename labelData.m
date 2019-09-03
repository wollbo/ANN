function [labeledVector] = labelData(dataMatrix)
dataSize = size(dataMatrix);
nLables = dataSize(3);

labeledVector = [];

for i = 1:nLables
    tempData = dataMatrix(:,:,i);
    label = ones(dataSize(1),1)*i;
    labeledVector = [labeledVector;tempData label];
end
end