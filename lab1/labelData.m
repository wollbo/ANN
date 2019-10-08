function [labeledVector] = labelData(dataMatrix,labels)
dataSize = size(dataMatrix);
if max(size((dataSize))) >= 3
    nLables = dataSize(3);
else
    nLables = 1;
end

labeledVector = [];


for i = 1:nLables
    tempData = dataMatrix(:,:,i);
    label = ones(dataSize(1),1)*labels(i);
    labeledVector = [labeledVector;tempData label];
end
end