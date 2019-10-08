function [data target] = splitData(labledData)
dataSize = size(labledData);

data = labledData(:,1:dataSize(2)-1);
target = labledData(:,dataSize(2));
end