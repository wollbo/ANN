%Label and rearrange column datavector
function [randomData] = rearrangeData(dataVector)
randomData = dataVector(randperm(size(dataVector,1)),:);
end 
