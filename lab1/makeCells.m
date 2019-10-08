function [cells] = makeCells(netStruct)
cells = cell(1,length(netStruct)-1)
for i = 2:(length(netStruct))
    cells{i-1} = zeros(netStruct(i),netStruct(i-1));
end
end