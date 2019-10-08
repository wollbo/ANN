function [numErr, MSE] = evalError(data, target)
numErr = sum(abs(data(:,end) ~= target));
MSE =   sum((data(:,end) ~= target).^2)/length(target);
end
