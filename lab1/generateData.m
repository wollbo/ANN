%Generates nSamples number of data samples
function [data target] = generateData(nSamples, mean, sigma, labels) % mean = [mA mB]
if length(mean) ~= length(sigma)
    error('Not the same number of sigmas and means')
end

dataDist = zeros(nSamples,size(sigma,2),size(mean(1),1));

for i = 1:size(sigma,1)
    for j = 1:size(mean(1,:),2)
        dataDist(:,j,i) = randn(1,nSamples) .* sigma(i,j) + mean(i,j);
    end
end
[data target] = splitData(rearrangeData(labelData(dataDist,labels)));
end