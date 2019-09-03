%Generates nSamples number of data samples
function [data] = generateData(nSamples, mean, sigma) % mean = [mA mB]
if length(mean) ~= length(sigma)
    error('Not the same number of sigmas and means')
end

data = zeros(nSamples,length(sigma),length(mean(1)));

for i = 1:length(sigma)
    for j = 1:length(mean(1,:))
        data(:,j,i) = randn(1,nSamples) .* sigma(i,j) + mean(i,j);
    end
end
end