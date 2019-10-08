function [shuffled,stargets] = shuffle(input1,input2,target1,target2)
%% Shuffles two vectors into one
% length(inputs) == length(targets)
len = length(input1)+length(input2);
unshuffled = [input1 input2];
targets = [target1 target2];
shuffled = zeros(size(unshuffled));
stargets = zeros(size(targets));
for i = 1:len
    decI = randi((len+1-i),1,1);
    shuffled(:,i) = unshuffled(:,decI);
    stargets(i) = targets(decI);
    unshuffled(decI) = [];
    targets(decI) = [];
end

