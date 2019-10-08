function [a, z] = forwardGeneral(weightMatrix,input)
%FORWARDGENERAL Generalized forward
%   Input could come from previous layer!
a = weightMatrix*input;
z = sigmoid(a);
end

