function delta = backwardGeneral(activation,weightMatrix,delta)
%BACKWARDGENERAL Generalized backward step in hidden layer

[~,dPhi] = sigmoid(activation);
delta = dPhi.*weightMatrix*delta;

end

