function delta = backwardGeneral(activation,weightMatrix,delta)
%BACKWARDGENERAL Generalized backward step in hidden layer

[~,dPhi] = sigmoid2(activation);

delta = (weightMatrix'*delta);
% delta = delta(1:end-1,:).*dPhi.*(1-dPhi);
delta = delta(1:end-1,:).*dPhi;


end

