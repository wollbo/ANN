function weightMatrixDelta = updateGeneral(weightMatrixDelta,eta,alpha,delta,output)

weightMatrixDelta = -eta*(alpha*weightMatrixDelta-(1-alpha)*delta*output');
end