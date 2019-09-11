function weightMatrixDelta = updateGeneral(weightMatrixDelta,eta,alpha,delta,output)

weightMatrixDelta = -eta*((weightMatrixDelta.*alpha)-(delta*output').*(1-alpha));
end