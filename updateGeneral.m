function weightMatrixDelta = updateGeneral(weightMatrixDelta,alpha,delta,output)

weightMatrixDelta = (weightMatrixDelta.*alpha)-(delta*output').*(1-alpha);
end