function [] = update(W,V,eta,alpha,delta)

weightMatrixDelta(:,:,k) = -eta*(alpha*weightMatrixDelta(:,:,k)-(1-alpha)*delta*output');