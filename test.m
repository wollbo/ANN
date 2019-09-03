%%test

testdata = randn(2,100);
matrix = randn(2,2,2);
targets = randn(1,100);
wmd = zeros(size(matrix));
eta = 0.1;
alpha = 0.9;

%%
for k = 1:10000
    [a,y] = forward(matrix,testdata);
    delta = backward(targets,matrix,y(:,:,end),a);
    dw = update(matrix,eta,alpha,delta,y,wmd);
    matrix = matrix+dw
end