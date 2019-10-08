% cellTest

% little init ofc
cellVec = [2 2 1];
c = makeCells(cellVec);
eta = 0.01;
initC = initCells(c,eta);
x = randn(2,100);
t = randn(1,100);

%% forward

[a, o] = forward(initC,x);

%% backward

d = backward(t,initC,o{end},a);
