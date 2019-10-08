function initCell = initCells(cellStruct,eta)
    K = size(cellStruct,2);
    for k = 1:K
        dims = size(cellStruct{k});
        cellStruct{k} = eta*randn(dims);
    end
    initCell = cellStruct;
end
