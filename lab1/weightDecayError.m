function error = weightDecayError(output,target,lambda,weights)
    % weights: cell of matrices
    N = length(weights);
    mNorm = 0;
    for i = 1:N
        mNorm = mNorm + norm(weights{i},'fro');
    end
    error = (output-target) + lambda*mNorm;
end

        