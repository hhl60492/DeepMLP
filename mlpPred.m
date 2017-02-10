function y = mlpPred(model, X)
% Multilayer perceptron prediction
% Input:
%   model: model structure
%   X: d x n data matrix
% Ouput:
%   Y: p x n response matrix
% Written by Mo Chen (sth4nth@gmail.com).
% Modifications by Haihan Lan, UBC ECE
W = model.W;
L = length(W)+1;
Z = cell(L);
Z{1} = X;
for l = 2:L
   biases = repmat(W{l-1,2}',1,size(Z{l-1},2));
   Z{l} = tanh_activation(W{l-1}'*Z{l-1} + biases);
end
y = Z{L};
