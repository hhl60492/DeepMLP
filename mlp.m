function [model, mse, test_mse] = mlp(X, Y, h, maxiter, eta, mu, pc, test_input_set,test_response_set)
% Deep Multilayer Perceptron (DeepMLP) v0.2
% Input:
%   X: d x n data matrix
%   Y: p x n response matrix
%   h: L x 1 vector specify number of hidden nodes in each layer l
% Ouput:
%   model: model structure
%   mse: mean square error
% Original Code Written by Mo Chen (sth4nth@gmail.com).
% v0.2 Modifications for Nesterov momentum, biases and test MSE output
% written by Haihan Lan, UBC ECE (hhl60492@yahoo.com)
h = [size(X,1);h(:);size(Y,1)];
L = numel(h);
W = cell(L-1);
dW_prev = cell(L-1);

% build deep MLP
for l = 1:L-1
    W{l} = rand(h(l),h(l+1))-0.5; % input weights
    W{l,2} = rand(1,h(l+1))-0.5; % bias weights
    
    % input layer weights are a random lienar combination of the 1st and
    % 2nd principal components of the data set. This gives faster
    % convergence/better performance than pure random initialization
    if( l == 1)
        span1 = repmat(pc(:,1),1,h(l+1));
        span2 = repmat(pc(:,2),1,h(l+1));
        scalar1 = rand(h(l),h(l+1))-0.5;
        scalar2 = rand(h(l),h(l+1))-0.5;
        span =  scalar1.*span1 + scalar2.*span2;
        W{l} = span; % input weights
    
    end
    
    dW_prev{l} = W{l} .* ones(h(l),h(l+1)); % previous deltaW for momentum calculation
    dW_prev{l,2} = W{l,2} .* ones(1,h(l+1)); % previous deltaWBias
end
Z = cell(L);
Z{1} = X;

mse = zeros(1,maxiter);
test_mse = zeros(1,maxiter);

% training and test loop
for iter = 1:maxiter
    Z{1} = X;
    
    %     forward
    for l = 2:L
        biases = repmat(W{l-1,2}',1,size(Z{l-1},2));
        Z{l} = tanh_activation(W{l-1}'*Z{l-1} + biases);
    end
%     backward
    E = Y-Z{L};
    mse(iter) = 1/size(E,2) * (dot(E(:) , E(:)));
    fprintf('Iteration: %d MSE: %8.4f ', iter, mse(iter));
    
    for l = L-1:-1:1
        df = (ones(size(Z{l+1})) - Z{l+1} .* Z{l+1}); % tanh derivative
        dG = df.*E; % current layer delta
        dW = Z{l}*dG'; % current delta W
        
        dW = W{l} + eta(l) .* dW; % compute delta W for weights
       
        
        bias_values = ones([size(dG,2), 1])'; % bias values are set to 1 by default
        
        dWBias = bias_values*dG'; % bias deltaW
        dWBias = W{l,2} + eta(l) .* dWBias;
        
        W{l} = ((1 - mu) *  dW) + (mu * dW_prev{l}); % do weight update here
        W{l,2} =  ((1 - mu) * dWBias) + (mu * dW_prev{l,2}); % bias weight update
        E = W{l}*dG; % set error for next layer
        dW_prev{l} = dW; % update prev delta Ws
        dW_prev{l,2} = dWBias;
        
    end
    
    % validate (test) NN and get test MSE measure

    % feed test set data
    Z{1} = test_input_set';
    for l = 2:L
       biases = repmat(W{l-1,2}',1,size(Z{l-1},2));
       Z{l} = tanh_activation(W{l-1}'*Z{l-1} + biases);
    end
    Y_test = Z{L};

    % get test error
    E = (Y_test - test_response_set');
    test_mse(iter) = 1/size(E,2) * (dot(E(:),E(:)));
   
    fprintf('Test MSE: %8.4f\n', test_mse(iter));
    
    
    % shuffle data vector
    Z{1} = X(randperm(size(X,1)),:);
end
mse = mse(1:iter);
model.W = W;