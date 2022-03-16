function [theta, J_history] = gradientDescent(x, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    h = theta' .* x;
    h = h(:,1) + h(:,2);
    delta0 = (h - y) .* x(:,1);
    delta1 = (h - y) .* x(:,2);
    delta0 = sum(delta0,'all');
    delta1 = sum(delta1,'all');
    delta = [delta0;delta1];
    
    theta = theta - (alpha * (1/m)) * delta;
    






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(x, y, theta);

end

end
