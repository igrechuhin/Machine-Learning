function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

h_theta_y = X*theta - y;

J = (sum((h_theta_y).^2) + lambda*sum(theta(2:end).^2)) / (2*m);

gr = sum(repmat(h_theta_y, 1, size(X, 2)) .* X) / m;
th = theta;
th(1) = 0;
gr_reg = (lambda/m) * th;
grad = gr' + gr_reg;





% =========================================================================

grad = grad(:);

end
