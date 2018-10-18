function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% n = 27;
% m*(n+1) * (n+1)*1 = m*1
h = sigmoid(X * theta);

% theta = 27*1 vector; thetaReg (1+26)*1 vector; set theta0 = 0; regularized range:[1->n]
thetaReg = [0; theta(2:size(theta), 1)];
% 1*m * m*1 - 1*m * m*1 = 1*1; regularization part: coeff * 1*27 * 27*1 = 1*1
J = (((-y)' * log(h) - (1 - y)' * log(1 - h)) / m) + (lambda / (2 * m) * (thetaReg)' * thetaReg);

% calculate grads
% (n+1)*m * (m*1 - m*1) = (n+1)*1  (n+1 = #ofNodes/input include bias node 0)
grad = ((X' * (h - y)) + (lambda * thetaReg)) / m;




% =============================================================

end
