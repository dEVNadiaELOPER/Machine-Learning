function [J, grad] = cost(theta, X, y, lambda)

%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));



hypo=(sigmoid(X*theta));



J=(1/m)*sum( -y.*log(hypo) - (1-y).*(log(1-hypo)) ) + ((lambda/(2*m))*sum(theta(2:length(theta)).^2));
%could do same with ()'*X and remove the need for sum??
grad = (1/m)*(hypo - y)'*X;
grad(2:length(grad))= grad(2:length(grad))' + (lambda/m)*(theta(2:length(theta)));



% =============================================================

end