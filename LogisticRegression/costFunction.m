function [J, grad] = costFunction(theta, X, y)

m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));

hx = sigmoid(X * theta);

J = sum(-y' * log(hx) - (1 - y')*log(1 - hx)) / m;
grad = X' * (hx - y) / m;


end
