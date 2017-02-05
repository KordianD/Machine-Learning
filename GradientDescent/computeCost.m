function J = computeCost(X, y, theta)

result = ((y - X * theta).^2)/(2*length(y));
J = sum(result);

end