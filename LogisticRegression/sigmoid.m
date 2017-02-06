function g = sigmoid(z)

g = zeros(size(z));

m = ones(size(z));

g = 1./(m+exp(-z));

end
