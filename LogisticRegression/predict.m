function p = predict(theta, X)

m = size(X, 1); % Number of training examples

p = zeros(m, 1);

p = sigmoid(X*theta);

for i = 1 : 100
    if p(i) >= 0.5
        p(i) = 1;
    else
        p(i) = 0;
    end
end

end
