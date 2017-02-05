function [X_norm, mu, sigma] = featureNormalize(X)

mu = mean(X);
sigma = std(X);

X_norm = bsxfun(@minus, X, mu);
X_norm = bsxfun(@rdivide, X_norm, sigma);

end
