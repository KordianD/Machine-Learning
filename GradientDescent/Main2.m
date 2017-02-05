%Loading data
data = load('dataMulti.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

%Normaliation for faster convergence
[X, mu, sigma] = featureNormalize(X);

%Added colum of ones
X = [ones(m, 1) X];

%Learning rate
alpha = 0.01;

%Number of iterations
num_iters = 400;
 
%Starting points for theta
theta = zeros(3, 1);

%Using gradient descent
[theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

