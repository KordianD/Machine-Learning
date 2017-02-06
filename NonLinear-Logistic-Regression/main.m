%Load data
load data2.txt;

%Assign to appropiate matrix
X = data2(:, [1 2]);
y = data2(:, 3);
m = length(y);

%It important to see how it looks
plotData(X, y);

% Labels and Legend
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')

% Specified in plot order
legend('y = 1', 'y = 0')
hold off;

%Since out data looks non linear, we need
%to change out data
X = [ones(m, 1), X, X(:,1).*X(:,2), X(:,1).^2, X(:,2).^2];

%Set theta, 
initial_theta = zeros(6,1);

%Cost function for initial theta
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf(' %f \n', grad);

%We use built-in function
% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Run fminunc to obtain the optimal theta
% This function will return theta and the cost 
[theta, cost] =  fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

% Plot Boundary
plotDecisionBoundary(theta, X, y, 2);

%We want to check our precision
p = predict(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);


