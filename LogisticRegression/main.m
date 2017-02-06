%Load data
data = load('data1.txt');
X = data(:, [1, 2]);
y = data(:, 3);

%Plot data to see what is going on
plotData(X, y);

%Setting for plot
xlabel('Exam 1 score');
ylabel('Exam 2 score');
legend('Admitted', 'Not admitted')
hold off;

%Set the size of X
[m, n] = size(X);

% Add colums of ones
X = [ones(m, 1) X];

% Set initial theta
initial_theta = zeros(n + 1, 1);

% Cost function for initial theta
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
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
plotDecisionBoundary(theta, X, y);


prob = sigmoid([1 45 85] * theta);
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n'], prob);

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('\n');


