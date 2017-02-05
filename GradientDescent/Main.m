%Loading data
data = load('data.txt');
X = data(:, 1);
y = data(:, 2);
m = length(y); 

%Ploting data
plotData(X, y);

X = [ones(m, 1), data(:,1)]; 

%Starting point
theta = zeros(2, 1); 

%Number of iterations
iterations = 1000;
alpha = 0.01;

%Cost for theta [0 0]
cost = computeCost(X, y, theta);

disp(sprintf('Cost for zeros theta,%.2d', cost));

%Gradient Descent
[theta, J_history] = gradientDescent(X, y, theta, alpha, iterations);

fprintf('\nTheta found by gradient descent: ');
fprintf('%f %f \n', theta(1), theta(2));

%Ploting results (linear regression)
hold on; 
plot(X(:,2), X * theta, '-')
legend('Training data', 'Linear regression')
hold off 

% Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5] * theta;

fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);

predict2 = [1, 7] * theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);


