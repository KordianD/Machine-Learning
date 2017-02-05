%Load data
load data.csv

x = data(:, 1);
y = data(:, 2);
m = length(y);

%Plot data
plotData(x, y);

%Add column of ones
x = [ones(m, 1), x];

%Set the theta value for 0
theta = zeros(2, 1); 

%Set learning rate and number of iterations
alpha = 0.00001;
numOfIterations = 400;

%Compute initial cost fot this theta
firstCost = computeCost(x, y, theta);

%Display initial cost
disp(firstCost);

%Use gradient descent to calculate optimum theta
[theta, J_history] = gradientDescent(x, y, theta, alpha, numOfIterations);

disp(theta);

%Use calculated theta to plot data
hold on;
plot(data(:,1), x * theta, '-');


%Adapt quadratic functions 
%We need to add one columns as x^2
x = [x, x(:,2).^2];

%Initialize theta one more time
theta2 = zeros(3, 1); 

%Set learning rate and number of iterations
alpha = 0.00000000001;
numOfIterations = 1000;

secondCost = computeCost(x, y, theta2);

%Use gradient descent to calculate optimum theta
[theta2, J_history] = gradientDescent(x, y, theta2, alpha, numOfIterations);

%Use calculated theta to plot data
hold on;
plot(x(:,2), x * theta2, 'b.');
legend('Points','Linear','Quadratic','Location','northwest','Orientation','vertical')
    







