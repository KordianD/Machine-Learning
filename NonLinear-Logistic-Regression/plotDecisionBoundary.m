function plotDecisionBoundary(theta, X, y, degree)

% Plot Data
plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3
   
    plot_x = [min(X(:,2)),  max(X(:,2))];

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    axis([30, 100, 30, 100])
else
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j), degree)*theta;
        end
    end
    z = z'; 

    % Plot z = 0
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end
