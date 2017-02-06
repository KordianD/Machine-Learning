function plotDecisionBoundary(theta, X, y)

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

end
end
