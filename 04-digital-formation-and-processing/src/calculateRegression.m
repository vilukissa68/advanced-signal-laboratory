% 5. Calculate regression lines
function [p] = calculateRegression(meanValues, varianceValues)
    % Fit a straight line to the data
    p = polyfit(meanValues, varianceValues, 1);
end