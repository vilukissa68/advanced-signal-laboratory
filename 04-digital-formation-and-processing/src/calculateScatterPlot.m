% 4. Calculate scatter plots
function [meanValues, varianceValues] = calculateScatterPlot(channel)
    % Define the window size for sliding window analysis
    windowSize = [2, 2];

    % Define a function to calculate mean and variance for each window
    calculateMeanVar = @(block_struct) [mean(block_struct.data(:)), var(block_struct.data(:))];

    % Apply the sliding window operator using blockproc
    meanVarResults = blockproc(channel, windowSize, calculateMeanVar);

    % Extract mean and variance values
    meanValues = meanVarResults(:, 1);
    varianceValues = meanVarResults(:, 2);

    % Reshape the results for scatterplot creation
    meanValues = meanValues(:);
    varianceValues = varianceValues(:);
end
