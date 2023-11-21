function produceScatterPlot(meanR, varR, pR, meanG1, varG1, pG1, meanG2, varG2, pG2, meanB, varB, pB)
    % Create a new figure
    figure;
    
    % Fit a straight line to the Red channel data and plot it
    subplot(2,2,1);
    scatter(meanR, varR);
    xlabel('Local Sample Mean');
    ylabel('Local Sample Variance');
    title('Mean-Variance Scatterplot for Red');
    hold on;
    fR = polyval(pR, meanR);
    plot(meanR, fR, 'r');
    hold off;
    
    % Fit a straight line to the Green 1 channel data and plot it
    subplot(2,2,2);
    scatter(meanG1, varG1);
    xlabel('Local Sample Mean');
    ylabel('Local Sample Variance');
    title('Mean-Variance Scatterplot for Green 1');
    hold on;
    fG1 = polyval(pG1, meanG1);
    plot(meanG1, fG1, 'g');
    hold off;

    % Fit a straight line to the Green 2 channel data and plot it
    subplot(2,2,3);
    scatter(meanG2, varG2);
    xlabel('Local Sample Mean');
    ylabel('Local Sample Variance');
    title('Mean-Variance Scatterplot for Green 2');
    hold on;
    fG2 = polyval(pG2, meanG2);
    plot(meanG2, fG2, 'g');
    hold off;
    
    % Fit a straight line to the Blue channel data and plot it
    subplot(2,2,4);
    scatter(meanB, varB);
    xlabel('Local Sample Mean');
    ylabel('Local Sample Variance');
    title('Mean-Variance Scatterplot for Blue');
    hold on;
    fB = polyval(pB, meanB);
    plot(meanB, fB, 'b');
    hold off;
end
