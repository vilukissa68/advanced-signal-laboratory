function produceScatterPlot(meanR, varR, pR, meanG1, varG1, pG1, meanB, varB, pB)
    % Create a new figure
    figure;
    
    % Fit a straight line to the Red channel data and plot it
    subplot(3,1,1);
    scatter(meanR, varR);
    xlabel('Local Sample Mean');
    ylabel('Local Sample Variance');
    title('Mean-Variance Scatterplot for Red');
    hold on;
    fR = polyval(pR, meanR);
    plot(meanR, fR, 'r');
    hold off;
    
    % Fit a straight line to the Green 1 channel data and plot it
    subplot(3,1,2);
    scatter(meanG1, varG1);
    xlabel('Local Sample Mean');
    ylabel('Local Sample Variance');
    title('Mean-Variance Scatterplot for Green 1');
    hold on;
    fG1 = polyval(pG1, meanG1);
    plot(meanG1, fG1, 'g');
    hold off;
    
    % Fit a straight line to the Blue channel data and plot it
    subplot(3,1,3);
    scatter(meanB, varB);
    xlabel('Local Sample Mean');
    ylabel('Local Sample Variance');
    title('Mean-Variance Scatterplot for Blue');
    hold on;
    fB = polyval(pB, meanB);
    plot(meanB, fB, 'b');
    hold off;
end
