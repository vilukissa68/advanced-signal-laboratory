clc;
clear;

% 1. Load image and convert to double
img = imread('images/natural.tiff');
img = im2double(img);

% 2. Visualize Images, Bayer mosaic array
% Define the block size
imshow(img, [])

% 3. Separe image into subchannels
R = img(1:2:end, 1:2:end);
G1 = img(1:2:end, 2:2:end);
G2 = img(2:2:end, 1:2:end);
B = img(2:2:end, 2:2:end);

% Process the image in blocks
figure;
subplot(2,2,1); imshow(R, []); title('Red');
subplot(2,2,2); imshow(G1, []); title('Green 1');
subplot(2,2,3); imshow(G2, []); title('Green 2');
subplot(2,2,4); imshow(B, []); title('Blue');

% 4. Sliding window
% Define the window size for sliding window analysis
windowSize = [2, 2];

% Define a function to calculate mean and variance for each window
calculateMeanVar = @(block_struct) [mean(block_struct.data(:)), var(block_struct.data(:))];

% Apply the sliding window operator using blockproc for each subchannel
meanVarResultsR = blockproc(R, windowSize, calculateMeanVar);
meanVarResultsG1 = blockproc(G1, windowSize, calculateMeanVar);
meanVarResultsB = blockproc(B, windowSize, calculateMeanVar);

% Extract mean and variance values
meanValuesR = meanVarResultsR(:, 1);
varianceValuesR = meanVarResultsR(:, 2);

meanValuesG1 = meanVarResultsG1(:, 1);
varianceValuesG1 = meanVarResultsG1(:, 2);

meanValuesB = meanVarResultsB(:, 1);
varianceValuesB = meanVarResultsB(:, 2);

% Reshape the results for scatterplot creation
meanValuesR = meanValuesR(:);
varianceValuesR = varianceValuesR(:);

meanValuesG1 = meanValuesG1(:);
varianceValuesG1 = varianceValuesG1(:);

meanValuesB = meanValuesB(:);
varianceValuesB = varianceValuesB(:);

% 5. Plot scatter plots and regression lines
% Create a new figure
figure;

% Fit a straight line to the Red channel data and plot it
subplot(3,1,1);
scatter(meanValuesR, varianceValuesR);
xlabel('Local Sample Mean');
ylabel('Local Sample Variance');
title('Mean-Variance Scatterplot for Red');
hold on;
pR = polyfit(meanValuesR, varianceValuesR, 1);
fR = polyval(pR, meanValuesR);
plot(meanValuesR, fR, 'r');
hold off;

% Fit a straight line to the Green 1 channel data and plot it
subplot(3,1,2);
scatter(meanValuesG1, varianceValuesG1);
xlabel('Local Sample Mean');
ylabel('Local Sample Variance');
title('Mean-Variance Scatterplot for Green 1');
hold on;
pG1 = polyfit(meanValuesG1, varianceValuesG1, 1);
fG1 = polyval(pG1, meanValuesG1);
plot(meanValuesG1, fG1, 'g');
hold off;

% Fit a straight line to the Blue channel data and plot it
subplot(3,1,3);
scatter(meanValuesB, varianceValuesB);
xlabel('Local Sample Mean');
ylabel('Local Sample Variance');
title('Mean-Variance Scatterplot for Blue');
hold on;
pB = polyfit(meanValuesB, varianceValuesB, 1);
fB = polyval(pB, meanValuesB);
plot(meanValuesB, fB, 'b');
hold off;

% 6. Transformation

% Define a function to apply the transformation
applyTransformation = @(block_struct, ac, bc) sqrt(2*block_struct.data + 3/8 + bc) / ac^2;

% Apply the transformation using blockproc for each subchannel
transformedR = blockproc(R, [1 1], @(block_struct) applyTransformation(block_struct, pR(1), pR(2)));
transformedG1 = blockproc(G1, [1 1], @(block_struct) applyTransformation(block_struct, pG1(1), pG1(2)));
transformedB = blockproc(B, [1 1], @(block_struct) applyTransformation(block_struct, pB(1), pB(2)));

% 7. Compute scatter plots

