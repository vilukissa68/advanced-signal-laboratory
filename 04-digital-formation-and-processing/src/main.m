clc;
clear;

% Constants
lambda = 0.02; % Thresholding value used in DCT filtering
transformBlockSize = [3, 3];


% 1. Load image and convert to double
img = imread('images/outoffocus.tiff');
img = im2double(img);

% 2. Visualize Images, Bayer mosaic array
% Define the block size
imshow(img, [])

% 3. Separe image into subchannels
R = img(1:2:end, 1:2:end);
G1 = img(1:2:end, 2:2:end);
G2 = img(2:2:end, 1:2:end);
B = img(2:2:end, 2:2:end);

% Plot channels
figure;
subplot(2,2,1); imshow(R, []); title('Red');
subplot(2,2,2); imshow(G1, []); title('Green 1');
subplot(2,2,3); imshow(G2, []); title('Green 2');
subplot(2,2,4); imshow(B, []); title('Blue');

% 4. Sliding window
[meanValuesR, varianceValuesR] = calculateScatterPlot(R);
[meanValuesG1, varianceValuesG1] = calculateScatterPlot(G1);
[meanValuesB, varianceValuesB] = calculateScatterPlot(B);

% 5. Plot scatter plots and regression lines
pR = calculateRegression(meanValuesR, varianceValuesR);
pG1 = calculateRegression(meanValuesG1, varianceValuesG1);
pB = calculateRegression(meanValuesB, varianceValuesB);

produceScatterPlot(meanValuesR, varianceValuesR, pR, ...
    meanValuesG1, varianceValuesG1, pG1, ...
    meanValuesB, varianceValuesB, pB)

% 6. Transformation

% Define a function to apply the transformation
applyTransformation = @(block_struct, ac, bc)(2 * sqrt( (block_struct.data / ac) + (3/8) + (bc / (ac^2)) ))

% Apply the transformation using blockproc for each subchannel
transformedR = blockproc(R, transformBlockSize, @(block_struct) applyTransformation(block_struct, pR(1), pR(2)));
transformedG1 = blockproc(G1, transformBlockSize, @(block_struct) applyTransformation(block_struct, pG1(1), pG1(2)));
transformedB = blockproc(B, transformBlockSize, @(block_struct) applyTransformation(block_struct, pB(1), pB(2)));

% 7. Compute scatter plots
[meanValuesTransformedR, varianceValuesTransformedR] = calculateScatterPlot(transformedR);
[meanValuesTransformedG1, varianceValuesTransformedG1] = calculateScatterPlot(transformedG1);
[meanValuesTransformedB, varianceValuesTransformedB] = calculateScatterPlot(transformedB);

pTR = calculateRegression(meanValuesTransformedR, varianceValuesTransformedR);
pTG1 = calculateRegression(meanValuesTransformedG1, varianceValuesTransformedG1);
pTB = calculateRegression(meanValuesTransformedB, varianceValuesTransformedB);

produceScatterPlot(meanValuesTransformedR, varianceValuesTransformedR, pTR, ...
    meanValuesTransformedG1, varianceValuesTransformedG1, pTG1, ...
    meanValuesTransformedB, varianceValuesTransformedB, pTB)
%% 

lambda = 0.02

% 8. DCT
filteredRT = DCTImageDenoising(transformedR, lambda);
filteredG1T = DCTImageDenoising(transformedG1, lambda);
filteredBT = DCTImageDenoising(transformedB, lambda);

% 9. Inverse transformation
% Define a function for inverse transformation
inverseTransformation = @(block_struct, ac, bc)(ac * (0.25 * block_struct.data.^2 + ...
    0.25 * sqrt(3/2) * block_struct.data.^(-1) - ...
    (11/8)*(block_struct.data.^(-2)) + ...
    (5/8) * sqrt(3/2) * block_struct.data.^(-3) - ...
    1/8 - ...
    bc/(ac^2)))

% Apply inverse transformation for each channel
inverseR = blockproc(filteredRT, transformBlockSize, @(block_struct) inverseTransformation(block_struct, pR(1), pR(2)));
inverseG1 = blockproc(filteredG1T, transformBlockSize, @(block_struct) inverseTransformation(block_struct, pG1(1), pG1(2)));
inverseB = blockproc(filteredBT, transformBlockSize, @(block_struct) inverseTransformation(block_struct, pB(1), pB(2)));

figure;
subplot(3,1,1); imshow(inverseR, []); title('Red');
subplot(3,1,2); imshow(inverseG1, []); title('Green 1');
subplot(3,1,3); imshow(inverseB, []); title('Blue');