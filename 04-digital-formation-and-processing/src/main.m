clc;
clear;

% Constants
lambda = 0.02; % Thresholding value used in DCT filtering
transformBlockSize = [16, 16];


% 1. Load image and convert to double
img = imread('images/outoffocus.tiff');
img = im2double(img);

% 2. Visualize Images, Bayer mosaic array
% Define the block size
figure;
imshow(img, [])

% 3. Separe image into subchannels
R = img(1:2:end, 1:2:end);
G1 = img(2:2:end, 1:2:end);
G2 = img(1:2:end, 2:2:end);
B = img(2:2:end, 2:2:end);

[M, N] = size(img);
% Initialize the channels
%R = zeros(M, N, 'double');
%G1 = zeros(M, N, 'double');
%G2 = zeros(M, N, 'double');
%B = zeros(M, N, 'double');

% Separate the channels
%R(1:2:end, 1:2:end) = img(1:2:end, 1:2:end);  % Red
%G1(1:2:end, 2:2:end) = img(1:2:end, 2:2:end);  % Green in Red rows
%G2(2:2:end, 1:2:end) = img(2:2:end, 1:2:end);  % Green in Blue rows
%B(2:2:end, 2:2:end) = img(2:2:end, 2:2:end);  % Blue

%plotColorChannels(R, G1, G2, B)

% Plot channels
figure;
subplot(2,2,1); imshow(R, []); title('Red');
subplot(2,2,2); imshow(G1, []); title('Green 1');
subplot(2,2,3); imshow(G2, []); title('Green 2');
subplot(2,2,4); imshow(B, []); title('Blue');
%% 

% 4. Sliding window
[meanValuesR, varianceValuesR] = calculateScatterPlot(R);
[meanValuesG1, varianceValuesG1] = calculateScatterPlot(G1);
[meanValuesG2, varianceValuesG2] = calculateScatterPlot(G2);
[meanValuesB, varianceValuesB] = calculateScatterPlot(B);

% 5. Plot scatter plots and regression lines
pR = calculateRegression(meanValuesR, varianceValuesR);
pG1 = calculateRegression(meanValuesG1, varianceValuesG1);
pG2 = calculateRegression(meanValuesG2, varianceValuesG2);
pB = calculateRegression(meanValuesB, varianceValuesB);

produceScatterPlot(meanValuesR, varianceValuesR, pR, ...
    meanValuesG1, varianceValuesG1, pG1, ...
    meanValuesG2, varianceValuesG2, pG2, ...
    meanValuesB, varianceValuesB, pB)

% 6. Transformation
% Define transformation
% applyTransformation = @(block_struct, ac, bc)(2 * sqrt( (block_struct.data / ac) + (3/8) + (bc / (ac^2)) ));

% Apply the transformation using blockproc for each subchannel
transformedR = blockproc(R, transformBlockSize, @(block_struct) applyTransformation(block_struct, pR(1), pR(2)));
transformedG1 = blockproc(G1, transformBlockSize, @(block_struct) applyTransformation(block_struct, pG1(1), pG1(2)));
transformedG2 = blockproc(G2, transformBlockSize, @(block_struct) applyTransformation(block_struct, pG2(1), pG2(2)));
transformedB = blockproc(B, transformBlockSize, @(block_struct) applyTransformation(block_struct, pB(1), pB(2)));

% 7. Compute scatter plots
[meanValuesTransformedR, varianceValuesTransformedR] = calculateScatterPlot(transformedR);
[meanValuesTransformedG1, varianceValuesTransformedG1] = calculateScatterPlot(transformedG1);
[meanValuesTransformedG2, varianceValuesTransformedG2] = calculateScatterPlot(transformedG2);
[meanValuesTransformedB, varianceValuesTransformedB] = calculateScatterPlot(transformedB);

pTR = calculateRegression(meanValuesTransformedR, varianceValuesTransformedR);
pTG1 = calculateRegression(meanValuesTransformedG1, varianceValuesTransformedG1);
pTG2 = calculateRegression(meanValuesTransformedG2, varianceValuesTransformedG2);
pTB = calculateRegression(meanValuesTransformedB, varianceValuesTransformedB);

produceScatterPlot(meanValuesTransformedR, varianceValuesTransformedR, pTR, ...
    meanValuesTransformedG1, varianceValuesTransformedG1, pTG1, ...
    meanValuesTransformedG2, varianceValuesTransformedG2, pTG2, ...
    meanValuesTransformedB, varianceValuesTransformedB, pTB)
%% 

% 8. DCT
filteredRT = DCTImageDenoising(transformedR, lambda);
filteredG1T = DCTImageDenoising(transformedG1, lambda);
filteredG2T = DCTImageDenoising(transformedG2, lambda);
filteredBT = DCTImageDenoising(transformedB, lambda);

% 9. Inverse transformation
%Define a function for inverse transformation
% inverseTransformation = @(block_struct, ac, bc)(ac * (0.25 * block_struct.data.^2 + ...
%     0.25 * sqrt(3/2) * block_struct.data.^(-1) - ...
%     (11/8)*(block_struct.data.^(-2)) + ...
%     (5/8) * sqrt(3/2) * block_struct.data.^(-3) - ...
%     1/8 - ...
%     bc/(ac^2)))

% Apply inverse transformation for each channel
inverseR = blockproc(filteredRT, transformBlockSize, @(block_struct) inverseTransformation(block_struct, pR(1), pR(2)));
inverseG1 = blockproc(filteredG1T, transformBlockSize, @(block_struct) inverseTransformation(block_struct, pG1(1), pG1(2)));
inverseG2 = blockproc(filteredG2T, transformBlockSize, @(block_struct) inverseTransformation(block_struct, pG2(1), pG2(2)));
inverseB = blockproc(filteredBT, transformBlockSize, @(block_struct) inverseTransformation(block_struct, pB(1), pB(2)));

figure;
subplot(2,2,1); imshow(inverseR, []); title('Red');
subplot(2,2,2); imshow(inverseG1, []); title('Green 1');
subplot(2,2,3); imshow(inverseG2, []); title('Green 2');
subplot(2,2,4); imshow(inverseB, []); title('Blue');

% 10. TODO: Compare images
%% 

% 11. Demosaicking

% Combine the channels into a color image
demosaicRGB = simpleDemosaic(inverseR, inverseG1, inverseG2, inverseB);

%inverseG = (inverseG1 + inverseG2) ./ 2;
%output = cat(3, inverseR, inverseG, inverseB);
%img_w = whiteBalance(output);

img_w = whiteBalance(demosaicRGB);
figure;
imshow(img_w, []);

%% 
% Function definitions
% 6. Transformation
function output = applyTransformation(block_struct, ac, bc)
    % Calculate the value inside the square root
    sqrt_val = (block_struct.data / ac) + (3/8) + (bc / (ac^2));

    % Check if the value inside the square root is negative
    sqrt_val(sqrt_val < 0) = 1;

    % Calculate the output
    output = 2 * sqrt(sqrt_val);
end
% 
% % 9. Inverse transformation
function output = inverseTransformation(block_struct, ac, bc)
    % Calculate the values inside the square roots
    sqrt_val1 = 0.25 * sqrt(3/2) * block_struct.data.^(-1);
    sqrt_val2 = (5/8) * sqrt(3/2) * block_struct.data.^(-3);

    % Check if the values inside the square roots are negative
    sqrt_val1(block_struct.data.^(-1) < 0) = 0;
    sqrt_val2(block_struct.data.^(-3) < 0) = 0;

    % Calculate the output
    output = ac * (0.25 * block_struct.data.^2 + ...
        sqrt_val1 - ...
        (11/8)*(block_struct.data.^(-2)) + ...
        sqrt_val2 - ...
        1/8 - ...
        bc/(ac^2));
end
