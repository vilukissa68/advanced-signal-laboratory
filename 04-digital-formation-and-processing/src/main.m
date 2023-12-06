clc;
clear;

% Constants
lambda = 0.060; % Thresholding value used in DCT filtering NOTE: too small value here will cause major clipping due to max value pixels not being normalized
transformBlockSize = [32, 32];
dctBlockSize = [8 8];
%%

% 1. Load image and convert to double
focus = imread('images/outoffocus.tiff');
focus = im2double(focus);

natural = imread('images/natural.tiff');
natural = im2double(natural);


% 2. Visualize Images, Bayer mosaic array
% Define the block size
figure;
subplot(1,2,1); imshow(focus, []); title('out of focus image');
subplot(1,2,2); imshow(natural, []); title('natural');

% 3. Separe image into subchannels
R_focus = focus(1:2:end, 1:2:end);
G1_focus = focus(1:2:end, 2:2:end);
G2_focus = focus(2:2:end, 1:2:end);
B_focus = focus(2:2:end, 2:2:end);

R_nat = natural(1:2:end, 1:2:end);
G1_nat = natural(1:2:end, 2:2:end);
G2_nat = natural(2:2:end, 1:2:end);
B_nat = natural(2:2:end, 2:2:end);


% Illustrare the bayer array

% Plot channels
figure;
subplot(2,2,1); imshow(R_focus, []); title('Red');
subplot(2,2,2); imshow(G1_focus, []); title('Green 1');
subplot(2,2,3); imshow(G2_focus, []); title('Green 2');
subplot(2,2,4); imshow(B_focus, []); title('Blue');
sgtitle("Out of focus image");

figure;
subplot(2,2,1); imshow(R_nat, []); title('Red');
subplot(2,2,2); imshow(G1_nat, []); title('Green 1');
subplot(2,2,3); imshow(G2_nat, []); title('Green 2');
subplot(2,2,4); imshow(B_nat, []); title('Blue');
sgtitle("Natural image");
%% 

% 4. Sliding window
[meanValuesR_focus, varianceValuesR_focus] = calculateScatterPlot(R_focus);
[meanValuesG1_focus, varianceValuesG1_focus] = calculateScatterPlot(G1_focus);
[meanValuesG2_focus, varianceValuesG2_focus] = calculateScatterPlot(G2_focus);
[meanValuesB_focus, varianceValuesB_focus] = calculateScatterPlot(B_focus);


[meanValuesR_nat, varianceValuesR_nat] = calculateScatterPlot(R_nat);
[meanValuesG1_nat, varianceValuesG1_nat] = calculateScatterPlot(G1_nat);
[meanValuesG2_nat, varianceValuesG2_nat] = calculateScatterPlot(G2_nat);
[meanValuesB_nat, varianceValuesB_nat] = calculateScatterPlot(B_nat);


% 5. Plot scatter plots and regression lines

% Out of focus
pR_focus = calculateRegression(meanValuesR_focus, varianceValuesR_focus);
pG1_focus = calculateRegression(meanValuesG1_focus, varianceValuesG1_focus);
pG2_focus = calculateRegression(meanValuesG2_focus, varianceValuesG2_focus);
pB_focus = calculateRegression(meanValuesB_focus, varianceValuesB_focus);

% Natural
pR_nat = calculateRegression(meanValuesR_nat, varianceValuesR_nat);
pG1_nat = calculateRegression(meanValuesG1_nat, varianceValuesG1_nat);
pG2_nat = calculateRegression(meanValuesG2_nat, varianceValuesG2_nat);
pB_nat = calculateRegression(meanValuesB_nat, varianceValuesB_nat);

% Produce scatter plot for the out of focus image
produceScatterPlot(meanValuesR_focus, varianceValuesR_focus, pR_focus, ...
    meanValuesG1_focus, varianceValuesG1_focus, pG1_focus, ...
    meanValuesG2_focus, varianceValuesG2_focus, pG2_focus, ...
    meanValuesB_focus, varianceValuesB_focus, pB_focus)

% 6. Transformation
% Define transformation
% applyTransformation = @(block_struct, ac, bc)(2 * sqrt( (block_struct.data / ac) + (3/8) + (bc / (ac^2)) ));

% Apply the transformation using blockproc for each subchannel
transformedR_focus = blockproc(R_focus, transformBlockSize, @(block_struct) applyTransformation(block_struct, pR_focus(1), pR_focus(2)));
transformedG1_focus = blockproc(G1_focus, transformBlockSize, @(block_struct) applyTransformation(block_struct, pG1_focus(1), pG1_focus(2)));
transformedG2_focus = blockproc(G2_focus, transformBlockSize, @(block_struct) applyTransformation(block_struct, pG2_focus(1), pG2_focus(2)));
transformedB_focus = blockproc(B_focus, transformBlockSize, @(block_struct) applyTransformation(block_struct, pB_focus(1), pB_focus(2)));

transformedR_nat = blockproc(R_nat, transformBlockSize, @(block_struct) applyTransformation(block_struct, pR_nat(1), pR_nat(2)));
transformedG1_nat = blockproc(G1_nat, transformBlockSize, @(block_struct) applyTransformation(block_struct, pG1_nat(1), pG1_nat(2)));
transformedG2_nat = blockproc(G2_nat, transformBlockSize, @(block_struct) applyTransformation(block_struct, pG2_nat(1), pG2_nat(2)));
transformedB_nat = blockproc(B_nat, transformBlockSize, @(block_struct) applyTransformation(block_struct, pB_nat(1), pB_nat(2)));


% 7. Compute scatter plots
[meanValuesTransformedR_focus, varianceValuesTransformedR_focus] = calculateScatterPlot(transformedR_focus);
[meanValuesTransformedG1_focus, varianceValuesTransformedG1_focus] = calculateScatterPlot(transformedG1_focus);
[meanValuesTransformedG2_focus, varianceValuesTransformedG2_focus] = calculateScatterPlot(transformedG2_focus);
[meanValuesTransformedB_focus, varianceValuesTransformedB_focus] = calculateScatterPlot(transformedB_focus);

% Fit regression for out of focus image
pTR = calculateRegression(meanValuesTransformedR_focus, varianceValuesTransformedR_focus);
pTG1 = calculateRegression(meanValuesTransformedG1_focus, varianceValuesTransformedG1_focus);
pTG2 = calculateRegression(meanValuesTransformedG2_focus, varianceValuesTransformedG2_focus);
pTB = calculateRegression(meanValuesTransformedB_focus, varianceValuesTransformedB_focus);

% Plot the scatter plot
produceScatterPlot(meanValuesTransformedR_focus, varianceValuesTransformedR_focus, pTR, ...
    meanValuesTransformedG1_focus, varianceValuesTransformedG1_focus, pTG1, ...
    meanValuesTransformedG2_focus, varianceValuesTransformedG2_focus, pTG2, ...
    meanValuesTransformedB_focus, varianceValuesTransformedB_focus, pTB)
%% 

% 8. DCT

% Non-transformed
filteredR_nat = DCTImageDenoising(R_nat, lambda, dctBlockSize);
filteredG1_nat = DCTImageDenoising(G1_nat, lambda, dctBlockSize);
filteredG2_nat = DCTImageDenoising(G2_nat, lambda, dctBlockSize);
filteredB_nat = DCTImageDenoising(B_nat, lambda, dctBlockSize);

% Transformed
filteredRT_nat = DCTImageDenoising(transformedR_nat, lambda, dctBlockSize);
filteredG1T_nat = DCTImageDenoising(transformedG1_nat, lambda, dctBlockSize);
filteredG2T_nat = DCTImageDenoising(transformedG2_nat, lambda, dctBlockSize);
filteredBT_nat = DCTImageDenoising(transformedB_nat, lambda, dctBlockSize);

% 9. Inverse transformation
%Define a function for inverse transformation
% inverseTransformation = @(block_struct, ac, bc)(ac * (0.25 * block_struct.data.^2 + ...
%     0.25 * sqrt(3/2) * block_struct.data.^(-1) - ...
%     (11/8)*(block_struct.data.^(-2)) + ...
%     (5/8) * sqrt(3/2) * block_struct.data.^(-3) - ...
%     1/8 - ...
%     bc/(ac^2)))

% Apply inverse transformation for each channel

% Natural image
inverseRT_nat = blockproc(filteredRT_nat, transformBlockSize, @(block_struct) inverseTransformation(block_struct, pR_nat(1), pR_nat(2)));
inverseG1T_nat = blockproc(filteredG1T_nat, transformBlockSize, @(block_struct) inverseTransformation(block_struct, pG1_nat(1), pG1_nat(2)));
inverseG2T_nat = blockproc(filteredG2T_nat, transformBlockSize, @(block_struct) inverseTransformation(block_struct, pG2_nat(1), pG2_nat(2)));
inverseBT_nat = blockproc(filteredBT_nat, transformBlockSize, @(block_struct) inverseTransformation(block_struct, pB_nat(1), pB_nat(2)));

% Plot
figure;
subplot(2,2,1); imshow(inverseRT_nat, []); title('Red');
subplot(2,2,2); imshow(inverseG1T_nat, []); title('Green 1');
subplot(2,2,3); imshow(inverseG2T_nat, []); title('Green 2');
subplot(2,2,4); imshow(inverseBT_nat, []); title('Blue');
sgtitle("Transformed")


% 10. TODO: Compare images
%% 

% 11. Demosaicking

% Combine the channels into a color image
demosaicRGB = simpleDemosaic(filteredR_nat, filteredG1_nat, filteredG2_nat, filteredB_nat);
demosaicRGBT = simpleDemosaic(inverseRT_nat, inverseG1T_nat, inverseG2T_nat, inverseBT_nat);

% Remove extreme peak values from the image for better white balancing
%demosaicRGB = medianFilter(demosaicRGB, [32 32], 0.01);

% 12. White balancing
img = whiteBalance(demosaicRGB);
imgT = whiteBalance(demosaicRGBT);

%img_w = lin2rgb(demosaicRGB);
figure;
subplot(1,2,1); imshow(img, []); title("Non-transformed");
subplot(1,2,2); imshow(imgT, []); title("Transformed");
sgtitle("Demosaicking and white balancing natural image");


% 13. Contrast and saturation correction
imgCorrected = contrastAndSaturationCorrection(img, 0.9);
imgCorrectedT = contrastAndSaturationCorrection(imgT, 0.9);
figure;
subplot(1,2,1); imshow(imgCorrected, []); title("Non-transformed");
subplot(1,2,2); imshow(imgCorrectedT, []); title("Transformed");
sgtitle("Contrast and saturation correction for natural image");

%%
% Function definitions
% 6. Transformation
function output = applyTransformation(block_struct, ac, bc)
    % Calculate the value inside the square root
    sqrt_val = (block_struct.data / ac) + (3/8) + (bc / (ac^2));

    % Check if the value inside the square root is negative
    sqrt_val(sqrt_val < 0) = 0;

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
