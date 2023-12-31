function [denoised] = DCTImageDenoising(image, lambda, transformBlockSize)

    % Create a custom function to be applied to each block
    fun = @(block_struct) idct2(thresholdDCT(block_struct.data, lambda));

    % Apply the function to each block
    denoised = blockproc(image, transformBlockSize, fun);
end

function denoised = thresholdDCT(input, lambda)

    % Apply DCT to the block
    dctBlock = dct2(input);

    % Threshold the DCT coefficients
    dctBlock(abs(dctBlock) < lambda) = 0;
    denoised = dctBlock;

    % Apply inverse DCT
    %denoised = idct2(dctBlock);
end
