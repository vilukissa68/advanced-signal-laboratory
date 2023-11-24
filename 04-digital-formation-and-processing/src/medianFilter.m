function output = medianFilter(img, block_size, percentage)
    % Convert the image to the HSV color space
    img_hsv = rgb2hsv(img);

    % Calculate the threshold for the top percentage of intensities
    V = img_hsv(:,:,3);
    sorted_intensities = sort(V(:), 'descend');
    idx = round(percentage / 100 * numel(sorted_intensities));
    threshold = sorted_intensities(idx);

    % Define the function to apply to each block
    fun = @(block_struct) blockMedian(block_struct.data, threshold);

    % Apply the function to each block
    img_filtered = blockproc(img_hsv, block_size, fun);  % Adjust block size as needed

    % Convert the image back to the RGB color space
    output = hsv2rgb(img_filtered);
end

function block_out = blockMedian(block_in, threshold)
    % Apply the median filter to high-intensity pixels in the V channel
    V = block_in(:,:,3);
    high_intensity_pixels = V > threshold;
    V(high_intensity_pixels) = medfilt2(V(high_intensity_pixels), [9 9], 'symmetric');
    block_in(:,:,3) = V;

    block_out = block_in;
end
