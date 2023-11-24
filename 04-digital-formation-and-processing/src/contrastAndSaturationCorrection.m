function img_corrected = contrastAndSaturationCorrection(img, gamma_value)
    % Convert the image to the HSV color space
    img_hsv = rgb2hsv(img);

    % Perform histogram equalization on the V channel for contrast correction
    img_hsv(:,:,3) = histeq(img_hsv(:,:,3));

    % Perform gamma correction on the S channel for saturation correction
    img_hsv(:,:,2) = img_hsv(:,:,2).^gamma_value;

    % Convert the image back to the RGB color space
    img_corrected = hsv2rgb(img_hsv);
end
