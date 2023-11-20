function balancedImage = whiteBalance(image)

% Check the range of the image data
minVal = min(image(:));
maxVal = max(image(:));

% If the range is not [0, 1], scale the image data
if minVal < 0 || maxVal > 1
    image = (image - minVal) / (maxVal - minVal);
    max(image, [], 'all')
end


% Convert to HSV
hsvImg = rgb2hsv(image);

% Find location of maximum intensity from the hsv representation
maxPixelValue = max(hsvImg(:,:,3), [], 'all');
[x,y] = find(hsvImg(:,:,3)==maxPixelValue, 1, 'first');


% Divide all the pixel by the found pixel in rgb space
balancedImage(:,:,1) = image(:,:,1) ./ image(x,y,1);
balancedImage(:,:,2) = image(:,:,2) ./ image(x,y,2);
balancedImage(:,:,3) = image(:,:,3) ./ image(x,y,3);

end