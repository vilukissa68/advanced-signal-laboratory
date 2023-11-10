function H = FourierFiltering(image_path)
    % Read interferogram to frequency domain
    H = fftshift(fft2(imread(image_path)));
    H_abs = abs(H);

    % Crop
    mask = imbinarize(H_abs, "global");
    S = regionprops(mask,'BoundingBox','Area');
    [MaxArea,MaxIndex] = max(vertcat(S.Area));
    rect = S(MaxIndex).BoundingBox;
    H_abs = imcrop(H_abs, rect);

    % Zero centrum
    [width, height, c] = size(H_abs);
    radius = width * 0.05;
    for ii=1:width
        for jj=1:height
            x = ii - width/2;
            y = jj - height/2;

            if (sqrt(x^2 + y^2) < radius)
                H_abs(ii, jj) = 0;
            end
        end
    end

    % Zero left side of spectrum
    H_abs(:,1:width/2) = 0;

    % Zero top of spectrum
    H_abs(1:height/2,:) = 0;

    % Shift 2nd order to center
    maximum = max(max(H_abs));
    [x0, y0] = find(H_abs==maximum);

    H_tmp = zeros(width,height);
    x0 = x0 - width/2 - 1;
    y0 = y0 - height/2 - 1;
    for ii = 1:width-x0
        for jj = 1:height-y0    
            H_tmp(ii, jj) = H(ii+x0,jj+y0); 
        end
    end
    H = H_tmp;

    % Zeroing all but the center
    H_tmp = zeros(width,height);
    for ii=1:width
        for jj=1:height

        x = ii - width/2;
        y = jj - height/2;

        if (sqrt(x^2 + y^2) < radius) 
            H_tmp(ii, jj) = H(ii, jj); 
        end
        end
    end

    H = H_tmp;
end
