clear;
clc;

crop_size = [100, 100];

pixel_size = 3.45E-6;
pixel_amount = 5;

distance = 0.11;
lambda = 532E-9;

theta = asin((lambda * distance) / (pixel_size * pixel_amount));
eta = sin(theta)/lambda;

% Read interferogram to frequency domain
H = fftshift(fft2(imread("images/object5px.png")));
H_abs = abs(H);


% Crop
%mask = imbinarize(H_abs, "global");
%S = regionprops(mask,'BoundingBox','Area');
%[MaxArea,MaxIndex] = max(vertcat(S.Area));
%rect = S(MaxIndex).BoundingBox;

%H_abs = imcrop(H_abs, rect);

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
%imshow(log(abs(H)), [])

%%% TÄHÄN ASTI TOIMII

% Move to back to spatial domain
U0 = ifft2(ifftshift(H));


% Autofocus
min_distance = 0;
max_distance = 1;
num_samples=1000;

% Define a range of distances to sample
distances = linspace(min_distance, max_distance, num_samples);

% Initialize a variable to store the best distance and its corresponding focus measure
best_distance = 0;
best_focus_measure = -Inf;

% Loop over each distance
for d = distances
    % Perform wavefront reconstruction at the current distance
    [wavefront, TF] = AngularSpectrum(0, d, lambda, pixel_size);

    % Compute a focus measure (e.g., variance of Laplacian)
    PSF = abs(wavefront).^2;

    focus_measure = max(PSF(:).^2)


    % If the current focus measure is better than the best one found so far, update the best distance and focus measure
    if focus_measure > best_focus_measure
        best_distance = d;
        best_focus_measure = focus_measure;
    end
end
best_distance
best_focus_measure

% Now, best_distance should give you the distance that results in the most focused image
[prop_wavefront, TF] = AngularSpectrum(U0, best_distance, lambda, pixel_size);

% Amplitude
rec_abs = abs(prop_wavefront);

% Phase
rec_phase = angle(prop_wavefront);

% Create a new figure
figure;

% Plot rec_abs
subplot(1, 2, 1); % This creates a 1x2 grid of plots and selects the first plot
imshow(rec_abs, []);
title('Amplitude');

% Plot rec_phase
subplot(1, 2, 2); % This selects the second plot in the 1x2 grid
imshow(rec_phase, []);
title('Phase');

