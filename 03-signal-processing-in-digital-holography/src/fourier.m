% Define your constants
lambda = 532E-9; % Wavelength of light
pixel_size = 3.45E-9
pixel_amount = 12
theta = asin(lambda / (2 * pixel_size)) % Angle between object and reference wavefronts
eta = sin(theta) / lambda; % Shift between wavefronts in frequency domain

% Load your hologram
H = imread('images/object12px.png');

% Perform Fourier transformation
F_H = fft2(H);

% Define your filters for the second and third terms
filter_2 = F_H;
filter_2(abs(fftshift(-eta)) > 0.01) = 0; % Replace 0.01 with your desired threshold
filter_3 = F_H;
filter_3(abs(fftshift(eta)) > 0.01) = 0; % Replace 0.01 with your desired threshold

% Display the filters
figure;
subplot(1, 2, 1);
imshow(log(abs(filter_2) + 1), []);
title('Filter for the second term');
subplot(1, 2, 2);
imshow(log(abs(filter_3) + 1), []);
title('Filter for the third term');

% Apply the filters
U0_term_2 = ifft2(ifftshift(filter_2));
U0_term_3 = ifft2(ifftshift(filter_3));

% Display the filters
figure;
subplot(1, 2, 1);
imshow(abs(U0_term_2), []);
title('Filter for the second term');
subplot(1, 2, 2);
imshow(abs(U0_term_3), []);
title('Filter for the third term');

% Calculate U0
U0 = U0_term_2 + U0_term_3;
