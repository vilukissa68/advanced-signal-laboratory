clear;
clc;

crop_size = [100, 100];

pixel_size = 3.45E-6;
pixel_amount = 12;

distance = 0.11;
lambda = 532E-9;

% Scan iteration variables
start_distance = -0.30
step_distance = 0.0001
end_distance = -0.29

% Move to back to spatial domain
U0 = ifft2(ifftshift(FourierFiltering("images/object12px.png")));

% Define your range of distances
distances = start_distance:step_distance:end_distance;

% Initialize the figures
figure(1);
abs_axes = gca;  % Get the current axes for the amplitude image
abs_img = imagesc(abs(AngularSpectrum(U0, distances(1), lambda, pixel_size)));
colormap(gray);  % Change the color map to grayscale
title(abs_axes, sprintf('Amplitude (distance = %.4f)', distances(1)));  % Set the title for the amplitude image
colorbar;

figure(2);
phase_axes = gca;  % Get the current axes for the phase image
phase_img = imagesc(angle(AngularSpectrum(U0, distances(1), lambda, pixel_size)));
colormap(gray);  % Change the color map to grayscale
title(phase_axes, sprintf('Phase (distance = %.4f)', distances(1)));  % Set the title for the phase image
colorbar;

% Add a slider to the first figure
figure(1);
slider = uicontrol('Style', 'slider', 'Position', [20 20 200 20], 'Value', 1,...
    'Min', 1, 'Max', length(distances), 'SliderStep', [1/(length(distances)-1) 1/(length(distances)-1)],...
    'Callback', @(src, event) update_images(src, event, distances, U0, lambda, pixel_size, abs_img, phase_img, abs_axes, phase_axes));



%%%% COMPARISONS
% Assume wavefront1 and wavefront2 are your reconstructed wavefronts

wavefront_with_obj = FourierFiltering('images/object5px.png');
wavefront_no_obj = FourierFiltering('images/object12px.png');

best_distance_1 = -0.2977
best_distance_2 = -0.2986

wavefront_with_obj = AngularSpectrum(wavefront_with_obj, best_distance_1, lambda, pixel_size);
wavefront_no_obj = AngularSpectrum(wavefront_no_obj, best_distance_2, lambda, pixel_size);

% Flatten images by taking mean of each row
amplitude1 = abs(wavefront_with_obj);
phase1 = angle(wavefront_with_obj);
amplitude2 = abs(wavefront_no_obj);
phase2 = angle(wavefront_no_obj);

amplitude1 = mean(amplitude1, 1);
phase1 = mean(phase1, 1);
amplitude2 = mean(amplitude2, 1);
phase2 = mean(phase2, 1);

% Create a new figure for the line plots
figure;

% Plot the amplitude along the line for both wavefronts
subplot(2, 1, 1);
plot(amplitude1, 'b'); % Plot the amplitude of the first wavefront in blue
hold on;
plot(amplitude2, 'r'); % Plot the amplitude of the second wavefront in red
hold off;
title('Amplitude along the line');
xlabel('Position along the line');
ylabel('Amplitude');
legend('5px', '12px');

% Plot the phase along the line for both wavefronts
subplot(2, 1, 2);
plot(phase1, 'b'); % Plot the phase of the first wavefront in blue
hold on;
plot(phase2, 'r'); % Plot the phase of the second wavefront in red
hold off;
title('Phase along the line');
xlabel('Position along the line');
ylabel('Phase');
legend('5px', '12px');

% This function is called when the slider position changes
function update_images(source, ~, distances, U0, lambda, pixel_size, abs_img, phase_img, abs_axes, phase_axes)
    % Compute the propagated wavefront for the current distance
    distance = distances(round(source.Value));
    prop_wavefront = AngularSpectrum(U0, distance, lambda, pixel_size);

    % Update the images
    set(abs_img, 'CData', abs(prop_wavefront));
    title(abs_axes, sprintf('Amplitude (distance = %.4f)', distance));  % Update the title for the amplitude image
    set(phase_img, 'CData', angle(prop_wavefront));
    title(phase_axes, sprintf('Phase (distance = %.4f)', distance));  % Update the title for the phase image
end




