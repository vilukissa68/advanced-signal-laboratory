function [output] = simpleDemosaic(R, G1, G2, B)

    % Define output array;
    R_out = zeros(size(R)*2);
    G1_out = zeros(size(R)*2);
    G2_out = zeros(size(R)*2);
    B_out = zeros(size(R)*2);

    % Get the size of the images
    [M, N] = size(R_out);

    % Create a meshgrid for interpolation
    [Xout, Yout] = meshgrid(1:N, 1:M);

    % Shift the grids for each channel according to the Bayer pattern
    [Xr, Yr] = meshgrid(1:2:N, 1:2:M);
    [Xg1, Yg1] = meshgrid(1:2:N, 2:2:M);
    [Xg2, Yg2] = meshgrid(2:2:N, 1:2:M);
    [Xb, Yb] = meshgrid(2:2:N, 2:2:M);

    % Interpolate the images
    r_interp = interp2(Xr, Yr, R, Xout, Yout, 'nearest', 0);
    g1_interp = interp2(Xg1, Yg1, G1, Xout, Yout, 'nearest', 0);
    g2_interp = interp2(Xg2, Yg2, G2, Xout, Yout, 'nearest', 0);
    b_interp = interp2(Xb, Yb, B, Xout, Yout, 'nearest', 0);

    g_interp = (g1_interp + g2_interp)/2;

    output = cat(3, r_interp, g_interp, b_interp);
end

