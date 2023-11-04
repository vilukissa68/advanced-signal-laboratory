 function [out] = plane_wave_decomp(in,lambda,X,x_out,z_out)
%PLANE_WAVE_DECOMP    Scalar field diffraction via plane wave decomposition.
%   One-dimensional implementation of the plane wave decomposition method
%   for scalar field diffraction. Usage:
%
%   PLANE_WAVE_DECOMP(in, lambda, X, x_out, z_out)
%
%   The function parameters:
%    in = Input signal (1D field) as an array.
%    lambda = Wavelength of the monochromatic light.
%    X = Sample step of the input (in meters). A single numerical value.
%    x_out = The x coordinates of the output samples (in meters) as an array.
%    z_out = The x coordinates of the output samples (in meters). Can be
%            given either as an array or a single constant.
%    out = Output field as an array.

% Tampere University
% Jani Mäkinen, 2019
 
 % frequencies
 N = numel(in);
 F_x = 1./(N*X).*(-N/2:N/2-1);
 F_z = sqrt(1/lambda^2-F_x.^2);
 
 % coefficients for plane waves
 in_dft = rot90(fft_centered(in));
 
 % check the size of the field (number of samples)
 if N >= 2^15
     % output field @ (x_out, z_out)
     out = zeros(size(z_out));
     % for each frequency, propagate the corresp. plane wave and sample at
     % arbitrary x and z
     for f_x_ind = 1:numel(F_x)
         a = in_dft(f_x_ind); % coefficient for this frequency
         f_x = F_x(f_x_ind); % frequency in x
         f_z = F_z(f_x_ind); % frequency in z
         % plane wave propagation and sampled at x,z
         out_f = a*exp(1j*2*pi*(f_x*x_out + f_z*z_out));
         % sum the contributions
         out = out + out_f; 
     end
 else
     % replace for-loop with matrix/vector operations 
     % NOTE: cannot be used with large N, takes too much memory
     out_mat = in_dft .* exp(1j*2*pi*(F_x'.*x_out + F_z'.*z_out));
     % sum the contributions
     out = sum(out_mat,1);
 end
