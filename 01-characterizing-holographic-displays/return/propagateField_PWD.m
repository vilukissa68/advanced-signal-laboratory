function [result, u_sensor, X_u] = propagateField_PWD(U_x_after, lambda, X_x, N, eyeloc, z_focus, T)
%PROPAGATEFIELD_PWD    Numerical retinal image formation model.
%   One-dimensional numerical retinal image formation model. Simulates
%   light propagation with plane wave decomposition. Usage:
%
%   [result, u_sensor, X_u] = PROPAGATEFIELD_PWD(U_x_after, lambda, X_x, 
%                               N, eyeloc, z_focus, T)
%
%   The function parameters:
%    U_x_after = Input signal (1D field), i.e. the hologram, as an array.
%    lambda = Wavelength of the monochromatic light.
%    X_x = Sample step of the input (in meters). A single numerical value.
%    N = Number of samples in the input signal.
%    eyeloc = The location of the eye (in meters) as an array [x z].
%    z_focus = Focus distance of the simulated eye (in meters). A single
%              numerical value.
%    T = Pupil size of the simulated eye (in meters).
% 
%   Outputs:
%    result = The retinal "image" intensities as an array. 
%    u_sensor = Sensor grid (in meters) as an array.
%    X_u = The sensor grid sampling step (in meters).

% Tampere University
% Jani Mäkinen, 2019

%% Definition of basic parameters

% Viewer related parameters (position & focus)
x_eye = eyeloc(1);
z_eye = eyeloc(2);
l = 25e-3;  % l is fixed, the human eye focuses by changing f; i.e. works like a zoom lens.
f = (1/(z_eye-z_focus)+1/l)^-1; % focal length, z_focus w.r.t hologram

% Hologram plane parameters
W = X_x*N; % width (physical)

%% Sampling grids

% Hologram plane grid (x)
% x_holo = -N/2*X_x:X_x:N/2*X_x - X_x; % Only needed if the sampling is to
% be changed before propagation

% Lens plane grid (s)
X_s_lens = lambda*f/T; % X_s <= lambda*f/T: condition to avoid aliasing in sampling the lens transmittance function
if X_x < X_s_lens
    X_s_lens = X_x;
end
N_lens = ceil(W/X_s_lens); % Number of samples on the eye plane
s_max = X_s_lens*N_lens/2;
ss = -s_max:X_s_lens:s_max-X_s_lens; % Grid on s,t plane centered at x = 0
s_lens = ss + x_eye; % Shift according to the position of the eye

% Sensor plane grid (u)
W_u = T;
X_u= 1.22 * lambda * l/T; % Diffraction limited resolution
N_sensor = ceil(W_u/X_u);  
u_max = N_sensor/2*X_u;
u_sensor = -u_max:X_u:u_max - X_u;

%% Simulation: hologram -> image

% [1] Propagation from x to s (hologram to lens)
U_s_before_lens = plane_wave_decomp(U_x_after, lambda, X_x, s_lens, z_eye);

% [2] Field through the lens
% 1D eye pupil
lens = double( (abs(s_lens-x_eye)<= T/2) .* exp(-1j*pi/lambda/f*((s_lens-x_eye).^2)));
% Multiply the field with the lens transm. function to obtain the field after lens 
U_s_after = U_s_before_lens .* lens;

% [3] Propagation from s to u (lens to sensor)
U_u = plane_wave_decomp(U_s_after, lambda, X_s_lens, u_sensor, l);
im = (abs(U_u)).^2; % Intensity values on the sensor plane (amplitude^2)

% The image is mirrored on the retina, flip to get the correct one
result = fliplr(im);

end