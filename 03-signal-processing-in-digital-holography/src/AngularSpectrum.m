function [prop_wavefront, TF] = AngularSpectrum(wavefront,distance,lambda,dx)
% propagation using angular spectrum... 
% wavefront is a complex valued wavefront at the initial plane
% distance - propagation distance in m
% lambda is the wavelength in m
% dx - the sample spacings (computational pixel size) in the x and y in meters 
% 

%% Setup matrices representing reciprocal space coordinates
[L,K,~] = size(wavefront);
k = -K/2:K/2 - 1;
l = -L/2:L/2 - 1;
% l = l./(N*dy);
[k,l] = meshgrid(k,l);

U=1 -lambda^2*((k/(dx*K)).^2+(l/(dx*L)).^2);
TF= exp(1i*2*pi/lambda*distance*sqrt(U));
TF(U<0)=0;%%%%TRANSFER FUNCTION

% convolution
prop_wavefront =ifft2(ifftshift(TF.*fftshift(fft2((wavefront)))));

end

