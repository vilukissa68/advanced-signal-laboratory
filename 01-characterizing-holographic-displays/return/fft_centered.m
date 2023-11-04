function out=fft_centered(in)
%%computes fft of signal defined on  index: in m=[m_0:m_0+N-1] out k=[k_0:k_0+N-1]

% Number of samples
N=length(in);

% input grid
m = (0:N-1);
m_0 = -N/2; % starting point of desired grid.

% output grid
% kk=0:N-1; 
% [l,k] = meshgrid(kk); 
k_0 = -N/2; % starting point of desired grid.

% out=fft2(in.*exp(-1j*2*pi*k_0*(m+n)/N)).*exp(-1j*2*pi*m_0*(k+l)/N)*exp(1j*2*pi*2*m_0*k_0/N);
% out=fft2(in.*exp(-1j*2*pi*k_0*(m+n)/N)).*exp(-1j*2*pi*m_0*(m+n)/N)*exp(1j*2*pi*2*m_0*k_0/N); % <- 2D
out = fft(in .* exp(-1j*2*pi*k_0*m./N)) .* exp(-1j*2*pi*m_0*m./N) * exp(1j*2*pi*2*m_0*k_0/N);
% clear in;


