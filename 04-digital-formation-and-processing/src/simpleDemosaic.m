% function [demosaicked] = simpleDemosaic(image, channel)
% 
%     % Define the block size
%     blockSize = [4, 4];  % Adjust as needed
% 
%     % Create a custom function to be applied to each block
%     fun = @(block_struct) interpolateBlock(block_struct.data, channel);
% 
%     % Apply the function to each block
%     demosaicked = blockproc(image, blockSize, fun);
% end
% 
% function output = interpolateBlock(block, channel)
% 
%     % Get the size of the block
%     [rows, cols] = size(block);
% 
%     % Original points
%     [origX, origY] = meshgrid(1:cols, 1:rows);
% 
%     % Points to map to
% 
%     switch channel
%         case 'r'
%             [newX, newY] = meshgrid(1:1:cols*2, 1:1:rows*2);
%         case 'g1'
%             [newX, newY] = meshgrid(2:1:cols*2, 1:1:rows*2);
%         case 'g2'
%             [newX, newY] = meshgrid(1:1:cols*2, 2:1:rows*2);
%         case 'b'
%             [newX, newY] = meshgrid(2:1:cols*2, 2:1:rows*2);
%         otherwise
%             error('Invalid channel. Must be ''r'', ''g1'', ''g2'', or ''b''.');
%     end
% 
%     % Interpolate the specified channel
%     origX
%     origY
%     newX
%     newY
%     output = interp2(newX, newY, block, newX, newY, 'linear')
% 
% end

function [output] = simpleDemosaic(R, G1, G2, B)
    
    % Define output array;
    R_out = zeros(size(R)*2);
    G1_out = zeros(size(R)*2);
    G2_out = zeros(size(R)*2);
    B_out = zeros(size(R)*2);

    %R_out(1:2:end, 1:2:end) = R;  % Red
    %G1_out(1:2:end, 2:2:end) = G1;  % Green in Red rows
    %G2_out(2:2:end, 1:2:end) = G2;  % Green in Blue rows
    %B_out(2:2:end, 2:2:end) = B;  % Blue


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
    r_interp = interp2(Xr, Yr, R, Xout, Yout, 'linear', 0);
    g1_interp = interp2(Xg1, Yg1, G1, Xout, Yout, 'linear', 0);
    g2_interp = interp2(Xg2, Yg2, G2, Xout, Yout, 'linear', 0);
    b_interp = interp2(Xb, Yb, B, Xout, Yout, 'linear', 0);

    g_interp = (g1_interp + g2_interp);
    output = cat(3, r_interp, g_interp, b_interp);
end

