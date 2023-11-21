function plotColorChannels(R, G1, G2, B)
    % Remove all the empty spaces for better plots
    % Create RGB images for each channel
    R_img = cat(3, R, zeros(size(R)), zeros(size(R)));
    G1_img = cat(3, zeros(size(G1)), G1, zeros(size(G1)));
    G2_img = cat(3, zeros(size(G2)), G2, zeros(size(G2)));
    B_img = cat(3, zeros(size(B)), zeros(size(B)), B);
    
    % Display the images
    figure;
    subplot(2, 2, 1); imshow(R_img, []); title('Red Channel');
    subplot(2, 2, 2); imshow(G1_img, []); title('Green1 Channel');
    subplot(2, 2, 3); imshow(G2_img, []); title('Green2 Channel');
    subplot(2, 2, 4); imshow(B_img, []); title('Blue Channel');
end