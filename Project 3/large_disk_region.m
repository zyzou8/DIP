clc; clear; close all;

% Read img
org_img = imread('disk.gif');
% Convert to grayscale if RGB
if size(org_img,3) == 3
    org_img = rgb2gray(org_img);
end

%init starting imgs
filtered_image = org_img;

for i =1:5
    filtered_image = mean_filter_5x5(filtered_image);
    %filtered_image = median_filter_5x5(filtered_image);
    %filtered_image = alpha_trimmed_mean_5x5(filtered_image,0.25);
    %filtered_image = sigma_filter_5x5(filtered_image,20);
    %filtered_image = snn_mean_filter_5x5(filtered_image);
end

% Get image dimensions
[rows, cols] = size(filtered_image);

% Create coordinate matrices
[x, y] = meshgrid(1:cols, 1:rows);

% Define disk center and radius (adjust these values based on your image)
center_x = 95;  % Assuming disk is centered
center_y = 115;  % Assuming disk is centered
radius = 60;  % Adjust this based on disk size

% Create circular mask
disk_mask = ((x - center_x).^2 + (y - center_y).^2) <= radius^2;

% Calculate statistics using the mask
pixels_in_disk = double(filtered_image(disk_mask));
disk_mean = mean(pixels_in_disk);
disk_std = std(pixels_in_disk);

% Display results
fprintf('Disk region statistics:\n');
fprintf('Mean: %.2f\n', disk_mean);
fprintf('Standard deviation: %.2f\n', disk_std);

% Optionally visualize the mask
figure;
subplot(1,3,1);imshow(org_img);
title('Original img');
subplot(1,3,2);imshow(filtered_image);
title('Original img');
subplot(1,3,3);imshow(disk_mask);
title('Disk Region Mask');

function [mean] = mean_filter_5x5(f) %from mean3x3.m
    % Get image dimensions
    [M, N] = size(f);
    
    % Convert f to a 16-bit number, so we can do  sums > 255 correctly
    
    g = uint16(f);
    
    % Define the coordinate limits for output pixels that can be properly
    %     computed by the 3X3 filter
    
    xlo = 3;   % Can't process first 2 column
    xhi = M-2; % Can't process last 2 column
    ylo = 3;   % Can't process first 2 row
    yhi = N-2; % Can't process last 2 row
    
    % Compute the filtered output image
    
    for x = xlo : xhi        % Don't consider boundary pixels that can't
        for y = ylo : yhi    %    be processed! 
            mean(x,y) = 0;
            for i = -2 : 2
                for j = -2 : 2   
                    mean(x,y) = mean(x,y) + g(x-i,y-j);
                end
            end
            mean(x,y) = mean(x,y) / 25.;
        end
    end
    
    % Convert back to an 8-bit image
    
    mean = uint8(mean);
end

function output = median_filter_5x5(input_img)
    % Use MATLAB's built-in median filter function
    output = medfilt2(input_img, [5 5]);
end

function output = alpha_trimmed_mean_5x5(input_img, alpha)
    % Alpha-trimmed mean filter using ordfilt2
    
    % Convert to double for processing
    input_img = double(input_img);
    
    % Define neighborhood size (5x5)
    nhood_size = 5;
    total_pixels = nhood_size * nhood_size;
    
    % Calculate number of pixels to trim from each end
    d = round(total_pixels * alpha / 2); % =3 if alpha=0.25
    
    % Lower and upper order statistics to keep
    low_order = d + 1;
    high_order = total_pixels - d;
    
    % Create matrix of zeros with same size as input
    output = zeros(size(input_img));
    
    % Compute sum of middle order statistics using ordfilt2
    for k = low_order:high_order
        output = output + ordfilt2(input_img, k, true(nhood_size));
    end
    
    % Calculate mean of the kept values
    output = output / (high_order - low_order + 1);
    
    output = uint8(output);
end

function output = sigma_filter_5x5(input_image, sigma)
    % Convert to double for processing
    input_image = double(input_image);
    
    % Get image dimensions
    [rows, cols] = size(input_image);
    output = zeros(rows, cols);
    
    % Pad the image to handle the boundaries
    padded_image = padarray(input_image, [2 2], 'replicate');
    
    % Process each pixel
    for i = 1:rows
        for j = 1:cols
            % Extract the 5x5 neighborhood
            neighborhood = padded_image(i:i+4, j:j+4);
            
            % Get the center pixel value
            center_value = padded_image(i+2, j+2);
            
            % Find pixels within 2*sigma range of the center pixel
            valid_pixels = neighborhood(abs(neighborhood - center_value) <= 2*sigma);
            
            % Calculate the mean of the valid pixels
            if ~isempty(valid_pixels)
                output(i, j) = mean(valid_pixels);
            else
                output(i, j) = center_value; % If no valid pixels, use center
            end
        end
    end
    output = uint8(output);
end

function output = snn_mean_filter_5x5(input_image)
    % Convert to double for processing
    input_image = double(input_image);
    % Get image dimensions
    [rows, cols] = size(input_image);
    output = zeros(rows, cols);
    
    % Pad the image to handle the boundaries
    padded_image = padarray(input_image, [2 2], 'replicate');
    
    % Define the symmetric pairs (based on the center pixel)
    % For a 5x5 filter, we have 12 symmetric pairs
    pairs = [
        -2, -2,  2,  2;  % corners
        -2, -1,  2,  1;
        -2,  0,  2,  0;
        -2,  1,  2, -1;
        -2,  2,  2, -2;  % corners
        -1, -2,  1,  2;
        -1, -1,  1,  1;
        -1,  0,  1,  0;
        -1,  1,  1, -1;
        -1,  2,  1, -2;
         0, -2,  0,  2;
         0, -1,  0,  1;
    ];
    
    % Process each pixel
    for i = 1:rows
        for j = 1:cols
            % Get the center pixel value
            center_value = padded_image(i+2, j+2);
            
            % Initialize sum for the mean calculation
            sum_value = center_value;  % Start with the center pixel
            
            % Check each symmetric pair
            for p = 1:size(pairs, 1)
                % Get the values at the symmetric positions
                val1 = padded_image(i+2+pairs(p,1), j+2+pairs(p,2));
                val2 = padded_image(i+2+pairs(p,3), j+2+pairs(p,4));
                
                % Choose the value closer to the center pixel
                if abs(val1 - center_value) <= abs(val2 - center_value)
                    sum_value = sum_value + val1;
                else
                    sum_value = sum_value + val2;
                end
            end
            
            % Calculate the mean (center + 12 chosen pixels = 13 total pixels)
            output(i, j) = sum_value / 13;
        end
    end
    output = uint8(output);
end