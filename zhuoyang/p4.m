% Project 4 - Texture Segmentation using Gabor Filters

% Clear workspace
clear all;
close all;
clc;

% Define functions used in the script
% =======================================================================
function h1 = getGaborFilterx(sigma, F, theta)
    % Length of truncated GEF
    n = 2*sigma;
    
    % Precompute look-up table for GEF - array size should be (4*sigma + 1)
    h1 = zeros(1, 4*sigma + 1);
    
    % Function uses the "degrees" versions of the sine and cosine functions
    % Creating a vector from -2*sigma to 2*sigma
    x_values = -n:1:n;
    
    for i = 1:length(x_values)
        x = x_values(i);
        g = exp(-(x^2)/(2*sigma^2)); % Don't need scale factor for Gaussian
        h1(i) = g*exp(1j*2*pi*F*(x*cosd(theta)));
    end
end

function h2 = getGaborFiltery(sigma, F, theta)
    % Length of truncated GEF
    n = 2*sigma;
    
    % Precompute look-up table for GEF
    h2 = zeros(1, 4*sigma + 1);
    
    % Function uses the "degrees" versions of the sine and cosine functions
    y_values = -n:1:n;
    
    for i = 1:length(y_values)
        y = y_values(i);
        g = exp(-(y^2)/(2*sigma^2)); % Don't need scale factor for Gaussian
        h2(i) = g*exp(1j*2*pi*F*(y*sind(theta)));
    end
end

function g_filter = getGaussianFilter(sigma)
    % Create a (4*sigma + 1) x (4*sigma + 1) Gaussian filter
    n = 2*sigma;
    filter_size = 4*sigma + 1;
    
    g_filter = zeros(filter_size, filter_size);
    
    for i = 1:filter_size
        for j = 1:filter_size
            x = i - (2*sigma + 1);
            y = j - (2*sigma + 1);
            g_filter(i, j) = exp(-(x^2 + y^2)/(2*sigma^2));
        end
    end
    
    % Normalize the filter
    g_filter = g_filter / sum(g_filter(:));
end

function [m, valid_region] = applyGaborFilter(I, F, theta, sigma)
    % Convert input to double if necessary
    if ~isa(I, 'double')
        I = double(I);
    end
    
    % Get image dimensions
    [rows, cols] = size(I);
    
    % Get Gabor filters
    h1 = getGaborFilterx(sigma, F, theta);
    h2 = getGaborFiltery(sigma, F, theta);
    
    % Calculate valid region
    n = 2*sigma;
    valid_x_min = n + 1;
    valid_x_max = cols - n;
    valid_y_min = n + 1;
    valid_y_max = rows - n;
    valid_region = [valid_y_min, valid_y_max, valid_x_min, valid_x_max];
    
    % Initialize intermediate and output images
    i1 = zeros(rows, cols);
    i2 = zeros(rows, cols, 'like', 1i); % Complex output
    m = zeros(rows, cols);
    
    % Step 1: Convolve with h1(x) in x-direction
    for y = valid_y_min:valid_y_max
        for x = valid_x_min:valid_x_max
            sum_val = 0;
            for x_prime = -n:n
                idx = x_prime + n + 1; % Index into h1
                sum_val = sum_val + I(y, x-x_prime) * h1(idx);
            end
            i1(y, x) = sum_val;
        end
    end
    
    % Step 2: Convolve with h2(y) in y-direction
    for y = valid_y_min:valid_y_max
        for x = valid_x_min:valid_x_max
            sum_val = 0;
            for y_prime = -n:n
                idx = y_prime + n + 1; % Index into h2
                sum_val = sum_val + i1(y-y_prime, x) * h2(idx);
            end
            i2(y, x) = sum_val;
        end
    end
    
    % Step 3: Take magnitude
    m(valid_y_min:valid_y_max, valid_x_min:valid_x_max) = abs(i2(valid_y_min:valid_y_max, valid_x_min:valid_x_max));
end

function [m_prime, valid_region] = applyGaussianSmoothing(m, sigma, gabor_valid_region)
    % Get dimensions of input
    [rows, cols] = size(m);
    
    % Get Gaussian filter
    g_filter = getGaussianFilter(sigma);
    
    % Calculate valid region after smoothing
    filter_size = 4*sigma + 1;
    half_size = 2*sigma;
    
    valid_y_min = gabor_valid_region(1) + half_size;
    valid_y_max = gabor_valid_region(2) - half_size;
    valid_x_min = gabor_valid_region(3) + half_size;
    valid_x_max = gabor_valid_region(4) - half_size;
    valid_region = [valid_y_min, valid_y_max, valid_x_min, valid_x_max];
    
    % Initialize output
    m_prime = zeros(rows, cols);
    
    % Apply smoothing filter
    for y = valid_y_min:valid_y_max
        for x = valid_x_min:valid_x_max
            sum_val = 0;
            for i = -half_size:half_size
                for j = -half_size:half_size
                    filter_i = i + half_size + 1;
                    filter_j = j + half_size + 1;
                    sum_val = sum_val + m(y+i, x+j) * g_filter(filter_i, filter_j);
                end
            end
            m_prime(y, x) = sum_val;
        end
    end
end

function [I, m, m_prime, segmented, valid_region] = textureSegmentation(img_path, F, theta, sigma, sigma_prime, use_smoothing, threshold)
    % Read image
    I = imread(img_path);
    
    % Convert to grayscale if it's a color image
    if size(I, 3) > 1
        I = rgb2gray(I);
    end
    
    % Apply Gabor filter
    [m, gabor_valid_region] = applyGaborFilter(I, F, theta, sigma);
    
    % Apply Gaussian smoothing if requested
    if use_smoothing && sigma_prime > 0
        [m_prime, valid_region] = applyGaussianSmoothing(m, sigma_prime, gabor_valid_region);
    else
        m_prime = [];
        valid_region = gabor_valid_region;
    end
    
    % Create segmentation
    segmented = zeros(size(I));
    
    % Threshold the output
    if use_smoothing && sigma_prime > 0
        % Threshold the smoothed output
        valid_y_min = valid_region(1);
        valid_y_max = valid_region(2);
        valid_x_min = valid_region(3);
        valid_x_max = valid_region(4);
        
        % Normalize m_prime for better thresholding
        m_prime_valid = m_prime(valid_y_min:valid_y_max, valid_x_min:valid_x_max);
        m_prime_norm = (m_prime_valid - min(m_prime_valid(:))) / (max(m_prime_valid(:)) - min(m_prime_valid(:)));
        
        segmented(valid_y_min:valid_y_max, valid_x_min:valid_x_max) = m_prime_norm > threshold;
    else
        % Threshold the Gabor filter output
        valid_y_min = gabor_valid_region(1);
        valid_y_max = gabor_valid_region(2);
        valid_x_min = gabor_valid_region(3);
        valid_x_max = gabor_valid_region(4);
        
        % Normalize m for better thresholding
        m_valid = m(valid_y_min:valid_y_max, valid_x_min:valid_x_max);
        m_norm = (m_valid - min(m_valid(:))) / (max(m_valid(:)) - min(m_valid(:)));
        
        segmented(valid_y_min:valid_y_max, valid_x_min:valid_x_max) = m_norm > threshold;
    end
end

function visualizeResults(I, m, m_prime, segmented, valid_region, title_text, has_smoothing)
    % Create figure for images
    fig1 = figure('Name', title_text, 'Position', [100, 100, 1200, 800]);
    
    % Extract valid region coordinates
    valid_y_min = valid_region(1);
    valid_y_max = valid_region(2);
    valid_x_min = valid_region(3);
    valid_x_max = valid_region(4);
    
    % Display original image
    subplot(2, 3, 1);
    imshow(I);
    title('Original Image');
    
    % Display Gabor filter output - only show valid region
    subplot(2, 3, 2);
    m_valid = m(valid_y_min:valid_y_max, valid_x_min:valid_x_max);
    
    % Create a new normalized version for display
    m_norm = (m_valid - min(m_valid(:))) / (max(m_valid(:)) - min(m_valid(:)));
    
    % Display only the valid region
    imshow(m_norm);
    title('Gabor Filter Output m(x,y)');
    
    % Display smoothed output if available
    if has_smoothing && ~isempty(m_prime)
        subplot(2, 3, 3);
        m_prime_valid = m_prime(valid_y_min:valid_y_max, valid_x_min:valid_x_max);
        
        % Find non-zero values in the smoothed output
        valid_mask = m_prime_valid > 0;
        
        % Only normalize using actual valid data
        if any(valid_mask(:))
            m_prime_min = min(m_prime_valid(valid_mask));
            m_prime_max = max(m_prime_valid(valid_mask));
            m_prime_norm = (m_prime_valid - m_prime_min) / (m_prime_max - m_prime_min);
        else
            m_prime_norm = zeros(size(m_prime_valid));
        end
        
        % Display only the valid region
        imshow(m_prime_norm);
        title('Smoothed Output m''(x,y)');
    end
    
    % Display segmentation result - only valid region
    subplot(2, 3, 4);
    segmented_valid = segmented(valid_y_min:valid_y_max, valid_x_min:valid_x_max);
    imshow(segmented_valid);
    title('Segmentation Result');
    
    % Display segmentation overlaid on original - only valid region
    subplot(2, 3, 5);
    I_valid = I(valid_y_min:valid_y_max, valid_x_min:valid_x_max);
    overlay = cat(3, I_valid, I_valid, I_valid); % Convert to RGB
    
    if isa(overlay, 'uint8')
        overlay = double(overlay) / 255;
    end
    
    % Create red overlay for segmented regions
    mask = zeros(size(I_valid));
    mask(segmented_valid > 0) = 1;
    
    for c = 1:3
        if c == 1 % Red channel
            overlay(:,:,c) = overlay(:,:,c) .* (1 - mask) + mask;
        else % Green and blue channels
            overlay(:,:,c) = overlay(:,:,c) .* (1 - mask);
        end
    end
    
    imshow(overlay);
    title('Segmentation Overlay');
    
    % Create 3D plot of filter output
    subplot(2, 3, 6);
    
    % Set colormap
    colormap(parula);
    
    if has_smoothing && ~isempty(m_prime)
        % Only plot non-zero values
        [rows, cols] = size(m_prime_valid);
        valid_indices = find(m_prime_valid > 0);
        
        if ~isempty(valid_indices)
            [valid_rows, valid_cols] = ind2sub([rows, cols], valid_indices);
            min_row = min(valid_rows);
            max_row = max(valid_rows);
            min_col = min(valid_cols);
            max_col = max(valid_cols);
            
            % Only plot the actually valid region
            [X, Y] = meshgrid(min_col:max_col, min_row:max_row);
            X = X + valid_x_min - 1;
            Y = Y + valid_y_min - 1;
            
            surf(X, Y, m_prime_valid(min_row:max_row, min_col:max_col), 'EdgeColor', 'none');
            shading interp;
            title('3D Plot of m''(x,y)');
        else
            text(0.5, 0.5, 'No valid data for 3D plot', 'HorizontalAlignment', 'center');
            axis off;
        end
    else
        % For m, plot the entire valid region
        [X, Y] = meshgrid(1:size(m_valid, 2), 1:size(m_valid, 1));
        X = X + valid_x_min - 1;
        Y = Y + valid_y_min - 1;
        
        surf(X, Y, m_valid, 'EdgeColor', 'none');
        shading interp;
        title('3D Plot of m(x,y)');
    end
    
    % Create a separate figure with 2D and 3D visualizations
    fig2 = figure('Name', [title_text ' - Surface Plot'], 'Position', [100, 500, 800, 400]);
    colormap(parula);
    
    if has_smoothing && ~isempty(m_prime)
        % Find valid non-zero region for smoothed output
        [rows, cols] = size(m_prime_valid);
        valid_indices = find(m_prime_valid > 0);
        
        if ~isempty(valid_indices)
            [valid_rows, valid_cols] = ind2sub([rows, cols], valid_indices);
            min_row = min(valid_rows);
            max_row = max(valid_rows);
            min_col = min(valid_cols);
            max_col = max(valid_cols);
            
            % 2D heatmap
            subplot(1, 2, 1);
            imagesc(m_prime_valid(min_row:max_row, min_col:max_col));
            title('m''(x,y)');
            axis equal tight;
            
            % 3D surface plot
            subplot(1, 2, 2);
            [X, Y] = meshgrid(min_col:max_col, min_row:max_row);
            X = X + valid_x_min - 1;
            Y = Y + valid_y_min - 1;
            
            surf(X, Y, m_prime_valid(min_row:max_row, min_col:max_col), 'EdgeColor', 'none');
            shading interp;
            title('m''(x,y) surface plot');
        else
            subplot(1, 2, 1);
            text(0.5, 0.5, 'No valid data for plot', 'HorizontalAlignment', 'center');
            axis off;
            
            subplot(1, 2, 2);
            text(0.5, 0.5, 'No valid data for 3D plot', 'HorizontalAlignment', 'center');
            axis off;
        end
    else
        % For m, plot the entire valid region
        subplot(1, 2, 1);
        imagesc(m_valid);
        title('m(x,y)');
        axis equal tight;
        
        subplot(1, 2, 2);
        [X, Y] = meshgrid(1:size(m_valid, 2), 1:size(m_valid, 1));
        X = X + valid_x_min - 1;
        Y = Y + valid_y_min - 1;
        
        surf(X, Y, m_valid, 'EdgeColor', 'none');
        shading interp;
        title('m(x,y) surface plot');
    end
end

% Main script starts here
% =======================================================================
disp('Running Texture Segmentation Project');
disp('====================================');

% Test 1: Texture2 with dash and slash textures
disp('Running Test 1: Texture2 with dash and slash textures');
disp('F = 0.059 cycles/pixel, θ = 135°, σ = 8, σ'' = 24');
[I1, m1, m1_prime, seg1, valid_region1] = textureSegmentation('texture2.gif', 0.059, 135, 8, 24, true, 0.5);
visualizeResults(I1, m1, m1_prime, seg1, valid_region1, 'Test 1: Texture2', true);

% Test 2: Texture1 with plus and L textures
disp('Running Test 2: Texture1 with plus and L textures');
disp('F = 0.042 cycles/pixel, θ = 0°, σ = 24, σ'' = 24');
[I2, m2, m2_prime, seg2, valid_region2] = textureSegmentation('texture1.gif', 0.042, 0, 24, 24, true, 0.5);
visualizeResults(I2, m2, m2_prime, seg2, valid_region2, 'Test 2: Texture1', true);

% Test 3: Brodatz textures d9d77 (grass lawn and cotton canvas)
disp('Running Test 3: Brodatz textures d9d77');
disp('F = 0.1 cycles/pixel, θ = 90°, σ = 16, No smoothing');
[I3, m3, ~, seg3, valid_region3] = textureSegmentation('d9d77.gif', 0.1, 90, 16, 0, false, 0.5);
visualizeResults(I3, m3, [], seg3, valid_region3, 'Test 3: d9d77', false);

% Test 4: Brodatz textures d4d29 (pressed cork and beach sand)
disp('Running Test 4: Brodatz textures d4d29');
disp('F = 0.1 cycles/pixel, θ = 0°, σ = 16, σ'' = 24');
[I4, m4, m4_prime, seg4, valid_region4] = textureSegmentation('d4d29.gif', 0.15, 0, 16, 24, true, 0.5);
visualizeResults(I4, m4, m4_prime, seg4, valid_region4, 'Test 4: d4d29', true);

% Display the valid regions for each test
disp('Valid regions for segmentation:');
disp(['Test 1: y=[', num2str(valid_region1(1)), ',', num2str(valid_region1(2)), '], x=[', num2str(valid_region1(3)), ',', num2str(valid_region1(4)), ']']);
disp(['Test 2: y=[', num2str(valid_region2(1)), ',', num2str(valid_region2(2)), '], x=[', num2str(valid_region2(3)), ',', num2str(valid_region2(4)), ']']);
disp(['Test 3: y=[', num2str(valid_region3(1)), ',', num2str(valid_region3(2)), '], x=[', num2str(valid_region3(3)), ',', num2str(valid_region3(4)), ']']);
disp(['Test 4: y=[', num2str(valid_region4(1)), ',', num2str(valid_region4(2)), '], x=[', num2str(valid_region4(3)), ',', num2str(valid_region4(4)), ']']);

