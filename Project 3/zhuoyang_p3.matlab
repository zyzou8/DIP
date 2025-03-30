function [mean_img] = mean5x5(f)
% 5×5 Mean filter implementation
[M, N] = size(f);
mean_img = zeros(M, N);

% Convert f to double for calculations
g = double(f);

% Define the coordinate limits for output pixels
xlo = 3; % Can't process first two columns
xhi = M-2; % Can't process last two columns
ylo = 3; % Can't process first two rows
yhi = N-2; % Can't process last two rows

% Compute the filtered output image
for x = xlo:xhi
    for y = ylo:yhi
        sum_val = 0;
        for i = -2:2
            for j = -2:2
                sum_val = sum_val + g(x+i, y+j);
            end
        end
        mean_img(x,y) = sum_val / 25;
    end
end



% Handle boundaries (you can modify this based on your preference)
mean_img = uint8(mean_img);
end


function [median_img] = median5x5(f)
% 5×5 Median filter using MATLAB's built-in function
median_img = medfilt2(f, [5 5]);
end



function [alpha_trim_img] = alphaTrimMean5x5(f, alpha)
% 5×5 Alpha-trimmed mean filter implementation
% alpha = 0.25 means trim 25% of values (discard 6 lowest and 6 highest)

[M, N] = size(f);
alpha_trim_img = zeros(M, N);

% Define the coordinate limits for output pixels
xlo = 3;
xhi = M-2;
ylo = 3;
yhi = N-2;

% Number of pixels to trim from each end
n_trim = round(25 * alpha);  % 25 is the total number of pixels in 5x5 window

% Compute the filtered output image
for x = xlo:xhi
    for y = ylo:yhi
        % Extract the 5x5 window
        window = f(x-2:x+2, y-2:y+2);

        % Reshape to 1D array and sort
        sorted_vals = sort(window(:));

        % Trim alpha% from each end
        trimmed_vals = sorted_vals(n_trim+1:end-n_trim);

        % Compute mean of remaining values
        alpha_trim_img(x,y) = mean(trimmed_vals);
    end
end

alpha_trim_img = uint8(alpha_trim_img);
end



function [sigma_img] = sigmaFilter5x5(f, sigma)
% 5×5 Sigma filter implementation
% Only includes pixels whose values are within σ of the center pixel

[M, N] = size(f);
sigma_img = zeros(M, N);

% Define the coordinate limits for output pixels
xlo = 3;
xhi = M-2;
ylo = 3;
yhi = N-2;

% Compute the filtered output image
for x = xlo:xhi
    for y = ylo:yhi
        center_val = double(f(x, y));
        sum_val = 0;
        count = 0;

        % Process the 5x5 window
        for i = -2:2
            for j = -2:2
                neighbor_val = double(f(x+i, y+j));

                % Check if the neighbor is within sigma of the center
                if abs(neighbor_val - center_val) <= sigma
                    sum_val = sum_val + neighbor_val;
                    count = count + 1;
                end
            end
        end

        % Compute average of valid neighbors
        if count > 0
            sigma_img(x,y) = sum_val / count;
        else
            sigma_img(x,y) = center_val;  % If no valid neighbors, use center
        end
    end
end

sigma_img = uint8(sigma_img);
end


function [snn_img] = symmetricNNMean5x5(f)
% 5×5 Symmetric Nearest Neighbor Mean filter
% For each pixel, take the average of pairs that are symmetric about center

[M, N] = size(f);
snn_img = zeros(M, N);

% Define the coordinate limits for output pixels
xlo = 3;
xhi = M-2;
ylo = 3;
yhi = N-2;

% Compute the filtered output image
for x = xlo:xhi
    for y = ylo:yhi
        center_val = double(f(x, y));
        sum_val = center_val;  % Include center pixel
        count = 1;

        % Process symmetric pairs in the 5x5 window
        for i = -2:2
            for j = -2:2
                if i == 0 && j == 0
                    continue;  % Skip center pixel
                end

                % Get symmetric position about center
                sym_i = -i;
                sym_j = -j;

                % Get values at symmetric positions
                val1 = double(f(x+i, y+j));
                val2 = double(f(x+sym_i, y+sym_j));

                % Use the value closer to the center
                if abs(val1 - center_val) <= abs(val2 - center_val)
                    sum_val = sum_val + val1;
                else
                    sum_val = sum_val + val2;
                end
                count = count + 1;
            end
        end

        snn_img(x,y) = sum_val / count;
    end
end

snn_img = uint8(snn_img);
end


% Main script to run all filters on the disk image
% Load the disk image
disk = imread('disk.gif');
if islogical(disk)
    disk = uint8(disk) * 255;  % Convert binary to uint8 if needed
end

% Get image dimensions
[M, N] = size(disk);

% Apply each filter for 1 iteration
mean_1iter = mean5x5(disk);
median_1iter = median5x5(disk);
alpha_1iter = alphaTrimMean5x5(disk, 0.25);
sigma_1iter = sigmaFilter5x5(disk, 20);
snn_1iter = symmetricNNMean5x5(disk);

% Apply each filter for 5 iterations
mean_5iter = mean_1iter;
median_5iter = median_1iter;
alpha_5iter = alpha_1iter;
sigma_5iter = sigma_1iter;
snn_5iter = snn_1iter;

for i = 2:5
    mean_5iter = mean5x5(mean_5iter);
    median_5iter = median5x5(median_5iter);
    alpha_5iter = alphaTrimMean5x5(alpha_5iter, 0.25);
    sigma_5iter = sigmaFilter5x5(sigma_5iter, 20);
    snn_5iter = symmetricNNMean5x5(snn_5iter);
end

% Display results for 1 iteration
figure(1);
subplot(2,3,1); imshow(disk); title('Original');
subplot(2,3,2); imshow(mean_1iter); title('Mean (1 iter)');
subplot(2,3,3); imshow(median_1iter); title('Median (1 iter)');
subplot(2,3,4); imshow(alpha_1iter); title('Alpha-Trim (1 iter)');
subplot(2,3,5); imshow(sigma_1iter); title('Sigma (1 iter)');
subplot(2,3,6); imshow(snn_1iter); title('SNN (1 iter)');

% Display results for 5 iterations
figure(2);
subplot(2,3,1); imshow(disk); title('Original');
subplot(2,3,2); imshow(mean_5iter); title('Mean (5 iter)');
subplot(2,3,3); imshow(median_5iter); title('Median (5 iter)');
subplot(2,3,4); imshow(alpha_5iter); title('Alpha-Trim (5 iter)');
subplot(2,3,5); imshow(sigma_5iter); title('Sigma (5 iter)');
subplot(2,3,6); imshow(snn_5iter); title('SNN (5 iter)');

% Display histograms for 5 iterations
figure(3);
subplot(2,3,1); imhist(disk); title('Original Histogram');
subplot(2,3,2); imhist(mean_5iter); title('Mean (5 iter) Hist');
subplot(2,3,3); imhist(median_5iter); title('Median (5 iter) Hist');
subplot(2,3,4); imhist(alpha_5iter); title('Alpha-Trim (5 iter) Hist');
subplot(2,3,5); imhist(sigma_5iter); title('Sigma (5 iter) Hist');
subplot(2,3,6); imhist(snn_5iter); title('SNN (5 iter) Hist');

% Calculate statistics for the interior of the large disk
% Define a ROI based on the semi-circle dimensions provided

% Create a mask for the semi-circle region
[rows, cols] = size(disk);  % Get dimensions of the disk image
mask = zeros(rows, cols);   % Initialize mask

% Center point
center_y = 90;
center_x = 114;

% Semi-circle dimensions
y_radius = 122/2;
x_radius = 134/2;

% Create the mask
for x = 1:rows
    for y = 1:cols
        % Check if point is within the semi-circle
        if ((x - center_x)^2 / x_radius^2 + (y - center_y)^2 / y_radius^2 <= 1)
            mask(x, y) = 1;
        end
    end
end

% Visualize the mask
figure;
subplot(1,2,1);
imshow(disk);
title('Original Disk Image');

subplot(1,2,2);
imshow(mask);
title('Semi-circular Mask');

figure;
% Create a colored overlay
overlay = zeros(rows, cols, 3);
overlay(:,:,1) = double(disk)/255;  % Red channel = original image
overlay(:,:,2) = double(disk)/255;  % Green channel = original image
overlay(:,:,3) = double(disk)/255;  % Blue channel = original image

% Add a colored tint to the masked region
overlay(:,:,1) = overlay(:,:,1) + mask*0.5;  % Increase red channel
overlay(:,:,1) = min(overlay(:,:,1), 1);     % Ensure values don't exceed 1

imshow(overlay);
title('Mask Overlay on Original Image');

% Apply mask to the original and filtered images
masked_orig = double(disk) .* mask;
masked_mean = double(mean_5iter) .* mask;
masked_median = double(median_5iter) .* mask;
masked_alpha = double(alpha_5iter) .* mask;
masked_sigma = double(sigma_5iter) .* mask;
masked_snn = double(snn_5iter) .* mask;

% Count non-zero pixels in mask for proper mean calculation
num_pixels = sum(mask(:));

% Calculate mean and standard deviation for each filtered image
orig_mean = sum(masked_orig(:)) / num_pixels;
orig_std = sqrt(sum((masked_orig(:) - orig_mean).^2 .* mask(:)) / num_pixels);

mean_5iter_mean = sum(masked_mean(:)) / num_pixels;
mean_5iter_std = sqrt(sum((masked_mean(:) - mean_5iter_mean).^2 .* mask(:)) / num_pixels);

median_5iter_mean = sum(masked_median(:)) / num_pixels;
median_5iter_std = sqrt(sum((masked_median(:) - median_5iter_mean).^2 .* mask(:)) / num_pixels);

alpha_5iter_mean = sum(masked_alpha(:)) / num_pixels;
alpha_5iter_std = sqrt(sum((masked_alpha(:) - alpha_5iter_mean).^2 .* mask(:)) / num_pixels);

sigma_5iter_mean = sum(masked_sigma(:)) / num_pixels;
sigma_5iter_std = sqrt(sum((masked_sigma(:) - sigma_5iter_mean).^2 .* mask(:)) / num_pixels);

snn_5iter_mean = sum(masked_snn(:)) / num_pixels;
snn_5iter_std = sqrt(sum((masked_snn(:) - snn_5iter_mean).^2 .* mask(:)) / num_pixels);

% Display statistics
fprintf('Original: Mean = %.2f, Std = %.2f\n', orig_mean, orig_std);
fprintf('Mean (5 iter): Mean = %.2f, Std = %.2f\n', mean_5iter_mean, mean_5iter_std);
fprintf('Median (5 iter): Mean = %.2f, Std = %.2f\n', median_5iter_mean, median_5iter_std);
fprintf('Alpha-Trim (5 iter): Mean = %.2f, Std = %.2f\n', alpha_5iter_mean, alpha_5iter_std);
fprintf('Sigma (5 iter): Mean = %.2f, Std = %.2f\n', sigma_5iter_mean, sigma_5iter_std);
fprintf('SNN (5 iter): Mean = %.2f, Std = %.2f\n', snn_5iter_mean, snn_5iter_std);


function diffused_img = anisotropic_diffusion(img, num_iter, lambda, K, option)
% ANISOTROPIC_DIFFUSION Performs anisotropic diffusion on an image
%   img: Input image
%   num_iter: Number of iterations
%   lambda: Stability factor (0 < lambda < 0.25)
%   K: Conductance parameter
%   option: 1 or 2, selects which function g(.) to use
%      1: g(∇I) = exp(-(|∇I|/K)²)
%      2: g(∇I) = 1 / (1 + (|∇I|/K)²)

% Convert input image to double
img = double(img);
[rows, cols] = size(img);

% Initialize output
diffused_img = img;

% Create a figure to show progress if desired
% figure;
% subplot(1,2,1); imshow(uint8(img)); title('Original Image');
% h = subplot(1,2,2); imshow(uint8(diffused_img)); title('Iteration: 0');

% Perform iterations
for i = 1:num_iter
    % Compute gradients using central differences with symmetric boundary conditions
    % North, South, East, West neighbors
    N = [diffused_img(1,:); diffused_img(1:rows-1,:)];
    S = [diffused_img(2:rows,:); diffused_img(rows,:)];
    E = [diffused_img(:,2:cols), diffused_img(:,cols)];
    W = [diffused_img(:,1), diffused_img(:,1:cols-1)];
    
    % Compute differences
    dN = N - diffused_img;
    dS = S - diffused_img;
    dE = E - diffused_img;
    dW = W - diffused_img;
    
    % Compute conduction coefficients
    if option == 1
        % g(∇I) = exp(-(|∇I|/K)²)
        cN = exp(-(dN/K).^2);
        cS = exp(-(dS/K).^2);
        cE = exp(-(dE/K).^2);
        cW = exp(-(dW/K).^2);
    else % option == 2
        % g(∇I) = 1 / (1 + (|∇I|/K)²)
        cN = 1./(1 + (dN/K).^2);
        cS = 1./(1 + (dS/K).^2);
        cE = 1./(1 + (dE/K).^2);
        cW = 1./(1 + (dW/K).^2);
    end
    
    % Update image using discrete approximation of the diffusion equation
    diffused_img = diffused_img + lambda * (cN.*dN + cS.*dS + cE.*dE + cW.*dW);
    
    % Update figure to show progress if desired
    % set(h, 'Name', ['Iteration: ' num2str(i)]);
    % imshow(uint8(diffused_img), 'Parent', h);
    % title(h, ['Iteration: ' num2str(i)]);
    % drawnow;
end

% Convert back to uint8 for display
diffused_img = uint8(diffused_img);
end

% Main script to run anisotropic diffusion on images

% Parameters
lambda = 0.25;  % Stability factor
K = 15;         % Conductance parameter 
iterations = [0, 5, 20, 100];  % Iterations to display

% Load images
cwheelnoise = imread('cwheelnoise.gif');
cameraman = imread('cameraman.tif');

% Initialize figures for cwheelnoise
figure('Name', 'Anisotropic Diffusion - cwheelnoise - g1');
figure('Name', 'Anisotropic Diffusion - cwheelnoise - g2');
figure('Name', 'Histograms - cwheelnoise - g1');
figure('Name', 'Histograms - cwheelnoise - g2');
figure('Name', 'Line Plots - cwheelnoise - g1');
figure('Name', 'Line Plots - cwheelnoise - g2');
figure('Name', 'Segmentations - cwheelnoise - g1');
figure('Name', 'Segmentations - cwheelnoise - g2');

% Initialize figures for cameraman
figure('Name', 'Anisotropic Diffusion - cameraman - g1');
figure('Name', 'Anisotropic Diffusion - cameraman - g2');

% Process cwheelnoise with both g functions
for option = 1:2
    g_name = ['g' num2str(option)];
    
    % Process for different iterations
    for idx = 1:length(iterations)
        iter = iterations(idx);
        
        % For iteration 0, use original image
        if iter == 0
            diffused = cwheelnoise;
        else
            diffused = anisotropic_diffusion(cwheelnoise, iter, lambda, K, option);
        end
        
        % 1. Display the image
        figure(option);
        subplot(2,2,idx);
        imshow(diffused);
        title(['Iter: ' num2str(iter)]);
        
        % 2. Display histogram
        figure(option+2);
        subplot(2,2,idx);
        imhist(diffused);
        title(['Histogram, Iter: ' num2str(iter)]);
        
        % 3. Plot the line y = 128
        figure(option+4);
        subplot(2,2,idx);
        plot(diffused(128,:));
        title(['Line y=128, Iter: ' num2str(iter)]);
        ylim([0 255]);
        
        % 4. Segmented version

        threshold = 120; 
        segmented = diffused > threshold;
        
        figure(option+6);
        subplot(2,2,idx);
        imshow(segmented);
        title(['Segmented, Iter: ' num2str(iter)]);
    end
end

% Process cameraman with both g functions
for option = 1:2
    g_name = ['g' num2str(option)];
    
    % Process for different iterations
    for idx = 1:length(iterations)
        iter = iterations(idx);
        
        % For iteration 0, use original image
        if iter == 0
            diffused = cameraman;
        else
            diffused = anisotropic_diffusion(cameraman, iter, lambda, K, option);
        end
        
        % Display the image
        figure(option+8);
        subplot(2,2,idx);
        imshow(diffused);
        title(['Iter: ' num2str(iter)]);
    end
end
