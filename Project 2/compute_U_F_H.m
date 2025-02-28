clc; clear; close all;
% Load the binary image
img = imbinarize(imread('match1.gif'));
% Label connected components
[labeled_img, num_objects] = bwlabel(img);

% Display seperated objs
figure;
for i = 1:num_objects    
    object = (labeled_img == i);
    subplot(2, 2, i), imshow(object), title('Object ' + string(i));
end

% Define the maximum radius for the disk
max_radius = 20; % Adjust based on the size of objects

% Initialize variables
U = zeros(max_radius, num_objects); % Size distribution
f = zeros(max_radius, num_objects); % Pectrum
H = zeros(1, num_objects); % Complexity

% Compute size distribution, pectrum, and complexity for each object
for i = 1:num_objects
    % Extract the i-th object
    object = (labeled_img == i);
    
    % Compute size distribution U(r)
    for r = 1:max_radius
        % Create a disk structuring element of radius r
        se = strel('square', 3+(2*(r-1))); %r=1 3x3 r=2 5x5 r=7x7
        
        % opening the object with the se
        opened_object = imopen(object, se);
        
        % Compute the area of the openned object
        U(r, i) = sum(opened_object(:));
    end
    
    % Compute pectrum f(r)
    f(1:max_radius-1, i) = (U(1:max_radius-1, i) - U(2:max_radius, i))/sum(object(:));
    
    % Normalize the pectrum to get probabilities
    p = f(:, i) / sum(f(:, i));
    p(p == 0) = []; % Remove zero values to avoid log(0)
    
    % Compute complexity H(X|B) as entropy
    H(i) = -sum(p .* log2(p));
end

% Display results
disp('Size Distribution U(r):');
disp(U);

disp('Pectrum f(r):');
disp(f);

disp('Complexity H(X|B):');
disp(H);

% Plot size distribution and pectrum for each object
figure;
for i = 1:num_objects
    subplot(num_objects, 2, 2*i-1);
    plot(1:max_radius, U(:, i), 'b-o');
    xlabel('Radius (r)');
    ylabel('U(r)');
    title(sprintf('Size Distribution - Object %d', i));
    
    subplot(num_objects, 2, 2*i);
    plot(1:max_radius-1, f(1:max_radius-1, i), 'r-o');
    xlabel('Radius (r)');
    ylabel('f(r)');
    title(sprintf('Pectrum - Object %d', i));
end

% Determine the most complex object
[~, most_complex] = max(H);
fprintf('The most complex object is Object %d with complexity H(X|B) = %.4f\n', most_complex, H(most_complex));