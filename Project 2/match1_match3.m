clc; clear; close all;

% Load images
match1 = imbinarize(imread('match1.gif'));
match3 = imbinarize(imread('match3.gif'));

% Label connected components
[labeled_match1, num_objects_match1] = bwlabel(match1);
[labeled_match3, num_objects_match3] = bwlabel(match3);
% Display seperated objs
figure;
for i = 1:num_objects_match1
    object = (labeled_match1 == i);
    subplot(2, 4, i), imshow(object), title('match1 obj ' + string(i));
end
for i = 1:num_objects_match3
    object = (labeled_match3 == i);
    subplot(2, 4, i+4), imshow(object), title('match3 obj ' + string(i));
end

% Define the maximum radius for the se
max_radius = 20;

% Initialize variables to store pectra
f_match1 = zeros(max_radius, num_objects_match1); % Pectra for match1
f_match3 = zeros(max_radius, num_objects_match3); % Pectra for match3

% Compute pectra for objects in match1
for i = 1:num_objects_match1
    % Extract the i-th object
    object = (labeled_match1 == i);

    %init the size distribution U(r)
    U = zeros(max_radius, 1);
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
    %f_match1(1:max_radius-1, i) = U(1:max_radius-1, i) - U(2:max_radius, i);
    f_match1(1:max_radius-1, i) = (U(1:max_radius-1, i) - U(2:max_radius, i))/sum(object(:));
end

% Compute pectra for objects in match3
for i = 1:num_objects_match3
    % Extract the i-th object
    object = (labeled_match3 == i);

    %init the size distribution U(r)
    U = zeros(max_radius, 1);
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
    %f_match3(1:max_radius-1, i) = U(1:max_radius-1, i) - U(2:max_radius, i);
    f_match3(1:max_radius-1, i) = (U(1:max_radius-1, i) - U(2:max_radius, i))/sum(object(:));
end

disp('Pectrum f_match1(r):');
disp(f_match1);
disp('Pectrum f_match3(r):');
disp(f_match3);


% Compare pectra using Euclidean distance
distance_matrix = zeros(num_objects_match1, num_objects_match3);
for i = 1:num_objects_match1
    for j = 1:num_objects_match3
        % Compute Euclidean distance between pectra
        euclidean_distance = norm(f_match1(:, i) - f_match3(:, j));

        % Manhattan Distance
        manhattan_distance = sum(abs(f_match1(:, i) - f_match3(:, j)));

        % Cosine Similarity
        cosine_similarity = dot(f_match1(:, i), f_match3(:, j)) / (norm(f_match1(:, i)) * norm(f_match3(:, j)));
        cosine_distance = 1 - cosine_similarity;
        
        % Correlation Distance
        correlation = corrcoef(f_match1(:, i), f_match3(:, j));
        correlation_distance = 1 - correlation(1, 2);
        
        % Hellinger Distance
        hellinger_distance = sqrt(0.5 * sum((sqrt(f_match1(:, i)) - sqrt(f_match3(:, j))).^2));
        
        %select a Distance Metric to test results
        distance_matrix(i, j) = euclidean_distance;       
    end
end

% Display the distance matrix
disp('Distance Matrix:');
disp(distance_matrix);

% Find the best matches
best_matches = zeros(num_objects_match1, 1);
for i = 1:num_objects_match1
    [~, best_matches(i)] = min(distance_matrix(i, :));
    fprintf('Object %d in match1 best matches Object %d in match3 with distance %.4f\n', i, best_matches(i), distance_matrix(i, best_matches(i)));
end