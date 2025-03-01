function matchShadows()
    % Step 1: Load and prepare images
    % shadow1 = imread('../images/shadow1.gif');
    % shadow1rotated = imread('../images/shadow1rotated.gif');
    shadow1 = imread('../images/match1.gif');
    shadow1rotated = imread('../images/match3.gif');
    shadow1 = convert2binary(shadow1);
    shadow1rotated = convert2binary(shadow1rotated);

    % Display the loaded images to verify
    figure;
    subplot(1,2,1); imshow(shadow1); title('Shadow1 (Original)');
    subplot(1,2,2); imshow(shadow1rotated); title('Shadow1rotated');
    
    % Step 2: Filter out small objects and keep only solid
    fprintf('Keeping only the solid objects...\n');
    [shadow1_filtered, labeled1] = keepLargestObjects(shadow1, 4);
    [shadow1rotated_filtered, labeled2] = keepLargestObjects(shadow1rotated, 4);
    
    % Display the filtered objects for verification
    figure;
    subplot(1,2,1); imshow(shadow1_filtered); title('Shadow1 - Main Objects');
    subplot(1,2,2); imshow(shadow1rotated_filtered); title('Shadow1rotated - Main Objects');
    
    % Display the labeled regions with colors
    figure;
    subplot(1,2,1);
    imshow(label2rgb(labeled1, 'jet', 'k', 'shuffle'));
    title('Labeled Main Objects - Shadow1');
    subplot(1,2,2);
    imshow(label2rgb(labeled2, 'jet', 'k', 'shuffle'));
    title('Labeled Main Objects - Shadow1rotated');
    
    % Step 3: Perform pecstral analysis on each object
    % Define structuring element as a 3x3 square
    B = ones(3, 3);
    
    % Compute pattern spectrums for each object in the first image
    figure('Position', [100, 100, 900, 700]);
    patterSpectrums1 = cell(4, 1);
    for i = 1:4
        % Extract this object only
        objMask = (labeled1 == i);
        
        % Get bounding box to crop the object
        props = regionprops(objMask, 'BoundingBox');
        bbox = props.BoundingBox;
        x = max(1, floor(bbox(1)));
        y = max(1, floor(bbox(2)));
        w = ceil(bbox(3));
        h = ceil(bbox(4));
        croppedObj = objMask(y:y+h-1, x:x+w-1);
        
        % Compute pattern spectrum using openings
        patternSpectrum = computePatternSpectrum(croppedObj, B);
        patterSpectrums1{i} = patternSpectrum;
        
        % Display the pattern spectrum
        subplot(2, 4, i);
        stem(patternSpectrum, 'LineWidth', 1.5);
        title(['Pattern Spectrum for Object ' num2str(i) ' in Shadow1']);
        xlabel('Structuring Element Size');
        ylabel('Pattern Spectrum f(n)');
        grid on;
    end
    
    % Compute pattern spectrums for each object in the second image
    patterSpectrums2 = cell(4, 1);
    for i = 1:4
        % Extract this object only
        objMask = (labeled2 == i);
        
        % Get bounding box to crop the object
        props = regionprops(objMask, 'BoundingBox');
        bbox = props.BoundingBox;
        x = max(1, floor(bbox(1)));
        y = max(1, floor(bbox(2)));
        w = ceil(bbox(3));
        h = ceil(bbox(4));
        croppedObj = objMask(y:y+h-1, x:x+w-1);
        
        % Compute pattern spectrum using morphological openings
        patternSpectrum = computePatternSpectrum(croppedObj, B);
        patterSpectrums2{i} = patternSpectrum;
        
        % Display the pattern spectrum
        subplot(2, 4, i+4);
        stem(patternSpectrum, 'LineWidth', 1.5);
        title(['Pattern Spectrum for Object ' num2str(i) ' in Shadow1rotated']);
        xlabel('Structuring Element Size');
        ylabel('Pattern Spectrum f(n)');
        grid on;
    end
    
    % Step 4: Normalize the pattern spectrums to get pecstrums
    pecstrums1 = cell(4, 1);
    for i = 1:4
        pecstrums1{i} = normalizePecstrum(patterSpectrums1{i});
    end
    
    pecstrums2 = cell(4, 1);
    for i = 1:4
        pecstrums2{i} = normalizePecstrum(patterSpectrums2{i});
    end
    
    % Step 5: Calculate distance matrix between pecstrums
    distances = zeros(4, 4);
    for i = 1:4
        for j = 1:4
            distances(i,j) = computePecstralDistance(pecstrums1{i}, pecstrums2{j});
        end
    end
    
    % Display the distance matrix
    fprintf('Pecstral Distance Matrix:\n');
    disp(distances);
    
    % Step 6: Find best matches based on pecstral distances[~, matches] = min(distances, [], 2);
    % [~, matches] = min(distances, [], 2);

    costOfNonAssignment = 1;
    pairs = matchpairs(distances, costOfNonAssignment);
    matches = zeros(4,1);
    for row = 1:4
        i = pairs(row,1);
        j = pairs(row,2);
        matches(i) = j;
    end

    
    % Step 7: Display matching results
    fprintf('Matching Results Based on Pecstral Analysis:\n');
    for i = 1:4
        fprintf('Shadow object %d matches rotated object %d (distance = %.4f)\n', ...
                i, matches(i), distances(i, matches(i)));
    end
    
    % Step 8: Visualize with colored objects
    visualizeMatches(shadow1_filtered, shadow1rotated_filtered, labeled1, labeled2, matches);
    
    % Step 9: Calculate complexity for each object
    complexities1 = zeros(4, 1);
    for i = 1:4
        complexities1(i) = computeComplexity(pecstrums1{i});
    end
    
    complexities2 = zeros(4, 1);
    for i = 1:4
        complexities2(i) = computeComplexity(pecstrums2{i});
    end
    
    % Display complexities
    fprintf('\nComplexity H(X|B) for objects in Shadow1:\n');
    for i = 1:4
        fprintf('Object %d: %.4f\n', i, complexities1(i));
    end
    
    fprintf('\nComplexity H(X|B) for objects in Shadow1rotated:\n');
    for i = 1:4
        fprintf('Object %d: %.4f\n', i, complexities2(i));
    end
    
    % Find the most complex object
    [maxComplexity, maxIdx] = max(complexities1);
    fprintf('\nThe most complex object in Shadow1 is object %d with H(X|B) = %.4f\n', ...
            maxIdx, maxComplexity);
end

function patternSpectrum = computePatternSpectrum(X, B)
    % Compute the pattern spectrum using morphological openings
    % patternSpectrum(n) = pixelCount(open_n) - pixelCount(open_n+1)
    
    % Maximum size for structuring element iterations
    maxSize = 20;
    
    % Preallocate array for opened areas
    openedAreas = zeros(maxSize, 1);
    
    % Compute first opening (size 1 - original structuring element)
    opened = imopen(X, B);
    openedAreas(1) = sum(opened(:));
    
    % Compute openings with larger structuring elements
    for n = 2:maxSize
        % Create structuring element of size n
        B_n = strel('square', 2*n+1);
        
        % Perform opening
        opened = imopen(X, B_n);
        
        % Compute area after opening
        openedAreas(n) = sum(opened(:));
        
        % If the opened image is empty, stop
        if openedAreas(n) == 0
            break;
        end
    end
    
    % Trim zeros at the end
    lastNonZero = find(openedAreas > 0, 1, 'last');
    if isempty(lastNonZero)
        patternSpectrum = zeros(1, 1);
        return;
    end
    
    openedAreas = openedAreas(1:lastNonZero);
    
    % Compute pattern spectrum as the derivative of the opened areas
    patternSpectrum = zeros(length(openedAreas)-1, 1);
    for i = 1:length(openedAreas)-1
        patternSpectrum(i) = openedAreas(i) - openedAreas(i+1);
    end
end

function pecstrum = normalizePecstrum(patternSpectrum)
    % Normalize pattern spectrum to create pecstrum
    % The sum of the pecstrum should equal 1
    
    totalSum = sum(patternSpectrum);
    
    if totalSum > 0
        pecstrum = patternSpectrum / totalSum;
    else
        % If pattern spectrum is all zeros, return a uniform distribution
        pecstrum = ones(size(patternSpectrum)) / length(patternSpectrum);
    end
end

function distance = computePecstralDistance(pecstrum1, pecstrum2)
    % Compute distance between two pecstrums
    % Ensure both pecstrums have the same length by padding with zeros
    
    len1 = length(pecstrum1);
    len2 = length(pecstrum2);
    
    if len1 > len2
        % Pad pecstrum2 with zeros
        pecstrum2 = [pecstrum2; zeros(len1-len2, 1)];
    elseif len2 > len1
        % Pad pecstrum1 with zeros
        pecstrum1 = [pecstrum1; zeros(len2-len1, 1)];
    end
    
    % Compute Euclidean distance
    distance = sqrt(sum((pecstrum1 - pecstrum2).^2));
end

function complexity = computeComplexity(pecstrum)
    % Compute complexity as the entropy of the pecstrum
    % H(X|B) = -sum(p(n) * log2(p(n)))
    
    % Avoid log(0) by only considering non-zero probabilities
    nonZeroIndices = pecstrum > 0;
    
    if any(nonZeroIndices)
        complexity = -sum(pecstrum(nonZeroIndices) .* log2(pecstrum(nonZeroIndices)));
    else
        complexity = 0;
    end
end

function [filteredImage, labeled] = keepLargestObjects(img, numToKeep)
    % Keep only the largest numToKeep objects in the image
    
    % Label all objects in the original image
    [labeled_all, numObjects] = bwlabel(img, 8);
    
    % If we found fewer objects than requested, keep all of them
    if numObjects <= numToKeep
        filteredImage = img;
        labeled = labeled_all;
        return;
    end
    
    % Get the area of each object
    stats = regionprops(labeled_all, 'Area');
    areas = [stats.Area];
    
    % Sort areas in descending order and get indices of largest objects
    [~, idx] = sort(areas, 'descend');
    
    % Keep only the largest numToKeep objects
    toKeep = idx(1:numToKeep);
    
    % Create a new image with only the largest objects
    filteredImage = false(size(img));
    for i = 1:numToKeep
        filteredImage = filteredImage | (labeled_all == toKeep(i));
    end
    
    % Relabel the objects consecutively from 1 to numToKeep
    labeled = zeros(size(img));
    for i = 1:numToKeep
        labeled(labeled_all == toKeep(i)) = i;
    end
end

function bw = convert2binary(img)
    % Convert input image to binary
    if ~islogical(img)
        if size(img, 3) > 1
            img = rgb2gray(img);
        end
        
        % Try multiple thresholding approaches if needed
        if max(img(:)) > 1
            % For 0-255 range
            bw = img > 128;
        else
            % For 0-1 range
            bw = img > 0.5;
        end
    else
        bw = img;
    end
end

function visualizeMatches(img1, img2, labeled1, labeled2, matches)
    % Create a figure for displaying the results
    figure('Position', [100, 100, 800, 400]);
    
    % Define distinct colors for objects
    colors = [
        1 0 0;
        0 1 0; 
        0 0 1; 
        1 1 0;
    ];
    
    % Create colored versions of both images
    colored1 = zeros([size(img1), 3]);
    colored2 = zeros([size(img2), 3]);
    
    % Color each object in the first image
    for i = 1:4
        mask = (labeled1 == i);
        for c = 1:3
            channel = colored1(:,:,c);
            channel(mask) = colors(i,c);
            colored1(:,:,c) = channel;
        end
    end
    
    % Color each object in the second image according to its match
    for i = 1:4
        mask = (labeled2 == i);
        % Find which object in image 1 matches this object
        match_idx = find(matches == i);
        
        if ~isempty(match_idx)
            for c = 1:3
                channel = colored2(:,:,c);
                channel(mask) = colors(match_idx,c);
                colored2(:,:,c) = channel;
            end
        end
    end
    
    % Display the images side by side
    subplot(1,2,1);
    imshow(colored1);
    title('Original Shadow Objects');
    
    % Add labels to each object in the first image
    props1 = regionprops(labeled1, 'Centroid');
    for i = 1:length(props1)
        text(props1(i).Centroid(1), props1(i).Centroid(2), num2str(i), ...
             'Color', 'w', 'FontSize', 14, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'center');
    end
    
    subplot(1,2,2);
    imshow(colored2);
    title('Matched Rotated Objects (Using Pecstral Analysis)');
    
    % Add labels to each object in the second image
    props2 = regionprops(labeled2, 'Centroid');
    for i = 1:length(props2)
        % Find which object in image 1 matches this object
        match_idx = find(matches == i);
        
        if ~isempty(match_idx)
            text(props2(i).Centroid(1), props2(i).Centroid(2), num2str(match_idx), ...
                 'Color', 'w', 'FontSize', 14, 'FontWeight', 'bold', ...
                 'HorizontalAlignment', 'center');
        end
    end
end