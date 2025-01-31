clear;

% -- define functions dilate and erose --
function im_dilated = dilate_image(input_image,kernal)
    %get im size
    [Row, Col] = size(input_image);
    % create a background im    
    im_dilated=ones(Row, Col);
    %get kernal size
    kernal=~kernal.Neighborhood; %reversed the filter because value 1 is background
    [ker_Row,ker_Col] = size(kernal);
    
    %dilation process
    for i_im  = 1 : Row %loop through rows
        for j_im  = 1 : Col %loop through columns        
            isHit=false;
            % check hit
            for i_kn = 1 : size(kernal,1)
                for j_kn = 1 : size(kernal,2)
                    check_i = i_im +i_kn-floor(ker_Row/2)-1;
                    check_j = j_im +j_kn-floor(ker_Col/2)-1;
                    %Check if the filter is outside the image
                    if check_i<1 || check_i>Row || check_j < 1 || check_j > Col
                        continue;
                    end
                    if kernal(i_kn,j_kn) == 0 && input_image(check_i,check_j) == 0
                        isHit = true;
                        break;
                    end
                end
            end
     
            if isHit
                im_dilated(i_im , j_im ) = 0 ;
            end
        end
    end
end

function im_erosed = erose_image(input_image,kernal)
    %get im size
    [Row, Col] = size(input_image);
    % create a background im    
    im_erosed=ones(Row, Col);
    %get kernal size
    kernal=~kernal.Neighborhood;%reversed the filter because value 1 is background
    [ker_Row,ker_Col] = size(kernal);
    
    %dilation process
    for i_im  = 1 : Row %loop through rows
        for j_im  = 1 : Col %loop through columns        
            isOverlapse=true;
            % check isOverlapse
            for i_kn = 1 : size(kernal,1)
                for j_kn = 1 : size(kernal,2)
                    check_i = i_im +i_kn-floor(ker_Row/2)-1;
                    check_j = j_im +j_kn-floor(ker_Col/2)-1;
                    %Check if the filter is outside the image
                    if check_i<1 || check_i>Row || check_j < 1 || check_j > Col
                        continue;
                    end
                    if (kernal(i_kn,j_kn) ~= input_image(check_i,check_j)) && ~kernal(i_kn,j_kn)
                        isOverlapse = false;
                        break;
                    end
                end
            end
     
            if ~isOverlapse
                im_erosed(i_im, j_im) = 1 ; 
            else
                im_erosed(i_im, j_im) = 0 ; 
            end
        end
    end
end
% --end define functions--

% Load and process the image
img = imread('RandomDisks.jpg'); % Load image
gray_img = rgb2gray(img);            % Convert to grayscale
binary_img = imbinarize(gray_img, 0.5); % Convert to binary (0.5 is threshole)

% Apply noise filtering
se_noise = strel('disk', 1); % Structuring element for noise removal
filtered_img = imclose(imopen(binary_img, se_noise), se_noise); % Noise reduction

%smallest disk is 18 biggest is 64
% Define kernal for detect biggest and smallest disk. Other values ​​could be considered.
ker1 = strel('disk', 28); 
ker2 = strel('disk', 10); 


% Perform the Hit-or-Miss Transform
biggest = dilate_image(erose_image(filtered_img, ker1),strel('disk', 35));
smallest = dilate_image(erose_image(filtered_img, ker2),strel('disk', 13));
hit_miss = ~(biggest & ~smallest);

% Subtract from original to extract smallest and largest disks
final_result = ~(~filtered_img & hit_miss);

%imtool(filtered_img);
%imtool(biggest);
%imtool(smallest);
%imtool(hit_miss);
%imtool(~final_result);

% Display results
%figure;
subplot(2,2,1); imshow(binary_img); title('Binary Image');
subplot(2,2,2); imshow(filtered_img); title('Filtered Image');
subplot(2,2,3); imshow(hit_miss); title('Middle-Sized Disks');
subplot(2,2,4); imshow(final_result); title('Smallest & Largest Disks');

% Save results
%imwrite(final_result, 'Detected_Disks.png');