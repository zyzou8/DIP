clc; clear; close all;

function im_erosed = erose_image(input_image,kernal)
    %get im size
    [Row, Col] = size(input_image);
    % create a background im    
    im_erosed=ones(Row, Col);
    %get kernal size
    [ker_Row,ker_Col] = size(kernal);
    
    %erosion process
    for i_im  = 1 : Row %loop through rows
        for j_im  = 1 : Col %loop through columns        
            isOverlapse=true;
            % check hit
            for i_kn = 1 : size(kernal,1)
                for j_kn = 1 : size(kernal,2)
                    check_i = i_im +i_kn-floor(ker_Row/2)-1;
                    check_j = j_im +j_kn-floor(ker_Col/2)-1;
                    if check_i<1 || check_i>Row || check_j < 1 || check_j > Col
                        continue;
                    end
                    if (kernal(i_kn,j_kn) ~= input_image(check_i,check_j)) && kernal(i_kn,j_kn)
                        isOverlapse = false;
                        break;
                    end
                end
            end
     
            if isOverlapse
                im_erosed(i_im, j_im) = 1 ; 
            else
                im_erosed(i_im, j_im) = 0 ; 
            end
        end
    end
end

% Load binary image
X = imread('penn256.gif');

% X = imread('bear.gif');
% X = imread('match1.gif');
% X = imread('rectangle.gif');
X = imbinarize(X); % convert to binary image

% Define 8 foreground structuring elements
Bf = {
    [0 0 0; 0 1 0; 1 1 1],  % B1f
    [0 0 0; 1 1 0; 1 1 0],  % B2f
    [1 0 0; 1 1 0; 1 0 0],  % B3f
    [1 1 0; 1 1 0; 0 0 0],  % B4f
    [1 1 1; 0 1 0; 0 0 0],  % B5f
    [0 1 1; 0 1 1; 0 0 0],  % B6f
    [0 0 1; 0 1 1; 0 0 1],  % B7f
    [0 0 0; 0 1 1; 0 1 1]   % B8f
};

% Define 8 background structuring elements
Bb = {
    [1 1 1; 0 0 0; 0 0 0],  % B1b
    [0 1 1; 0 0 1; 0 0 0],  % B2b
    [0 0 1; 0 0 1; 0 0 1],  % B3b
    [0 0 0; 0 1 0; 0 1 1],  % B4b
    [0 0 0; 0 0 0; 1 1 1],  % B5b
    [0 0 0; 1 0 0; 1 1 0],  % B6b
    [1 0 0; 1 0 0; 1 0 0],  % B7b
    [1 1 0; 1 0 0; 0 0 0]   % B8b
};

% Homotopic Skeletonization Algorithm
i = 0;
Xi = X;
X_prev = zeros(size(X)); %X_prev means X(i-1)
ero_tool = erosion;

while ~isequal(Xi, X_prev) %if X doesnt change anymore stop looping
    X_prev = Xi; %keep the previous X before the thinning
    i = i + 1;
    for j = 1:8
        %Xi = Xi - bwhitmiss(Xi, Bf{j}, Bb{j}); % Hit-or-miss thinning
        %Xi = Xi - ero_tool.bwhitmiss(Xi, Bf{j}, Bb{j}); % Hit-or-miss thinning
        Xi = Xi - bitand(erose_image(Xi, Bf{j}), erose_image((1-Xi),Bb{j}));
    end
    
    % Save results at X2, X5, X10
    if i == 2
        X2 = Xi;
    elseif i == 5
        X5 = Xi;
    elseif i == 10
        X10 = Xi;
    end
end

% Display results
figure;
subplot(2,3,1), imshow(X), title('Original');
subplot(2,3,2), imshow(X2), title('X_2');
subplot(2,3,3), imshow(X5), title('X_5');
subplot(2,3,4), imshow(X10), title('X_{10}');
subplot(2,3,5), imshow(Xi), title('Final Skeleton');

% Save results
%imwrite(X2, 'X2.png');
%imwrite(X5, 'X5.png');
%imwrite(X10, 'X10.png');
%imwrite(Xi, 'Skeletonized.png');
