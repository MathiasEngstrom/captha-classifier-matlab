tic
true_labels = importdata('labels.txt');
N = size(true_labels,1);
% The size of the matrix for storing HOG feature vectors is based on a
% standard image size set by standard_image_size and HOG feature cell size
% set by HOG_Cell_size.
standard_image_size = [70 70];
HOG_Cell_size = [12 12];
HOGFeatures_vector_size = length(extractHOGFeatures(...
    zeros(standard_image_size), 'CellSize', HOG_Cell_size));
HOGFeatures = zeros(N*3, HOGFeatures_vector_size); 

counter = 1;
for k = 1:N
    % Read images
    im = imread(sprintf('imagedata/train_%04d.png', k));

    % Preprocess images to remove all but digits
    bw_im = im2bw(im);
    filt_bw_im = medfilt2(bw_im, [7 7], 'symmetric');
    filt_im_complement = imcomplement(filt_bw_im);
    

    % Reduce image to minimal size
    [row, col] = find(filt_im_complement);
    im_reduced = filt_im_complement(min(row):max(row),min(col):max(col));
    
    % Divide image into 3 equally sized images, (hopefully)
    % one for each digit. Digit width needs to be floored to prevent
    % non-integer indicies and indicies out of array bounds.
    digit_width = floor(size(im_reduced, 2)/3);
    
    digit_1 = im_reduced(:, 1:digit_width);
    digit_2 = im_reduced(:, digit_width+1 : 2*digit_width);
    digit_3 = im_reduced(:, 2*digit_width+1 : 3*digit_width);
    
    % The images need to be of the same size in order to 
    % obatain comparable vectors when classifying them.
    padding_rows = standard_image_size(1) - size(digit_1, 1);
    padding_cols = standard_image_size(2) - size(digit_1, 2);
    digit_1_padded = padarray(digit_1,[padding_rows padding_cols], 'post');
    digit_2_padded = padarray(digit_2,[padding_rows padding_cols], 'post');
    digit_3_padded = padarray(digit_3,[padding_rows padding_cols], 'post');
    
    % Extract HOG features using an appropriate cell size to achieve
    % sufficient separation at a reasonable time.
    HOGFeatures(counter ,:) = extractHOGFeatures(digit_1_padded,...
        'CellSize', HOG_Cell_size);
    counter = counter +1;
    HOGFeatures(counter ,:) = extractHOGFeatures(digit_2_padded,...
        'CellSize', HOG_Cell_size);
    counter = counter +1;
    HOGFeatures(counter,:) = extractHOGFeatures(digit_3_padded,...
        'CellSize', HOG_Cell_size);
    counter = counter +1;
end

% Reshape labels to allow training on single digits of the training data.
true_labels_vector = reshape(true_labels', size(true_labels, 2)*size(true_labels, 1), 1);
Mdl = fitcecoc(HOGFeatures, true_labels_vector);
saveCompactModel(Mdl, 'Captcha_classifier');
toc
