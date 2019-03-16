function S = myclassifier(im, model)
 % Support vector machince classifier of 3 digit captcha
 % images containing digits 0,1,2 using HOG features. The SVM was trained
 % using default settings.


% Parameters that are specific for corresponding paremeters used in 
% training.
standard_image_size = [70 70];
HOG_Cell_size = [12 12];

% Preprocess image to remove everything but the digits
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
HOGFeatures_1 = extractHOGFeatures(digit_1_padded,...
    'CellSize', HOG_Cell_size);
HOGFeatures_2 = extractHOGFeatures(digit_2_padded,...
    'CellSize', HOG_Cell_size);
HOGFeatures_3 = extractHOGFeatures(digit_3_padded,...
    'CellSize', HOG_Cell_size);

% Use the trained support vector machine to predict the digits.
S = predict(model, [HOGFeatures_1; HOGFeatures_2; HOGFeatures_3]);

end

