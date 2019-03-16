tic
true_labels = importdata('labels.txt');
my_labels = zeros(400, 3);
N = size(true_labels,1);
model = loadCompactModel('Captcha_classifier');
for k = 801:N
    im = imread(sprintf('imagedata/train_%04d.png', k));
    my_labels(k-800,:) = myclassifier(im, model);
end

fprintf('\n\nAverage precision: \n');
fprintf('%f\n\n',mean(sum(abs(true_labels(801:N,:) - my_labels),2)==0));
toc