% Competition entry of Mathias Engström, Erik Olby, Hampus Söderberg of team
% Hooligan Outlaw Gangster Supreme Victory Machine (HOG SVM).
% Main script (run this!)

tic
true_labels = importdata('labels.txt');
my_labels = zeros(size(true_labels));
N = size(true_labels,1);
model = loadCompactModel('Captcha_classifier');
for k = 1:N
    im = imread(sprintf('imagedata/train_%04d.png', k));
    my_labels(k,:) = myclassifier(im, model);
end

fprintf('\n\nAverage precision: \n');
fprintf('%f\n\n',mean(sum(abs(true_labels - my_labels),2)==0));
toc