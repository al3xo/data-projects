% Clean workspace
clear all; close all; clc

%% load 
load cam1_4.mat;
[height1, width1, rgb1, num_frames1] = size(vidFrames1_4);

load cam2_4.mat;
[height2, width2, rgb2, num_frames2] = size(vidFrames2_4);

load cam3_4.mat;
[height3, width3, rgb3, num_frames3] = size(vidFrames3_4);

%% plotting - image 1
for i = 1:num_frames1 
    X = rgb2gray(vidFrames1_4(:,:,:,i));
    
    M = max(X,[],'all');
    [My, Mx] = ind2sub(size(X),find(abs(X) == M));

    x_M = Mx;
    y_M = My;

    x1(i) = median(x_M);
    y1(i) = median(y_M);
end
[~, n] = size(y1);
t1 = 1:n;

figure(1)
subplot(1,3,1)
plot(t1, y1, 'r.')

% plotting - image 2

for i = 1:num_frames2 
    X = rgb2gray(vidFrames2_4(:,:,:,i));

    M = max(X,[],'all');
    [My, Mx] = ind2sub(size(X),find(abs(X) == M));

    x_M = Mx;
    y_M = My;

    x2(i) = mean(x_M);
    y2(i) = mean(y_M);
end
t2 = 1:num_frames2;

subplot(1,3,2)
plot(t2, y2, 'r.')

% plotting - image 3

for i = 1:num_frames3
    X = rgb2gray(vidFrames3_4(:,:,:,i));
    
    a = 0;
    M = max(X,[],'all');
    [Mx, My] = ind2sub(size(X),find(abs(X) == M));

    x_M = Mx;
    y_M = My;

    x3(i) = mean(x_M);
    y3(i) = mean(y_M);
end
t3 = 1:num_frames3;

subplot(1,3,3)
plot(t3, x3, 'k.')

% adjustments
y3 = y3(1:350);

y1 = y1(1:350);

y2 = y2 - 9;
y2 = y2(9:358);

t = 1:350;

for i = t
    y_avg(i) = y1(i) + y2(i) + y3(i);
    y_avg(i) = y_avg(i)/3;
end


x1 = x1(1:350);
x2 = x2(9:358);
x3 = x3(1:350);

for i = t
    x_avg(i) = x1(i) + x2(i) + x3(i);
    x_avg(i) = x_avg(i)/3;
end


% plotting
figure(2)
subplot(1, 2, 1)
plot(t, y1, 'b.', t, y2, 'r.', t, y3, 'k.', t, y_avg, 'c');
xlim([0, 150])

subplot(1, 2, 2)
plot(t, x1, 'b.', t, x2, 'r.', t, x3, 'k.', t, x_avg, 'c')


%% SVD
M = [x_avg; y_avg];
[U, S, V] = svd(M);

rank = 1;
M_rank = U(:,1:rank)*S(1:rank,1:rank)*V(:,1:rank).';

plot(M(1,:), M(2,:), 'k.', M_rank(1,:), M_rank(2,:), 'r.');

