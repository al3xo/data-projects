%% Reset Workspace
clc; clear; close all;

isSkiDrop = 0; % 1 for true, 0 for false

%% import and read generally

ski = VideoReader('ski_drop_low.mp4');
monte = VideoReader('monte_carlo_low.mp4');
 
video = ski;
skiX = zeros(video.Height*video.Width, video.NumFrames);
for i = 1:video.NumFrames
    skiX(:, i) = reshape(rgb2gray(read(video, i)), [video.Height*video.Width, 1]);
end

video = monte;
monteX = zeros(video.Height*video.Width, video.NumFrames);
for i = 1:video.NumFrames
    monteX(:, i) = reshape(rgb2gray(read(video, i)), [video.Height*video.Width, 1]);
end


%% Generate DMD Parts
video = monte;
X = monteX; 

dt = 1/video.FrameRate;
t = linspace(0, video.Duration, video.NumFrames);

X1 = X(:,1:end-1);
X2 = X(:,2:end);

[U, Sigma, V] = svd(X1,'econ');
S = U'*X2*V*diag(1./diag(Sigma));

[eV, D] = eig(S); % compute eigenvalues + eigenvectors
mu = diag(D); % extract eigenvalues
omega = log(mu)/dt;
Phi = U*eV;
b = Phi\X1(:, 1);

%% Low-Rank and Sparse Video Decomposition
low_ranks = (find(abs(omega) < 0.05))';

X_low = zeros(video.Height*video.Width, video.NumFrames);
for i = low_ranks
   X_low = X_low + b(i)*Phi(:, i)*exp(omega(i)*t(i)); 
end

X_sparse = X - abs(X_low);
%% Low Rank to Video
vidX = abs(X_low);

lowVid = VideoWriter('monteLow.avi');
lowVid.FrameRate = video.FrameRate;

open(lowVid)
for i = 1:video.numFrames
    img = mat2gray(reshape(vidX(:, i), [video.Height, video.Width]));
    writeVideo(lowVid, img);
end
close(lowVid)

%% Sparse to Video
vidX = X_sparse;

lowVid = VideoWriter('monteSparse.avi');
lowVid.FrameRate = video.FrameRate;

open(lowVid)
for i = 1:video.numFrames
    img = mat2gray(reshape(vidX(:, i), [video.Height, video.Width]));
    writeVideo(lowVid, img);
end
close(lowVid)
