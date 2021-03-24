%% setup
clear; close all; clc;


% figure(1)
[y, Fs] = audioread('GNR.m4a');
tr_gnr = length(y)/Fs; %record time in seconds
y = y(1:length(y))';

%% fourier transform
L = tr_gnr; n = length(y);
t2 = linspace(0,L,n+1); t = t2(1:n);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);

yt = fft(y);
yt_s = fftshift(yt);

[max_num, max_idx] = max(abs(yt_s));
[freq] = abs(ks(max_idx));

%% gabor filtering & spectrogram 

tau = 0:0.1:tr_gnr;
a = 400;

ygt_spec = zeros(length(t), length(tau));
for j = 1:length(tau)
   g = exp(-a*(t - tau(j)).^2);
   yg = g.*y;
   ygt = fft(yg);
   
   ygt_spec(:,j) = fftshift(abs(ygt));
end

figure(6)
pcolor(tau,ks,abs(ygt_spec))
shading interp
colormap(hot)

xlabel('time (t)')

ylim([200 500])
ylabel('frequency (k)')

yyaxis right
yticks([277.18 311.13 369.99 415.30])
yticklabels({'Db','Eb','Gb','Ab'})
ylim([200 500])
set(get(gca,'YLabel'),'rotation',-90,'VerticalAlignment','bottom')
ylabel('Notes')


