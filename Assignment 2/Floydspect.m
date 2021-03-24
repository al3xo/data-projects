%% setup
clear; close all; clc;

[og_y, Fs] = audioread('Floyd.m4a');
og_y = og_y(1:length(og_y)-1);

n = length(og_y)/5;
splitCount = 1; % break it into 5 pieces, and select which piece to analyze
b = (splitCount-1)*n + 1;
e = splitCount*n;
y = og_y(b:e, 1);

tr_floyd = length(y)/Fs; %record time in seconds

%% fourier transform
L = tr_floyd; n = length(y);
t2 = linspace(0,L,n+1); t = t2(1:n);
k = (1/L)*[0:n/2-1 -n/2:-1];
ks = fftshift(k);

yt = fft(y);
yt_s = fftshift(yt);

plot(ks,abs(yt_s),'r','Linewidth',2);
set(gca,'Fontsize',16)
xlabel('frequency (k)'), ylabel('fft(yt)')

%% gabor filtering & spectrogram 
tau = 0:0.25:tr_floyd;
a = 100;
ygt_spec = zeros(length(t), length(tau));
for j = 1:length(tau)
   g = exp(-a*(t - tau(j)).^2);
   yg = g.*(y');
   ygt = fft(yg);
   
   ygt_spec(:,j) = fftshift(abs(ygt));
end
pcolor(tau,ks,log(abs(ygt_spec) + 1))
shading interp 
colormap(hot)
xlabel('time (t)')
ylabel('frequency (k)')
ylim([0 150])
yyaxis right
yticks([82.41 87.31 98.00 110.00 123.47])
yticklabels({'E','F','G','A','B'})
ylim([0 150])
set(get(gca,'YLabel'),'rotation',-90,'VerticalAlignment','bottom')
ylabel('Notes')


