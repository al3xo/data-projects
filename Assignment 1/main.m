% Clean workspace
clear all; close all; clc

load subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); 
x = x2(1:n); y = x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; 
ks = fftshift(k);

[X,Y,Z] = meshgrid(x,y,z);
[Kx,Ky,Kz] = meshgrid(ks,ks,ks);

%% average fourier transform and determine central frequency
UtnAvg = zeros(n,n,n);
for j=1:49
    Un = reshape(subdata(:,j),n,n,n);
    Utn = fftn(Un);
    UtnAvg = UtnAvg + Utn;
end

UtnAvg = abs(fftshift(UtnAvg))/49;

M = max(UtnAvg,[],'all');
[Mx,My,Mz] = ind2sub(size(UtnAvg),find(abs(UtnAvg)== M))

Mx = (2*pi/(2*L))*Mx
My = (2*pi/(2*L))*My
Mz = (2*pi/(2*L))*Mz

%% filter original Un's and then shift back to space domain
tau = 0.1;
filter = exp(-tau*(Kx - Mx).^2).*exp(-tau*(Ky - My).^2).*exp(-tau*(Kz - Mz).^2);

coords = zeros(49, 3);
for j = 1:49
    Utn = fftshift(fftn(reshape(subdata(:,j),n,n,n)));
    Utnf = Utn.*filter;
    
    Unf = ifftn(ifftshift(Utnf));
    M = max(abs(Unf),[],'all');
    [coords(j, 1), coords(j, 2), coords(j, 3)] = ind2sub(size(Unf),find(abs(Unf)== M));
end 


plot3(coords(:, 1), coords(:, 2), coords(:, 3))
xlabel("x axis")
ylabel("y axis")
zlabel("z axis")

