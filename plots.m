filename = 'output.csv';

data = csvread(filename);
% data = data.^2;
%data = data / max(max(data));
% data = 10*log10(data);
fs=240e3; %Sampling rate
freq=99.3e6; %Frequency
t=1; %Integration interval
nPulses = 500;
nmax=1 ; %50

dopplerScale = (1:size(data,1))*fs/(nPulses^2);
dopplerScale = dopplerScale - max(dopplerScale)/2;
delayScale = (1:size(data,2))/fs*3e8/1e3;
scale=[-70,0];

figure(1);

imagesc(delayScale(1:120),dopplerScale,data(:,1:150));
xlabel('Bistatic range [km]');
ylabel('Doppler [Hz]');

%hold on
%scale=[0,100000];

%imagesc(pa, scale);
colorbar;
%
% imagesc(delayScale,dopplerScale,detTot)
% caxis([0,1])
% %colormap(flipud(gray))
% xlabel('Bistatic range [km]')
% ylabel('Doppler [Hz]')