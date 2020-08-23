clearvars;close all;clc;

load net_1024.mat
load tblValid.mat

layername = 'Routput';
im = tblValid.XValid{24};

tic
act1 = activations(net,im,layername);
toc

[sz1,sz2,sz3] = size(act1);
act1 = reshape(act1,[sz1 sz2 1 sz3]);

f = factor(sz3);
l = length(f);
m = prod(f(1:ceil(l/2)));
n = prod(f(ceil(l/2)+1:end));
rows = min(m,n);
cols = max(m,n);

I = imtile(act1,'GridSize',[rows cols]);
fig = figure(1);
% set(fig,'Position',[1920 0 1920 995]);

% if ~exist('hi','var')
    hi = imagesc(I);
    axis image
    colormap gray
    colorbar
% else
    set(hi,'CData',I);
    axis image
    drawnow
% end

title([layername,': ',num2str(sz1),'x',num2str(sz2),'x',num2str(sz3)])