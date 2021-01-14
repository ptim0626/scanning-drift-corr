function [imageRef] = makeRefUnitCell(s)

% 2020 Jan - Colin Ophus

% Make refefence image from mean unit cell
padding = 64;%32;
edgeBlend = 4;


% UC coordinates
UC = s.UCmean*s.imageIntSD + s.imageIntMean;
lat = s.lat;
r0 = lat(1,:);
u = lat(2,:);
v = lat(3,:);
UCsize = s.UCsize;

% Image coordinates
Nuc = size(UC);
Nim = size(s.image);
[ya,xa] = meshgrid( ...
    (1:Nim(2)) - r0(2),...
    (1:Nim(1)) - r0(1));


ab = ([u;v]' \ [xa(:) ya(:)]')';
aInd = mod(round(ab(:,1)*Nuc(1))-1,Nuc(1))+1;
bInd = mod(round(ab(:,2)*Nuc(2))-1,Nuc(2))+1;
ind = sub2ind(UCsize,aInd,bInd);
imageRef = reshape(UC(ind),Nim);

% Apply padding and edge blending
imageMed = median(imageRef(:));
w2 = tukeywin(Nim(1),2*edgeBlend/Nim(1)) ...
    *tukeywin(Nim(2),2*edgeBlend/Nim(2))';
imageRef(:) = imageRef.*w2 + (1-w2)*imageMed;
imageRef = padarray(imageRef,[1 1]*padding/2,imageMed,'both');

figure(11)
clf
imagesc(imageRef)
% imagesc(reshape(mod(ab(:,1),1),Nim) ...
%     + reshape(mod(ab(:,2),1),Nim))
axis equal off
colormap(gray(256))
set(gca,'position',[0 0 1 1])

end

