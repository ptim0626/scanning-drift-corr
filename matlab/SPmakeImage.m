function [sMerge] = SPmakeImage(sMerge,indImage,indLines)

% This function generates a resampled scanning probe image with dimensions
% of imageSize, from a an array of N scan lines given in scaneLines,
% (lines specified as image rows), from an array of Nx2 origins in scanOr.
% scanDir is a 2 element vector specifying the direction of the scan.
% All arrays are stored inside struct sMerge.  ind specified update index.
% indLines is a vector of binary values specifying which lines to include.

if nargin == 2
    %     indLines = 1:size(sMerge.scanLines,2);
    indLines = true(1,size(sMerge.scanLines,2));
end

% Expand coordinates
t = repmat(1:size(sMerge.scanLines,2),[sum(indLines) 1]);
x0 = repmat(sMerge.scanOr(indLines,1,indImage),[1 size(sMerge.scanLines,2)]);
y0 = repmat(sMerge.scanOr(indLines,2,indImage),[1 size(sMerge.scanLines,2)]);
xInd = x0(:) + t(:)*sMerge.scanDir(indImage,1);
yInd = y0(:) + t(:)*sMerge.scanDir(indImage,2);

% Prevent pixels from leaving image boundaries
xInd = max(min(xInd,sMerge.imageSize(1)-1),1);
yInd = max(min(yInd,sMerge.imageSize(2)-1),1);

% Convert to bilinear interpolants and weights
xIndF = floor(xInd);
yIndF = floor(yInd);
xAll = [xIndF xIndF+1 xIndF xIndF+1];
yAll = [yIndF yIndF yIndF+1 yIndF+1];
dx = xInd-xIndF;
dy = yInd-yIndF;
w = [(1-dx).*(1-dy) dx.*(1-dy) (1-dx).*dy dx.*dy];
indAll = sub2ind(sMerge.imageSize,xAll,yAll);

% Generate image
sL = sMerge.scanLines(indLines,:,indImage);
sig = reshape(accumarray(indAll(:),[ ...
    w(:,1).*sL(:);
    w(:,2).*sL(:);
    w(:,3).*sL(:);
    w(:,4).*sL(:)],...
    [prod(sMerge.imageSize) 1]),sMerge.imageSize);
count = reshape(accumarray(indAll(:),[ ...
    w(:,1);w(:,2);w(:,3);w(:,4)],...
    [prod(sMerge.imageSize) 1]),sMerge.imageSize);

% Apply KDE
r = max(ceil(sMerge.KDEsigma*3),5);
sm = fspecial('gaussian',2*r+1,sMerge.KDEsigma);
sm = sm / sum(sm(:));
sig = conv2(sig,sm,'same');
count = conv2(count,sm,'same');
sub = count > 0;
sig(sub) = sig(sub) ./ count(sub);
sMerge.imageTransform(:,:,indImage) = sig;

% Estimate sampling density
bound = count == 0;
bound([1 end],:) = true;
bound(:,[1 end]) = true;
sMerge.imageDensity(:,:,indImage) = ...
    sin(min(bwdist(bound)/sMerge.edgeWidth,1)*pi/2).^2;


% % Plot testing
% % dens = sMerge.imageDensity(:,:,indImage);
% figure(1)
% clf
% imagesc(sig)
% % imagesc(sig.*dens + (1-dens)*mean(sig(dens>.5)))
% % imagesc(sMerge.imageTransform(:,:,indImage))
% axis equal off
% set(gca,'position',[0 0 1 1])
% % imagesc(t)
% % hold on
% % ind = 17:3213:numel(xInd);
% % scatter(yInd(ind),xInd(ind),'r.')
% % % hold off
% % % axis equal off
% % % colormap(gray(256))


end