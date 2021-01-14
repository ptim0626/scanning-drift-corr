function [sMerge] = SPmerge01(scanAngles,varargin)

% Colin Ophus, National Center for Electron Microscopy, Molecular Foundry,
% Lawrence Berkeley National Laboratory, Berkeley, CA, USA. (Feb 2016).

% Merge multiple scanning probe images.  Assume scan origin is upper left
% corner of the image, and that the scan direction for zero degrees is
% horizontal (along MATLAB columns).

% Inputs:
% scanAngles            - Vector containg scan angles in degrees.
% images                - 2D image arrays, order in scanAngles, all same size.
% sigmaLP = 32; %       - sigma value in pixels for initial alignment LP filter. 
paddingScale = (1+1/4);%1.6;%1.125;    % - padding amount for scaling of the output.
sMerge.KDEsigma = 0.5*4; % - Smoothing between samples for KDE.
sMerge.edgeWidth = 4; % - size of edge blending in pixels.
flagSkipInitialAlignment = 0;  % Set to true to skip initial phase correlation.
flagCrossCorrelation = 1;  % Set to true to use cross correlation.
% sigmaAlignLP = 64;  % low pass filtering for initial alignment (pixels).

% Initialize struct containing all data
sMerge.imageSize = round([size(varargin{1},1) size(varargin{1},2)]*paddingScale/4)*4;
sMerge.scanAngles = scanAngles;
sMerge.numImages = length(scanAngles);
sMerge.scanLines = zeros(size(varargin{1},1),size(varargin{1},2),sMerge.numImages);
sMerge.scanOr = zeros(size(varargin{1},1),2,sMerge.numImages);
sMerge.scanDir = zeros(sMerge.numImages,2);
sMerge.imageTransform = zeros(sMerge.imageSize(1),sMerge.imageSize(2),sMerge.numImages);
sMerge.imageDensity = zeros(sMerge.imageSize(1),sMerge.imageSize(2),sMerge.numImages);


% Generate all initial variables 
for a0 = 1:sMerge.numImages
    % Raw images
    if nargin == (sMerge.numImages+1)
        % Sequential images
        sMerge.scanLines(:,:,a0) = varargin{a0};
    elseif nargin == 2
        % 3D image stack
         sMerge.scanLines(:,:,a0) = varargin{1}(:,:,a0);
    else
        
    end
    
    % Generate pixel origins
    if nargin == (sMerge.numImages+1)
        xy = [(1:size(varargin{a0},1))' ones(size(varargin{a0},1),1)];
        xy(:,1) = xy(:,1) - size(varargin{a0},1)/2;
        xy(:,2) = xy(:,2) - size(varargin{a0},2)/2;
    elseif nargin == 2
        xy = [(1:size(varargin{1}(:,:,a0),1))' ones(size(varargin{1}(:,:,a0),1),1)];
        xy(:,1) = xy(:,1) - size(varargin{1}(:,:,a0),1)/2;
        xy(:,2) = xy(:,2) - size(varargin{1}(:,:,a0),2)/2;
    end
    xy = [xy(:,1)*cos(scanAngles(a0)*pi/180) ...
        - xy(:,2)*sin(scanAngles(a0)*pi/180) ...
        xy(:,2)*cos(scanAngles(a0)*pi/180) ...
        + xy(:,1)*sin(scanAngles(a0)*pi/180)];
    xy(:,1) = xy(:,1) + sMerge.imageSize(1)/2;
    xy(:,2) = xy(:,2) + sMerge.imageSize(2)/2;
    xy(:,1) = xy(:,1) - mod(xy(1,1),1);
    xy(:,2) = xy(:,2) - mod(xy(1,2),1);
    sMerge.scanOr(:,:,a0) = xy;

    % Scan direction
    sMerge.scanDir(a0,:) = [cos(scanAngles(a0)*pi/180 + pi/2) ...
        sin(scanAngles(a0)*pi/180 + pi/2)];
    
    % Generate initial resampled images 
    sMerge = SPmakeImage(sMerge,a0);
end


if flagSkipInitialAlignment == 0
    % Initial alignment between images
    filt = circshift(padarray(hanningLocal(size(varargin{1},1)) ...
        * hanningLocal(size(varargin{1},2))',...
        sMerge.imageSize-[size(varargin{1},1) size(varargin{1},2)],0,'post'),...
        round((sMerge.imageSize-[size(varargin{1},1) size(varargin{1},2)])/2));
    G0 = fft2(sMerge.imageTransform(:,:,1).*filt);
    for a0 = 2:sMerge.numImages
        G = fft2(sMerge.imageTransform(:,:,a0).*filt);
        
        % Replace dftregistration with simple phase / cross correlation
        if flagCrossCorrelation == false
            Icorr = abs(ifft2(exp(1i*angle(G0.*conj(G)))));
        else
            Icorr = abs(ifft2(G0.*conj(G)));
        end
        % Get peak maximum
        [~,ind] = max(Icorr(:));
        [xInd,yInd] = ind2sub(sMerge.imageSize,ind);
        
        % Compute relative shifts.  Note that since matrix indices
        % start at 1, must be shifted by -1.
        dx = mod(xInd-1+sMerge.imageSize(1)/2,...
            sMerge.imageSize(1))-sMerge.imageSize(1)/2;
        dy = mod(yInd-1+sMerge.imageSize(2)/2,...
            sMerge.imageSize(2))-sMerge.imageSize(2)/2;
        
        sMerge.scanOr(:,1,a0) = sMerge.scanOr(:,1,a0) + dx;
        sMerge.scanOr(:,2,a0) = sMerge.scanOr(:,2,a0) + dy;
        sMerge = SPmakeImage(sMerge,a0);
    end
end

% Determine initial starting index, from difference image weighted away
% from center of image.
 [ya,xa] = meshgrid((1:sMerge.imageSize(2))-sMerge.imageSize(2)/2,...
    (1:sMerge.imageSize(1))-sMerge.imageSize(1)/2);
weight = xa.^2/(sMerge.imageSize(1)^2/8) ...
    + ya.^2/(sMerge.imageSize(2)^2/8) + 1;
Idiff = weight.*(mean(abs(sMerge.imageTransform ...
    - repmat(mean(sMerge.imageTransform,3),[1 1 sMerge.numImages])),3));
sub = min(sMerge.imageDensity==1,[],3) == 1;
Idiff(~sub) = max(Idiff(:));
sm = fspecial('disk',31);
Idiff = conv2(Idiff,sm,'same') ./ conv2(ones(size(Idiff)),sm,'same');
[~,ind] = min(Idiff(:));
[xRef,yRef] = ind2sub(sMerge.imageSize,ind);
sMerge.ref = [xRef yRef];


% Plot results, image with scanline origins overlaid
imagePlot = mean(sMerge.imageTransform,3);
dens = prod(sMerge.imageDensity,3);
% Scale intensity of image
mask = dens>0.5;
imagePlot = imagePlot - mean(imagePlot(mask));
imagePlot = imagePlot / sqrt(mean(imagePlot(mask).^2));

figure(1)
clf
imagesc(imagePlot)
hold on
cvals = [1 0 0;
    0 .7 0;
    0 .6 1;
    1 .7 0;
    1 0 1;
    0 0 1];
for a0 = 1:sMerge.numImages
    scatter(sMerge.scanOr(:,2,a0),sMerge.scanOr(:,1,a0),'marker','.',...
        'sizedata',25,'markeredgecolor',cvals(mod(a0-1,size(cvals,1))+1,:))
end
scatter(sMerge.ref(2),sMerge.ref(1),...
    'marker','+','sizedata',500,...
    'markeredgecolor',[1 1 0],'linewidth',6)
scatter(sMerge.ref(2),sMerge.ref(1),...
    'marker','+','sizedata',500,...
    'markeredgecolor','k','linewidth',2)
hold off
axis equal off
colormap(gray(256))
set(gca,'position',[0 0 1 1])
caxis([-3 3])   % units of image RMS 

end

% function [Iscale] = scaleImage(I,intRange,sigmaLP)
% Iscale = I - mean(I(:));
% Iscale = Iscale / sqrt(mean(I(:).^2));
% Iscale = (Iscale - intRange(1))/(intRange(2)-intRange(1));
% if sigmaLP > 0
%     sm = fspecial('gaussian',2*ceil(3*sigmaLP)+1,sigmaLP);
%     Iscale = conv2(Iscale,sm,'same') ...
%         ./ conv2(ones(size(Iscale)),sm,'same');
% end
% Iscale(Iscale<0) = 0;
% Iscale(Iscale>1) = 1;
% end

function [w] = hanningLocal(N)
% Replacement for simplest 1D hanning function to remove dependency.
w = sin(pi*((1:N)'/(N+1))).^2;
end