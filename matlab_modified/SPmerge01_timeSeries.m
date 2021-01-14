function [sMerge] = SPmerge01_timeSeries(scanAngle,imageRef,stack)

% Colin Ophus, National Center for Electron Microscopy, Molecular Foundry,
% Lawrence Berkeley National Laboratory, Berkeley, CA, USA. (Mar 2015).

% Merge multiple scanning probe images.  Assume scan origin is upper left
% corner of the image, and that the scan direction for zero degrees is
% horizontal (along MATLAB columns).

% Inputs:
% scanAngles            - Vector containg scan angles in degrees.
% images                - 2D image arrays, order in scanAngles, all same size.
% sigmaLP = 32; %       - sigma value in pixels for initial alignment LP filter. 
% paddingScale = 1.2+0.2*0;    % - padding amount for scaling of the output.
% sMerge.KDEsigma = 2; % - Smoothing between samples for KDE.
% sMerge.edgeWidth = 8; % - size of edge blending in pixels.
sMerge.KDEsigma = 1/2*4; % - Smoothing between samples for KDE.
sMerge.edgeWidth = 4; % - size of edge blending in pixels.
paddingScale =  size(imageRef) ./ size(stack(:,:,1));
flag_use_cross_corr = 1;

% Initialize struct containing all data
sMerge.imageRef = imageRef;
sMerge.imageSize = round(size(stack(:,:,1)).*paddingScale/2)*2;
sMerge.numImages = size(stack,3);
sMerge.scanAngles = scanAngle*ones(sMerge.numImages,1);
sMerge.scanLines = stack;
sMerge.scanOr = zeros(size(stack,1),2,sMerge.numImages);
sMerge.scanDir = zeros(sMerge.numImages,2);
sMerge.imageTransform = zeros(sMerge.imageSize(1),sMerge.imageSize(2),sMerge.numImages);
sMerge.imageDensity = zeros(sMerge.imageSize(1),sMerge.imageSize(2),sMerge.numImages);

% % Set this field to ignore initial alignment
% sMerge.scanActive = 0;  

% Generate all initial variables 
for a0 = 1:sMerge.numImages
    %     % Raw images
    %     sMerge.scanLines(:,:,a0) = varargin{a0};
    
    % Generate pixel origins
    xy = [(1:size(stack,1))' ones(size(stack,1),1)];
    xy(:,1) = xy(:,1) - size(stack,1)/2;
    xy(:,2) = xy(:,2) - size(stack,2)/2;
    if length(scanAngle) == 1
        xy = [xy(:,1)*cos(scanAngle*pi/180) ...
            - xy(:,2)*sin(scanAngle*pi/180) ...
            xy(:,2)*cos(scanAngle*pi/180) ...
            + xy(:,1)*sin(scanAngle*pi/180)];
    else
        xy = [xy(:,1)*cos(scanAngle(a0)*pi/180) ...
            - xy(:,2)*sin(scanAngle(a0)*pi/180) ...
            xy(:,2)*cos(scanAngle(a0)*pi/180) ...
            + xy(:,1)*sin(scanAngle(a0)*pi/180)];
    end
    xy(:,1) = xy(:,1) + sMerge.imageSize(1)/2;
    xy(:,2) = xy(:,2) + sMerge.imageSize(2)/2;
    xy(:,1) = xy(:,1) - mod(xy(1,1),1);
    xy(:,2) = xy(:,2) - mod(xy(1,2),1);
    
    sMerge.scanOr(:,:,a0) = xy;

    % Scan direction
    if length(scanAngle) == 1
        sMerge.scanDir(a0,:) = [cos(scanAngle*pi/180 + pi/2) ...
            sin(scanAngle*pi/180 + pi/2)];
    else
         sMerge.scanDir(a0,:) = [cos(scanAngle(a0)*pi/180 + pi/2) ...
            sin(scanAngle(a0)*pi/180 + pi/2)];
    end
    
    % Generate initial resampled images 
    sMerge = SPmakeImage(sMerge,a0);
end


% Initial alignment between images
[ya,xa] = meshgrid((1:sMerge.imageSize(2))-sMerge.imageSize(2)/2,...
    (1:sMerge.imageSize(1))-sMerge.imageSize(1)/2);
% filt = exp(-(xa.^2 + ya.^2)/((1/2)*size(stack,1)^2));
filt = exp(-(xa.^2 + ya.^2)/((1/8)*size(stack,1)^2));
% G0 = fft2(imageRef);
G0 = fft2(imageRef.*filt);
for a0 = 1:sMerge.numImages
    G = fft2(sMerge.imageTransform(:,:,a0).*filt);
    
    % Phase correlation
    if flag_use_cross_corr == 0
        phaseCorr = abs(ifft2(exp(1i*angle(G0.*conj(G)))));
        % Replace dftregistration with simple cross correlation
    else
        phaseCorr = abs(ifft2(G0.*conj(G)));
    end
    % Get peak maximum
    [~,ind] = max(phaseCorr(:));
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


% Determine initial starting index, from difference image weighted away
% from center of image.
weight = xa.^2/(sMerge.imageSize(1)^2/4) ...
    + ya.^2/(sMerge.imageSize(2)^2/4) + 1;
Idiff = weight.*(mean(abs(sMerge.imageTransform ...
    - repmat(mean(sMerge.imageTransform,3),[1 1 sMerge.numImages])),3));
sub = min(sMerge.imageDensity==1,[],3) == 1;
Idiff(~sub) = max(Idiff(:));
% sm = fspecial('gaussian',51,15);
sm = fspecial('disk',51);
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
scatter(sMerge.ref(2),sMerge.ref(2),...
    'marker','+','sizedata',500,...
    'markeredgecolor',[1 1 0],'linewidth',6)
scatter(sMerge.ref(2),sMerge.ref(2),...
    'marker','+','sizedata',500,...
    'markeredgecolor','k','linewidth',2)
hold off
axis equal off
colormap(gray(256))
set(gca,'position',[0 0 1 1])
caxis([-3 3])   % units of image RMS 










% % densMean = mean(sMerge.imageDensity,3) == 1;
% 
% % inds = zero(sMerge.imageSize);
% % inds(:) = 1:prod(sMerge.imageSize);
% % indsSub = inds(sub);
% % IdiffSub = Idiff(sub);
% % [val,ind] = min(

% figure(1)
% clf
% imagesc(Idiff)
% % % imagesc(densMean)
% % % imagesc(sMerge.imageTransform(:,:,1)+sMerge.imageTransform(:,:,2))
% % hold on
% % scatter(sMerge.ref(:,2),sMerge.ref(:,1),'marker','+','sizedata',200,...
% %     'markerfacecolor','none','markeredgecolor','g')
% % hold off
% axis equal off
% colormap(gray(256))
% set(gca,'position',[0 0 1 1])
% 
% 
% 
% 
% % I = sMerge.imageTransform(:,:,2).*sMerge.imageDensity(:,:,2) ...
% %     + (1-sMerge.imageDensity(:,:,2)) ...
% %     .*mean(mean(sMerge.imageTransform(:,:,2)));
% % Ip1 = scaleImage(I,[-3 3],sigmaLP);
% % 
% % % Select initial alignment marker
% % figure(1)
% % clf
% % imagesc(Ip1)
% % axis equal off
% % colormap(gray(256))
% % set(gca,'position',[0 0 1 1])
% % % % Get reference point
% % % [ym,xm] = ginput(1);
% % % xyRef = [xm ym];
% % % hold on
% % % scatter(xyRef(:,2),xyRef(:,1),'marker','+','sizedata',200,...
% % %     'markerfacecolor','none','markeredgecolor','g')
% % % hold off
% 
% 
% % figure(1)
% % clf
% % imagesc(ones(sMerge.imageSize))
% % hold on
% % % sub = round(2.^(0:.2:10));
% % sub = [1  100:100:1000];
% % scatter(xy(sub,2),xy(sub,1),'r.')
% % ind = 2;
% % line(200*[0 sMerge.scanDir(ind,2)]+sMerge.imageSize(2)/2,...
% %      200*[0 sMerge.scanDir(ind,1)]+sMerge.imageSize(1)/2,...
% %     'linewidth',2,'color','g')
% % line(20*[0 sMerge.scanDir(ind,2)]+sMerge.imageSize(2)/2,...
% %      20*[0 sMerge.scanDir(ind,1)]+sMerge.imageSize(1)/2,...
% %     'linewidth',8,'color','g')
% % hold off
% % % imagesc(s
% % % imagesc([LP1 rot90(LP2,1)])
% % axis equal off
% % % colormap(violetFire(256))
% % set(gca,'position',[0 0 1 1])
% % colormap(gray(256))




end


function [Iscale] = scaleImage(I,intRange,sigmaLP)
Iscale = I - mean(I(:));
Iscale = Iscale / sqrt(mean(I(:).^2));
Iscale = (Iscale - intRange(1))/(intRange(2)-intRange(1));
if sigmaLP > 0
    sm = fspecial('gaussian',2*ceil(3*sigmaLP)+1,sigmaLP);
    Iscale = conv2(Iscale,sm,'same') ...
        ./ conv2(ones(size(Iscale)),sm,'same');
end
Iscale(Iscale<0) = 0;
Iscale(Iscale>1) = 1;
end