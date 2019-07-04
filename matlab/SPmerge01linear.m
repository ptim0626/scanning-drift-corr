function [sMerge] = SPmerge01linear(scanAngles,varargin)

% Colin Ophus, National Center for Electron Microscopy, Molecular Foundry,
% Lawrence Berkeley National Laboratory, Berkeley, CA, USA. (Feb 2016).

% New version of SPmerge01.m - This script now searches over linear drift
% vectors, aligned to first two images.  This search is performed twice.

% Merge multiple scanning probe images.  Assume scan origin is upper left
% corner of the image, and that the scan direction for zero degrees is
% horizontal (along MATLAB columns).  All input images must have fast scan
% direction along the array rows (horizontal direction).  Original data is 
% stored in 3D arrray sMerge.scanLines

% Inputs:
% scanAngles            - Vector containg scan angles in degrees.
% images                - 2D image arrays, order in scanAngles, all same size.
% sigmaLP = 32; %       - sigma value in pixels for initial alignment LP filter. 
flagReportProgress = 1;  % Set to true to see updates on console.v =
paddingScale = (1+1/4);%1.5;    % - padding amount for scaling of the output.
sMerge.KDEsigma = 1/2; % - Smoothing between pixels for KDE.
sMerge.edgeWidth = 1/128; % - size of edge blending relative to input images.
sMerge.linearSearch = linspace(-0.02,0.02,1+2*2);  % Initial linear search vector, relative to image size.
% sMerge.linearSearch = linspace(-0.04,0.04,1+2*4);  % Initial linear search vector, relative to image size.

% flagSkipInitialAlignment = 0;  % Set to true to skip initial phase correlation.
% flagCrossCorrelation = 0+1;  % Set to true to use cross correlation.
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
end

if flagReportProgress == true
    reverseStr = ''; % initialize console message piece
end


% First linear alignment, search over possible linear drift vectors.
sMerge.linearSearch = sMerge.linearSearch * size(sMerge.scanLines,1);
[yDrift,xDrift] = meshgrid(sMerge.linearSearch);
sMerge.linearSearchScore1 = zeros(length(sMerge.linearSearch));
inds = linspace(-0.5,0.5,size(sMerge.scanLines,1))';
N = size(sMerge.scanLines);
w2 = circshift(padarray(hanningLocal(N(1))*hanningLocal(N(2))',...
    sMerge.imageSize-N(1:2),0,'post'),round((sMerge.imageSize-N(1:2))/2));
% IcorrNorm = ifft2(abs(fft2(w2)).^2,'symmetric');
for a0 = 1:length(sMerge.linearSearch)
    for a1 = 1:length(sMerge.linearSearch)
        % Calculate time dependent lienar drift
        xyShift = [inds*xDrift(a0,a1) inds*yDrift(a0,a1)];
        
        % Apply linear drift to first two images
        sMerge.scanOr(:,:,1:2) = sMerge.scanOr(:,:,1:2) ...
            + repmat(xyShift,[1 1 2]);
        % Generate trial images
        sMerge = SPmakeImage(sMerge,1);
        sMerge = SPmakeImage(sMerge,2);
        % Measure alignment score with hybrid correlation
        m = fft2(w2.*sMerge.imageTransform(:,:,1)) ...
            .* conj(fft2(w2.*sMerge.imageTransform(:,:,2)));
        Icorr = ifft2(sqrt(abs(m)).*exp(1i*angle(m)),'symmetric');
        sMerge.linearSearchScore1(a0,a1) = max(Icorr(:));
        % Remove linear drift from first two images
        sMerge.scanOr(:,:,1:2) = sMerge.scanOr(:,:,1:2) ...
            - repmat(xyShift,[1 1 2]);
        if flagReportProgress == true
            comp = (a1 / length(sMerge.linearSearch) ...
                + a0 - 1) / length(sMerge.linearSearch);
            msg = sprintf(['Initial Linear Drift Search = ' ...
                sprintf('%.02f',100*comp) ' percent complete']);
            fprintf([reverseStr, msg]);
            reverseStr = repmat(sprintf('\b'),1,length(msg));
        end
    end
end


% Second linear alignment, refine possible linear drift vectors.
[~,ind] = max(sMerge.linearSearchScore1(:));
[xInd,yInd] = ind2sub(size(sMerge.linearSearchScore1),ind);
% if xInd == 1
%     xInds = 1:2;
% elseif xInd == size(sMerge.linearSearchScore1,1)
%     xInds = (-1:0) + size(sMerge.linearSearchScore1,1);
% else
%     if sMerge.linearSearchScore1(xInd-1,yInd) ...
%             > sMerge.linearSearchScore1(xInd+1,yInd)
%         xInds = xInd + (-1:0);
%     else
%         xInds = xInd + (0:1);
%     end
% end
% if yInd == 1
%     yInds = 1:2;
% elseif yInd == size(sMerge.linearSearchScore1,2)
%     yInds = (-1:0) + size(sMerge.linearSearchScore1,2);
% else
%     if sMerge.linearSearchScore1(xInd,yInd-1) ...
%             > sMerge.linearSearchScore1(xInd,yInd+1)
%         yInds = yInd + (-1:0);
%     else
%         yInds = yInd + (0:1);
%     end
% end
step = sMerge.linearSearch(2) - sMerge.linearSearch(1);
xRefine = sMerge.linearSearch(xInd) ...
    + linspace(-0.5,0.5,length(sMerge.linearSearch))*step;
yRefine = sMerge.linearSearch(yInd) ...
    + linspace(-0.5,0.5,length(sMerge.linearSearch))*step;
[yDrift,xDrift] = meshgrid(yRefine,xRefine);
sMerge.linearSearchScore2 = zeros(length(sMerge.linearSearch));
for a0 = 1:length(sMerge.linearSearch)
    for a1 = 1:length(sMerge.linearSearch)
        % Calculate time dependent lienar drift
        xyShift = [inds*xDrift(a0,a1) inds*yDrift(a0,a1)];
        
        % Apply linear drift to first two images
        sMerge.scanOr(:,:,1:2) = sMerge.scanOr(:,:,1:2) ...
            + repmat(xyShift,[1 1 2]);
        % Generate trial images
        sMerge = SPmakeImage(sMerge,1);
        sMerge = SPmakeImage(sMerge,2);
        % Measure alignment score with hybrid correlation
        m = fft2(w2.*sMerge.imageTransform(:,:,1)) ...
            .* conj(fft2(w2.*sMerge.imageTransform(:,:,2)));
        Icorr = ifft2(sqrt(abs(m)).*exp(1i*angle(m)),'symmetric');
        sMerge.linearSearchScore2(a0,a1) = max(Icorr(:));
        % Remove linear drift from first two images
        sMerge.scanOr(:,:,1:2) = sMerge.scanOr(:,:,1:2) ...
            - repmat(xyShift,[1 1 2]);
        if flagReportProgress == true
            comp = (a1 / length(sMerge.linearSearch) ...
                + a0 - 1) / length(sMerge.linearSearch);
            msg = sprintf(['Linear Drift Refinement = ' ...
                sprintf('%.02f',100*comp) ' percent complete']);
            fprintf([reverseStr, msg]);
            reverseStr = repmat(sprintf('\b'),1,length(msg));
        end
    end
end
[~,ind] = max(sMerge.linearSearchScore2(:));
[xInd,yInd] = ind2sub(size(sMerge.linearSearchScore2),ind);
sMerge.xyLinearDrift = [xDrift(xInd) yDrift(yInd)];


% Apply linear drift to all images
xyShift = [inds*sMerge.xyLinearDrift(1) inds*sMerge.xyLinearDrift(2)];
for a0 = 1:sMerge.numImages
    sMerge.scanOr(:,:,a0) = sMerge.scanOr(:,:,a0) + xyShift;
    sMerge = SPmakeImage(sMerge,a0);
end

% Estimate initial alignment
dxy = zeros(sMerge.numImages,2);
G1 = fft2(w2.*sMerge.imageTransform(:,:,1));
for a0 = 2:sMerge.numImages
    G2 = fft2(w2.*sMerge.imageTransform(:,:,a0));
    m = G1.*conj(G2);
    Icorr = ifft2(sqrt(abs(m)).*exp(1i*angle(m)),'symmetric');
    [~,ind] = max(Icorr(:));
    [dx,dy] = ind2sub(size(Icorr),ind);
    dx = mod(dx - 1 + size(Icorr,1)/2,size(Icorr,1)) - size(Icorr,1)/2;
    dy = mod(dy - 1 + size(Icorr,2)/2,size(Icorr,2)) - size(Icorr,2)/2;
    dxy(a0,:) = dxy(a0-1,:) + [dx dy];
    G1 = G2;
end
dxy(:,1) = dxy(:,1) - mean(dxy(:,1));
dxy(:,2) = dxy(:,2) - mean(dxy(:,2));
% Apply alignments and regenerate images
for a0 = 1:sMerge.numImages
    sMerge.scanOr(:,1,a0) = sMerge.scanOr(:,1,a0) + dxy(a0,1);
    sMerge.scanOr(:,2,a0) = sMerge.scanOr(:,2,a0) + dxy(a0,2);
    sMerge = SPmakeImage(sMerge,a0);
end
% Set reference point
sMerge.ref = round(sMerge.imageSize/2);


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

if flagReportProgress == true
    fprintf([reverseStr ' ']);
end
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