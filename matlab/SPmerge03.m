function [imageFinal,signalArray,densityArray] = SPmerge03(sMerge)
tic

% Colin Ophus, National Center for Electron Microscopy, Molecular Foundry,
% Lawrence Berkeley National Laboratory, Berkeley, CA, USA. (Mar 2015).

% Final scanning probe merge script.  This script uses KDE and Fourier
% filtering to produce a combined image from all component scans.  
% The Fourier weighting used (if flag is enabled) is cos(theta)^2, where
% theta is the angle from the scan direction.  This essentially zeros out
% the slow scan direction, while weighting the fast scan direction at 100%.

% Inputs:
% sMerge                -struct containing data for STEM alignment from SPmerge02
KDEsigma = 0.5 * 2; %       -Gaussian sigma value used in kernel density estimator. (pixels)
% KDEcutoff = 4;  %       -Cutoff limit of the kernel in units of sigma.
upsampleFactor = 2;  %  -upsampling factor used in image generation (integer).
%                        Using a large upsampleFactor can be very slow.
sigmaDensity = 8;     % -Smoothing sigma value for density estimation. (pixels)
boundary = 8;  %       -Thickness of windowed boundary. (pixels)
flagFourierWeighting = 1; % Set to true to enable cos(theta)^2 Fourier weights.
flagDownsampleOutput = 1;  % Set to true to downsample output to the original
%                            resolution, as opposed to that of "upsampleFactor."

% Outputs:
% imageCombine - final combined image.
% signalArray  - image stack containing estimated image from each scan.
% densityArray - image stack containing estimated density of each scan.


% Initialize arrays
signalArray = zeros([sMerge.imageSize*upsampleFactor sMerge.numImages]);
densityArray = zeros([sMerge.imageSize*upsampleFactor sMerge.numImages]);
imageFinal = zeros(sMerge.imageSize*upsampleFactor);


% kernel generation in upsampled coordinates
x = makeFourierCoords(upsampleFactor*sMerge.imageSize(1),...
    1/sMerge.imageSize(1))';
y = makeFourierCoords(upsampleFactor*sMerge.imageSize(2),...
    1/sMerge.imageSize(2));
kernel = fft2(exp(-x.^2/(2*KDEsigma^2)) * exp(-y.^2/(2*KDEsigma^2)));
smoothDensityEstimate = fft2(exp(-x.^2/(2*sigmaDensity^2)) ...
    *exp(-y.^2/(2*sigmaDensity^2)) ...
    / (2*pi*sigmaDensity^2*upsampleFactor^2));


% Loop over scans and create images / densities 
t = repmat(1:size(sMerge.scanLines,2),[size(sMerge.scanLines,1) 1]);
for a0 = 1:sMerge.numImages
    % Expand coordinates
    x0 = repmat(sMerge.scanOr(:,1,a0),[1 size(sMerge.scanLines,2)]);
    y0 = repmat(sMerge.scanOr(:,2,a0),[1 size(sMerge.scanLines,2)]);
    xInd = x0(:)*upsampleFactor - (upsampleFactor-1)/2 ...
        + (t(:)*sMerge.scanDir(a0,1))*upsampleFactor;
    yInd = y0(:)*upsampleFactor - (upsampleFactor-1)/2 ...
        + (t(:)*sMerge.scanDir(a0,2))*upsampleFactor;
    xInd(:) = min(max(xInd,1),sMerge.imageSize(1)*upsampleFactor-1);
    yInd(:) = min(max(yInd,1),sMerge.imageSize(2)*upsampleFactor-1);
    
    % Create bilinear coordinates
    xIndF = floor(xInd);
    yIndF = floor(yInd);
    xAll = [xIndF xIndF+1 xIndF xIndF+1];
    yAll = [yIndF yIndF yIndF+1 yIndF+1];
    dx = xInd-xIndF;
    dy = yInd-yIndF;
    w = [(1-dx).*(1-dy) dx.*(1-dy) (1-dx).*dy dx.*dy];
    
    % Generate image and density
    image = sMerge.scanLines(:,:,a0);
    signalArray(:,:,a0) = accumarray([xAll(:) yAll(:)],...
        [image(:).*w(:,1);
         image(:).*w(:,2);
         image(:).*w(:,3);
         image(:).*w(:,4)],...
        sMerge.imageSize*upsampleFactor);
    densityArray(:,:,a0) = accumarray([xAll(:) yAll(:)],...
        [w(:,1); w(:,2); w(:,3); w(:,4)],...
        sMerge.imageSize*upsampleFactor);
end


% Apply KDE to both arrays
for a0 = 1:sMerge.numImages
    signalArray(:,:,a0) = real(ifft2(fft2(signalArray(:,:,a0)).*kernel));
    densityArray(:,:,a0) = real(ifft2(fft2(densityArray(:,:,a0)).*kernel));    
end
% Normalize image intensity by sampling density
sub = densityArray > 1e-8;
signalArray(sub) = signalArray(sub) ./ densityArray(sub);


% Calculate smooth density estimate, set max value to 2 to reduce edge
% effects, apply to images.
intensityMedian = median(sMerge.scanLines(:));
for a0 = 1:sMerge.numImages
    densityArray(:,:,a0) = real(ifft2(fft2(min(densityArray(:,:,a0),2)) ...
        .* smoothDensityEstimate));
    densityArray(:,:,a0) = sin((pi/2)*min((bwdist( ...
        densityArray(:,:,a0) < 0.5) ...
        / (boundary*upsampleFactor)),1)).^2;
    % Apply mask to each image
    signalArray(:,:,a0) = signalArray(:,:,a0).*densityArray(:,:,a0) ...
        + (1-densityArray(:,:,a0))*intensityMedian;
end



% Combine scans to produce final image
if flagFourierWeighting == true
    % Make Fourier coordinates
    qx = makeFourierCoords(size(imageFinal,1),1);
    qy = makeFourierCoords(size(imageFinal,2),1);
    [qya,qxa] = meshgrid(qy,qx);
    qTheta = atan2(qya,qxa);

    % Generate Fourier weighted final image
    weightTotal = zeros(size(imageFinal));
    for a0 = 1:sMerge.numImages
        % Filter array
        thetaScan = atan2(sMerge.scanDir(a0,2),sMerge.scanDir(a0,1));
        qWeight = cos(qTheta-thetaScan).^2;
%         qWeight = abs(cos(qTheta-thetaScan));
        
        qWeight(1,1) = 1;
        
        % Filtered image
        imageFinal = imageFinal ...
            + fft2(signalArray(:,:,a0)).*qWeight;
        weightTotal = weightTotal + qWeight;
    end
    imageFinal = real(ifft2(imageFinal ./ weightTotal));
    
    % apply global density mask
    density = prod(densityArray,3);
    % density = max(sum(densityArray,3),1);  % Temporary line for huge drifts
    imageFinal = imageFinal.*density ...
        + (1-density)*intensityMedian;
else
   % Density weighted average
   density = prod(densityArray,3);
   % density = max(sum(densityArray,3),1);  % Temporary line for huge drifts
   imageFinal(:) = mean(signalArray,3).*density ...
       + (1-density)*median(sMerge.scanLines(:));
   imageFinal = imageFinal / upsampleFactor^2;  % Keep intensity constant
end


% Downsample outputs if required
if (flagDownsampleOutput == true) && upsampleFactor > 1
    % Fourier subsets for downsampling
    xVec = [(1:(sMerge.imageSize(1)/2)) ...
        (((-sMerge.imageSize(1)/2+1):0) ...
        + sMerge.imageSize(1)*upsampleFactor)];
    yVec = [(1:(sMerge.imageSize(2)/2)) ...
        (((-sMerge.imageSize(2)/2+1):0) ...
        + sMerge.imageSize(2)*upsampleFactor)];
    
    % Downsample output image
    imageFinal = fft2(imageFinal);
    imageFinal = real(ifft2(imageFinal(xVec,yVec))) / upsampleFactor^2;
    if nargout > 1
        signalArray = fft2(signalArray);
        signalArray = real(ifft2(signalArray(xVec,yVec,:)));
    end
    if nargout > 2
        densityArray = fft2(densityArray);
        densityArray = real(ifft2(densityArray(xVec,yVec,:)));
    end
end


figure(1)
clf
imagesc(imageFinal)
axis equal off
colormap(gray(256))
set(gca,'position',[0 0 1 1])

toc
end


function [q] = makeFourierCoords(N,pSize)
% This function generates Fourier coordinates 
if mod(N,2) == 0
    q = circshift(((-N/2):(N/2-1))/(N*pSize),[0 -N/2]);
else
    q = circshift(((-N/2+.5):(N/2-.5))/((N-1)*pSize),[0 -N/2+.5]);
end
end