function [sMerge] = SPmerge02(sMerge,refineMaxSteps,initialRefineSteps)
% tic

% Colin Ophus, National Center for Electron Microscopy, Molecular Foundry,
% Lawrence Berkeley National Laboratory, Berkeley, CA, USA. (Mar 2015).

% Refinement function for scanning probe image merge.  Requires struct
% input from SPmerge01.  This script will perform both the initial
% alignment if it has not been performed, and refinement steps.

% Inputs:
% sMerge         - struct containing data for STEM alignment from SPmerge01
% refineMaxSteps - Maximum number of refinement steps (optional). If this
%                  value is not specified, it will be set to 32 iterations.
%                  Set to 0 to perform only initial refinement.
flagPlot = 1;
flagReportProgress = 1;  % Set to true to see updates on console.
densityCutoff = 0.8;     % density cutoff for image boundaries (norm. to 1).
distStart = mean([size(sMerge.scanLines,1) ...
    size(sMerge.scanLines,2)])/16;    % Radius of # of scanlines used for initial alignment.
% initialRefineSteps = 8/2;   % Number of initial refinement iterations.
initialShiftMaximum = 1/4;  % Max number of pixels shifted per line for the
%                             initial alignment step.  This value should
%                             have a maximum of 1, but can be set lower
%                             to stabilize initial alignment.
refineInitialStep = 1/2; % Initial step size for main refinement, in pixels.
pixelsMovedThreshold = 0.1;% If number of pixels shifted (per image) is
%                          below this value, refinement will be halted.
stepSizeReduce = 1/2;  % When a scanline origin does not move,
%                        step size will be reduced by this factor.
flagPointOrder = 1;    % Use this flag to force origins to be ordered, i.e.
%                        disallow points from changing their order.
flagGlobalShift = 1*0;   % If this flag is true, a global phase correlation
%                        performed each primary iteration (This is meant to
%                        fix unit cell shifts and similar artifacts).
%                        This option is highly recommended!
flagGlobalShiftIncrease = 0; % If this option is true, the global scoring
%                              function is allowed to increase after global
%                              phase correlation step. (false is more stable)
minGlobalShift = 1;    % Global shifts only if shifts > this value (pixels)
densityDist = mean([size(sMerge.scanLines,1) ...
    size(sMerge.scanLines,2)])/32; % density mask edge threshold
% To generate a moving average along the scanline origins
% (make scanline steps more linear), use the settings below:
originWindowAverage = 1;  % Window sigma in px for smoothing scanline origins.
%                           Set this value to zero to not use window avg.
%                           This window is relative to linear steps.
originInitialAverage = mean([size(sMerge.scanLines,1) ...
    size(sMerge.scanLines,2)])/4/4;  % Window sigma in px for initial smoothing.
resetInitialAlignment = 0;   % Set this value to true to redo initial alignment.


% Outputs:
% sMerge - struct containing data for STEM alignment


% Default number of iterations if user does not provide values
if nargin == 1
    refineMaxSteps = 32;
    initialRefineSteps = 4*0;
elseif nargin == 2
    initialRefineSteps = 4*0;
end


% Make kernel for moving average of origins
if originInitialAverage > 0   % || originLinearFraction > 0
    if originInitialAverage > 0
        r = ceil(3*originInitialAverage);
        v = (-r:r)';
        KDEorigin = exp(-v.^2/(2*originInitialAverage^2));
    else
        KDEorigin = 1;
    end
    KDEnorm = 1./convn(ones(size(sMerge.scanOr)),KDEorigin,'same');
    %     indsOr = repmat((1:size(sMerge.scanOr,1))',...
    %         [2 sMerge.numImages]);
    basisOr = [ones(size(sMerge.scanLines,1),1) ...
        (1:size(sMerge.scanLines,1))'];
    scanOrLinear = zeros(size(sMerge.scanOr));
end


% If initial alignment has not been performed, do it.
if flagReportProgress == true
    reverseStr = ''; % initialize console message piece
end
if (~isfield(sMerge,'scanActive') || resetInitialAlignment == true) ...
        || ((initialRefineSteps > 0) && (nargin == 3))
    for aInit = 1:initialRefineSteps

        sMerge.scanActive = false(size(sMerge.scanLines,1),...
            sMerge.numImages);

        % Get starting scanlines for initial alignment
        indStart = zeros(sMerge.numImages,1);
        for a0 = 1:sMerge.numImages
            % Scan line direction and origins
            v = [-sMerge.scanDir(a0,2) sMerge.scanDir(a0,1)];
            or = sMerge.scanOr(:,1:2,a0);

            % Determine closest scanline origin from point-line distance
            c = -sum(sMerge.ref.*v);
            dist = abs(v(1)*or(:,1)+v(2)*or(:,2) + c) / norm(v);
            [~,indStart(a0)] = min(dist);
            sub = dist < distStart;
            sMerge.scanActive(sub,a0) = true;
        end

        % Rough initial alignment of scanline origins, to nearest pixel
        inds = 1:size(sMerge.scanLines,1);
        dxy = [0 0;
            1 0;
            -1 0;
            0 1;
            0 -1];
        score = zeros(size(dxy,1),1);
        for a0 = 1:sMerge.numImages
            % Determine which image to align to, based on orthogonality
            [~,indAlign] = min(abs(sum(repmat(sMerge.scanDir(a0,:), ...
                [sMerge.numImages 1]).* sMerge.scanDir,2)));

            % Generate alignment image
            if ~isfield(sMerge,'imageRef')
                sMerge = SPmakeImage(sMerge,indAlign,sMerge.scanActive(:,indAlign));
                imageAlign = sMerge.imageTransform(:,:,indAlign) ...
                    .* (sMerge.imageDensity(:,:,indAlign)>densityCutoff);
            else
                imageAlign = sMerge.imageRef;
            end

            %                 sub = sMerge.imageDensity(:,:,indAlign)>densityCutoff;
            %         imageAlign = sMerge.imageTransform(:,:,indAlign).*sub ...
            %             + (1-sub)*intensityMedian;


            %         figure(111)
            %         clf
            %         imagesc(imageAlign)
            %         axis equal off
            %         colormap(violetFire)
            %         drawnow


            % align origins
            xyStep = mean(sMerge.scanOr(2:end,:,a0)-sMerge.scanOr(1:(end-1),:,a0),1);
            indAligned = false(size(sMerge.scanLines,1),1);
            indAligned(indStart(a0)) = true;
            while all(indAligned) == 0
                % Determine scanline indices to check next
                v = bwmorph(indAligned,'dilate',1);
                v(indAligned) = false;
                indMove = inds(v);

                % currently active scanlines
                indsActive = inds(indAligned);

% %                 % If needed, get linear estimates
% %                 if weightInitialLinear > 0
% %                     A = [ones(length(indsActive),1) indsActive'];
% %                     beta = A \ sMerge.scanOr(indAligned,1:2,a0);
% %                 end

                % Align selected scanlines
                for a1 = 1:length(indMove)
                    % determine starting point from neighboring scanline
                    [~,minDistInd] = min(abs(indMove(a1)-indsActive));

                    % Step perpendicular to scanDir
                    xyOr = sMerge.scanOr(indsActive(minDistInd),1:2,a0) ...
                        + xyStep * (indMove(a1)-indsActive(minDistInd));

                    % Refine score by moving origin of this scanline
                    %   ============ my change ============
                    indsN = 1:size(sMerge.scanLines,2);
                    xInd = round(xyOr(1) + indsN*sMerge.scanDir(a0,1));
                    yInd = round(xyOr(2) + indsN*sMerge.scanDir(a0,2));
                    %   ============ my change ============
                    %                     % Prevent pixels from leaving image boundaries
                    %                     xInd = max(min(xInd,sMerge.imageSize(1)-1),1);
                    %                     yInd = max(min(yInd,sMerge.imageSize(2)-1),1);
                    for a2 = 1:size(dxy,1)
                        score(a2) = sum(abs(imageAlign(sub2ind(sMerge.imageSize,...
                            xInd + dxy(a2,1),yInd + dxy(a2,2))) ...
                            - sMerge.scanLines(indMove(a1),:,a0)));
                    end
                    [~,ind] = min(score);
                    sMerge.scanOr(indMove(a1),1:2,a0) = xyOr ...
                        + dxy(ind,1:2) * initialShiftMaximum;
                    indAligned(indMove(a1)) = true;
                end

                % Report progess if flag is set to true
                if flagReportProgress == true
                    comp = sum(indAligned) / numel(indAligned);
                    msg = sprintf(['Initial refinement ' ...
                        num2str(aInit) '/' num2str(initialRefineSteps) ...
                        ' of image ' ...
                        num2str(a0) '/' num2str(sMerge.numImages) ' is ' ...
                        num2str(round(100*comp))  ' percent complete']);
                    fprintf([reverseStr, msg]);
                    reverseStr = repmat(sprintf('\b'),1,length(msg));
                end
            end
        end

        % If required, compute moving average of origins using KDE.
        if originInitialAverage > 0  % || originLinearFraction > 0
            % Linear fit to scanlines
            for a0 = 1:sMerge.numImages
                ppx = basisOr \ sMerge.scanOr(:,1,a0);
                ppy = basisOr \ sMerge.scanOr(:,2,a0);
                scanOrLinear(:,1,a0) = basisOr*ppx;
                scanOrLinear(:,2,a0) = basisOr*ppy;
            end
            % Subtract linear fit
            sMerge.scanOr = sMerge.scanOr - scanOrLinear;
            % Moving average of scanlines using KDE
            sMerge.scanOr = convn(sMerge.scanOr,KDEorigin,'same') ...
                .* KDEnorm;
            % Add linear fit back into to origins, and/or linear weighting
            %             sMerge.scanOr = sMerge.scanOr *(1-originLinearFraction) ...
            %                 + scanOrLinear;
            sMerge.scanOr = sMerge.scanOr  + scanOrLinear;
        end
    end
end



% Make kernel for moving average of origins
if originWindowAverage > 0   % || originLinearFraction > 0
    if originWindowAverage > 0
        r = ceil(3*originWindowAverage);
        v = (-r:r)';
        KDEorigin = exp(-v.^2/(2*originWindowAverage^2));
    else
        KDEorigin = 1;
    end
    KDEnorm = 1./convn(ones(size(sMerge.scanOr)),KDEorigin,'same');
    %     indsOr = repmat((1:size(sMerge.scanOr,1))',...
    %         [2 sMerge.numImages]);
    basisOr = [ones(size(sMerge.scanLines,1),1) ...
        (1:size(sMerge.scanLines,1))'];
    scanOrLinear = zeros(size(sMerge.scanOr));
end



% Main alignment steps
if flagReportProgress == true
    msg = sprintf('Beginning primary refinement ...');
    fprintf([reverseStr, msg]);
    reverseStr = repmat(sprintf('\b'),1,length(msg));
end
scanOrStep = ones(size(sMerge.scanOr,1),size(sMerge.scanOr,3)) ...
    *refineInitialStep;
inds = 1:size(sMerge.scanLines,2);
dxy = [0 0;
    1 0;
    -1 0;
    0 1;
    0 -1];
score = zeros(size(dxy,1),1);
alignStep = 1;
sMerge.stats = zeros(refineMaxSteps+1,2);
indsLoop = 1:sMerge.numImages;
while alignStep <= refineMaxSteps
    % Reset pixels moved count
    pixelsMoved = 0;

    % Compute all images from current origins
    for a0 = indsLoop
        sMerge = SPmakeImage(sMerge,a0);
    end

    % Get mean absolute difference as a fraction of the mean scanline intensity.
    Idiff = mean(abs(sMerge.imageTransform ...
        - repmat(mean(sMerge.imageTransform,3),[1 1 sMerge.numImages])),3);
    meanAbsDiff = mean(Idiff(min(sMerge.imageDensity,[],3)>densityCutoff)) ...
        / mean(abs(sMerge.scanLines(:)));
    sMerge.stats(alignStep,1:2) = [alignStep-1 meanAbsDiff];

    % If required, check for global alignment of images
    if flagGlobalShift == true
        if flagReportProgress == true
            msg = sprintf('Checking global alignment ...');
            fprintf([reverseStr, msg]);
            reverseStr = repmat(sprintf('\b'),1,length(msg));
        end

        % save current origins, step size and score
        scanOrCurrent = sMerge.scanOr;
        scanOrStepCurrent = scanOrStep;
        meanAbsDiffCurrent = meanAbsDiff;

        % Align to windowed image 1 or imageRef
        intensityMedian = median(sMerge.scanLines(:));
        densityMask = sin((pi/2)*min(bwdist( ...
            sMerge.imageDensity(:,:,1)<densityCutoff)/densityDist,1)).^2;
        if ~isfield(sMerge,'imageRef')
            imageFFT1 = fft2(sMerge.imageTransform(:,:,1).*densityMask ...
                + (1-densityMask)*intensityMedian);
            vecAlign = 2:sMerge.numImages;
        else
            imageFFT1 = fft2(sMerge.imageRef.*densityMask ...
                + (1-densityMask)*intensityMedian);
            vecAlign = 1:sMerge.numImages;
        end

        % Align datasets 2 and higher to dataset 1, or align all images to imageRef
        for a0 = vecAlign
            % Simple phase correlation
            densityMask = sin((pi/2)*min(bwdist( ...
                sMerge.imageDensity(:,:,a0)<densityCutoff)/64,1)).^2;
            imageFFT2 = conj(fft2(sMerge.imageTransform(:,:,a0).*densityMask ...
                + (1-densityMask)*intensityMedian));
            phaseCorr = abs(ifft2(exp(1i*angle(imageFFT1.*imageFFT2))));

            % Get peak maximum
            [~,ind] = max(phaseCorr(:));
            [xInd,yInd] = ind2sub(sMerge.imageSize,ind);

            % Compute relative shifts.  Note that since matrix indices
            % start at 1, must be shifted by -1.
            dx = mod(xInd-1+sMerge.imageSize(1)/2,...
                sMerge.imageSize(1))-sMerge.imageSize(1)/2;
            dy = mod(yInd-1+sMerge.imageSize(2)/2,...
                sMerge.imageSize(2))-sMerge.imageSize(2)/2;

            % Only apply shift if it is larger than 2 pixels
            if abs(dx) + abs(dy) > minGlobalShift
                % apply global origin shift, if possible
                xNew = sMerge.scanOr(:,1,a0) + dx;
                yNew = sMerge.scanOr(:,2,a0) + dy;

                % Verify shifts are within image boundaries
                if min(xNew) >= 1 ...
                        && max(xNew) < sMerge.imageSize(1)-1 ...
                        && min(yNew) >= 1 ...
                        && max(yNew) < sMerge.imageSize(2)-1
                    sMerge.scanOr(:,1,a0) = xNew;
                    sMerge.scanOr(:,2,a0) = yNew;

                    % Recompute image with new origins
                    sMerge = SPmakeImage(sMerge,a0);

                    % Reset search values for this image
                    scanOrStep(:,a0) = refineInitialStep;
                end
            end
        end

        if flagGlobalShiftIncrease == false
            % Verify global shift did not make mean abs. diff. increase.
            Idiff = mean(abs(sMerge.imageTransform ...
                - repmat(mean(sMerge.imageTransform,3),[1 1 sMerge.numImages])),3);
            meanAbsDiffNew = mean(Idiff(min(sMerge.imageDensity,[],3)>densityCutoff)) ...
                / mean(abs(sMerge.scanLines(:)));

            if meanAbsDiffNew < meanAbsDiffCurrent
                % If global shift decreased mean absolute different, keep.
                sMerge.stats(alignStep,1:2) = [alignStep-1 meanAbsDiff];
            else
                % If global shift incresed mean abs. diff., return origins
                % and step sizes to previous values.
                sMerge.scanOr = scanOrCurrent;
                scanOrStep = scanOrStepCurrent;
            end

        end
    end

    % Refine each image in turn, against the sum of all other images
    for a0 = indsLoop
        % Generate alignment image, mean of all other scanline datasets,
        % unless user has specified a reference image.
        if ~isfield(sMerge,'imageRef')
            indsAlign = 1:sMerge.numImages;
            indsAlign(a0) = [];
            imageAlign = sum(sMerge.imageTransform(:,:,indsAlign) ...
                .* (sMerge.imageDensity(:,:,indsAlign)>densityCutoff),3);
            dens = sum(sMerge.imageDensity(:,:,indsAlign)>densityCutoff,3);
            sub = dens > 0;
            imageAlign(sub) = imageAlign(sub) ./ dens(sub);
            imageAlign(~sub) = mean(imageAlign(sub));  % Replace zeros with mean
        else
            imageAlign = sMerge.imageRef;
        end

        % If ordering is used as a condition, determine parametric positions
        if flagPointOrder == true
            % Use vector perpendicular to scan direction (negative 90 deg)
            n = [sMerge.scanDir(a0,2) -sMerge.scanDir(a0,1)];
            vParam = n(1)*sMerge.scanOr(:,1,a0) + n(2)*sMerge.scanOr(:,2,a0);
        end

        % Loop through each scanline and perform alignment
        for a1 = 1:size(sMerge.scanLines,1)
            % Refine score by moving the origin of this scanline
            orTest = repmat(sMerge.scanOr(a1,1:2,a0),[size(dxy,1) 1]) ...
                + dxy * scanOrStep(a1,a0);

            % If required, force ordering of points
            if flagPointOrder == true
                vTest = n(1)*orTest(:,1) + n(2)*orTest(:,2);
                if a1 == 1
                    vBound = [-Inf vParam(a1+1)];
                elseif a1 == size(sMerge.scanLines,1)
                    vBound = [vParam(a1-1) Inf];
                else
                    vBound = [vParam(a1-1) vParam(a1+1)];
                end
                for a2 = 1:size(dxy,1)
                    if vTest(a2) < vBound(1)
                        orTest(a2,:) = orTest(a2,:) + n*(vBound(1)-vTest(a2));
                    elseif vTest(a2) > vBound(2)
                        orTest(a2,:) = orTest(a2,:) + n*(vBound(2)-vTest(a2));
                    end
                end
            end

            % Loop through origin tests
            for a2 = 1:size(dxy,1)
                xInd = orTest(a2,1) + inds*sMerge.scanDir(a0,1);
                yInd = orTest(a2,2) + inds*sMerge.scanDir(a0,2);
                % Prevent pixels from leaving image boundaries
                xInd = max(min(xInd,sMerge.imageSize(1)-1),1);
                yInd = max(min(yInd,sMerge.imageSize(2)-1),1);
                % Bilinear coordinates
                xF = floor(xInd);
                yF = floor(yInd);
                score(a2) = calcScore(imageAlign,xF,yF,xInd-xF,yInd-yF,...
                    sMerge.scanLines(a1,:,a0));
            end
            % Note that if moving origin does not change score, dxy = (0,0)
            % will be selected (ind = 1).
            [~,ind] = min(score);
            if ind == 1
                % Reduce the step size for this origin
                scanOrStep(a1,a0) = scanOrStep(a1,a0) * stepSizeReduce;
            else
                pixelsMoved = pixelsMoved ...
                    + norm(orTest(ind,:)-sMerge.scanOr(a1,1:2,a0));
                sMerge.scanOr(a1,1:2,a0) = orTest(ind,:);
            end

            % Report progress if requested
            if flagReportProgress == true && mod(a1,16) == 0
                comp = (a1 / size(sMerge.scanLines,1) ...
                    + a0 - 1) / sMerge.numImages;
                msg = sprintf([ ...
                    'Mean Abs. Diff. = ' sprintf('%.04f',100*meanAbsDiff) ...
                    ' percent, ' ...
                    'iter. ' num2str(alignStep) ...
                    '/' num2str(refineMaxSteps) ', ' ...
                    num2str(round(100*comp))  ' percent complete, ' ...
                    num2str(round(pixelsMoved)) ' px moved']);
                fprintf([reverseStr, msg]);
                reverseStr = repmat(sprintf('\b'),1,length(msg));
            end
        end
    end

    % If required, compute moving average of origins using KDE.
    if originWindowAverage > 0 % || originLinearFraction > 0
        % Linear fit to scanlines
        for a0 = 1:sMerge.numImages
            ppx = basisOr \ sMerge.scanOr(:,1,a0);
            ppy = basisOr \ sMerge.scanOr(:,2,a0);
            scanOrLinear(:,1,a0) = basisOr*ppx;
            scanOrLinear(:,2,a0) = basisOr*ppy;
        end
        % Subtract linear fit
        sMerge.scanOr = sMerge.scanOr - scanOrLinear;
        % Moving average of scanlines using KDE
        sMerge.scanOr = convn(sMerge.scanOr,KDEorigin,'same') ...
            .* KDEnorm;
        % Add linear fit back into to origins, and/or linear weighting
        %         sMerge.scanOr = sMerge.scanOr *(1-originLinearFraction) ...
        %             + scanOrLinear;
        sMerge.scanOr = sMerge.scanOr + scanOrLinear;
    end

    % If pixels moved is below threshold, halt refinement
    if (pixelsMoved/sMerge.numImages) < pixelsMovedThreshold
        alignStep = refineMaxSteps + 1;
    else
        alignStep = alignStep + 1;
    end
end



% Remake images for plotting
if flagReportProgress == true
    msg = sprintf('Recomputing images and plotting ...');
    fprintf([reverseStr, msg]);
    reverseStr = repmat(sprintf('\b'),1,length(msg));
end
for a0 = indsLoop
    sMerge = SPmakeImage(sMerge,a0);
end


if flagPlot == 1
    imagePlot = sum(sMerge.imageTransform.*sMerge.imageDensity,3);
    dens = sum(sMerge.imageDensity,3);
    mask = dens>0;
    imagePlot(mask) = imagePlot(mask) ./ dens(mask);
    % Scale intensity of image
    mask = dens>0.5;
    imagePlot = imagePlot - mean(imagePlot(mask));
    imagePlot = imagePlot / sqrt(mean(imagePlot(mask).^2));


    % Plot results, image with scanline origins overlaid
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
    % plot(squeeze(sMerge.scanOr(1,2,:)),...
    %     squeeze(sMerge.scanOr(1,1,:)),...
    %     'linewidth',4,'color',[1 1 1])
    for a0 = 1:sMerge.numImages
        scatter(sMerge.scanOr(:,2,a0),sMerge.scanOr(:,1,a0),'marker','.',...
            'sizedata',25,'markeredgecolor',cvals(mod(a0-1,size(cvals,1))+1,:))
    end
    hold off
    axis equal off
    colormap(gray(256))
    set(gca,'position',[0 0 1 1])
    caxis([-3 3])  % units of image RMS

    % Get final stats
    % Idiff = mean(abs(sMerge.imageTransform ...
    %     - repmat(mean(sMerge.imageTransform,3),[1 1 sMerge.numImages])),3);
    % meanAbsDiff = mean(Idiff(min(sMerge.imageDensity,[],3)>densityCutoff)) ...
    %     / mean(sMerge.scanLines(:));
    % sMerge.stats(end,1:2) = [refineMaxSteps meanAbsDiff];
    Idiff = mean(abs(sMerge.imageTransform ...
        - repmat(mean(sMerge.imageTransform,3),[1 1 sMerge.numImages])),3);
    meanAbsDiff = mean(Idiff(min(sMerge.imageDensity,[],3)>densityCutoff)) ...
        / mean(abs(sMerge.scanLines(:)));
    sMerge.stats(alignStep,1:2) = [alignStep-1 meanAbsDiff];

    % Plot statistics
    if size(sMerge.stats,1) > 1
        figure(2)
        clf
        plot(sMerge.stats(:,1),sMerge.stats(:,2)*100,'linewidth',2,'color','r')
        xlabel('Iteration [Step Number]')
        ylabel('Mean Absolute Difference [%]')
    end
end

if flagReportProgress == true
    fprintf([reverseStr ' ']);
end
% toc
end

function [score] = calcScore(image,xF,yF,dx,dy,intMeas)
% imageSample = interp2(image,yF+dy,xF+dx,'linear');  % Interp2 is too slow
imageSample = ...
    image(sub2ind(size(image),xF,  yF))   .*(1-dx).*(1-dy) ...
    + image(sub2ind(size(image),xF+1,yF))   .*dx    .*(1-dy) ...
    + image(sub2ind(size(image),xF,  yF+1)) .*(1-dx).*dy ...
    + image(sub2ind(size(image),xF+1,yF+1)) .*dx    .*dy;
score = sum(abs(imageSample-intMeas));
end
