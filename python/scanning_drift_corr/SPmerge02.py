"""The file contains the SPmerge02 function
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.ndimage.morphology import binary_dilation

from scanning_drift_corr.SPmakeImage import SPmakeImage
from scanning_drift_corr.tools import distance_transform

def SPmerge02(sm, refineMaxSteps=None, initialRefineSteps=None,
              only_initial_refinemen=False, flagGlobalShift=True):
    # only_initial_refinemen, used for testing only initial refinement
    # should split SPmerge02 into two parts later

    flagPlot = True

    # Set to true to see updates on console.
    flagReportProgress = True

    # density cutoff for image boundaries (norm. to 1).
    densityCutoff = 0.8

    # Radius of # of scanlines used for initial alignment.
    distStart = np.mean(sm.scanLines.shape[1:]) / 16

    # Max number of pixels shifted per line for the
    # initial alignment step.  This value should
    # have a maximum of 1, but can be set lower
    # to stabilize initial alignment.
    initialShiftMaximum = 1/4

    # Initial step size for main refinement, in pixels.
    refineInitialStep = 1/2

    # If number of pixels shifted (per image) is
    # below this value, refinement will be halted.
    pixelsMovedThreshold = 0.1

    # When a scanline origin does not move,
    # step size will be reduced by this factor.
    stepSizeReduce = 1/2

    # Use this flag to force origins to be ordered, i.e.
    # disallow points from changing their order.
    flagPointOrder = True

    # If this flag is true, a global phase correlation
    # performed each primary iteration (This is meant to
    # fix unit cell shifts and similar artifacts).
    # This option is highly recommended!
    flagGlobalShift = False

    # If this option is true, the global scoring
    # function is allowed to increase after global
    # phase correlation step. (false is more stable)
    flagGlobalShiftIncrease = False

    # Global shifts only if shifts > this value (pixels)
    minGlobalShift = 1

    # density mask edge threshold
    # To generate a moving average along the scanline origins
    # (make scanline steps more linear), use the settings below:
    densityDist =  np.mean(sm.scanLines.shape[1:]) / 32

    # Window sigma in px for smoothing scanline origins.
    # Set this value to zero to not use window avg.
    # This window is relative to linear steps.
    originWindowAverage = 1

    # Window sigma in px for initial smoothing.
    originInitialAverage = np.mean(sm.scanLines.shape[1:]) / 16

    # Set this value to true to redo initial alignment.
    resetInitialAlignment = False

    # Default number of iterations if user does not provide values
    nargs = 3
    if refineMaxSteps is None:
        refineMaxSteps = 32
        nargs -= 1
    if initialRefineSteps is None:
        initialRefineSteps = 0
        nargs -= 1

    # Make kernel for moving average of origins
    if originInitialAverage > 0:
        KDEorigin, KDEnorm, basisOr, scanOrLinear = _get_kernel_moving_origins(sm, originInitialAverage)
    else:
        KDEorigin, KDEnorm, basisOr, scanOrLinear = 1, None, None, None

    doInitialRefine = ((sm.scanActive is None) | resetInitialAlignment |
                       (initialRefineSteps > 0)) & (nargs == 3)
    if doInitialRefine:
        print('Initial refinement ...')

        _initial_refinement(sm, initialRefineSteps, distStart,
                            densityCutoff, initialShiftMaximum,
                            originInitialAverage, KDEorigin, KDEnorm, basisOr, scanOrLinear)

    # later remove this
    if only_initial_refinemen:
        return sm

    # ==================================
    # split here when refactoring
    # ==================================


    # Make kernel for moving average of origins
    if originWindowAverage > 0:
        KDEorigin, KDEnorm, basisOr, scanOrLinear = _get_kernel_moving_origins(sm, originWindowAverage)
    else:
        KDEorigin, KDEnorm, basisOr, scanOrLinear = 1, None, None, None


    # Main alignment steps
    print('Beginning primary refinement ...')


    scanOrStep = np.ones((sm.numImages, sm.nr)) * refineInitialStep
    dxy = np.array([[0,1,-1,0,0], [0,0,0,1,-1]])
    alignStep = 1
    sm.stats = np.zeros((refineMaxSteps+1, 2))


    while alignStep <= refineMaxSteps:
        # Reset pixels moved count
        pixelsMoved = 0

        # Compute all images from current origins
        for k in range(sm.numImages):
            # sMerge changed!
            sm = SPmakeImage(sm, k)

        # Get mean absolute difference as a fraction of the mean scanline intensity.
        imgT_mean = sm.imageTransform.mean(axis=0)
        Idiff = np.abs(sm.imageTransform - imgT_mean).mean(axis=0)
        dmask = sm.imageDensity.min(axis=0) > densityCutoff
        img_mean = np.abs(sm.scanLines).mean()
        meanAbsDiff = Idiff[dmask].mean() / img_mean
        sm.stats[alignStep-1, :] = np.array([alignStep-1, meanAbsDiff])

        # If required, check for global alignment of images
        if flagGlobalShift:
            print('Checking global alignment ...')
            _global_phase_correlation(sm, scanOrStep, meanAbsDiff, densityCutoff,
                                      densityDist,
                                      flagGlobalShiftIncrease,
                                      minGlobalShift, refineInitialStep, alignStep)

        # Refine each image in turn, against the sum of all other images
        for k in range(sm.numImages):
            # Generate alignment image, mean of all other scanline datasets,
            # unless user has specified a reference image.
            if sm.imageRef is None:
                indsAlign = np.arange(sm.numImages, dtype=int)
                indsAlign = indsAlign[indsAlign != k]

                dens_cut = sm.imageDensity[indsAlign, ...] > densityCutoff
                imageAlign = (sm.imageTransform[indsAlign, ...] * dens_cut).sum(axis=0)
                dens = dens_cut.sum(axis=0)
                sub = dens > 0
                imageAlign[sub] = imageAlign[sub] / dens[sub]
                imageAlign[~sub] = np.mean(imageAlign[sub])
            else:
                imageAlign = sm.imageRef


            # If ordering is used as a condition, determine parametric positions
            if flagPointOrder:
                # Use vector perpendicular to scan direction (negative 90 deg)
                nn = np.array([sm.scanDir[k, 1], -sm.scanDir[k, 0]])
                vParam = nn[0]*sm.scanOr[k, 0, :] + nn[1]*sm.scanOr[k, 1, :]

            # Loop through each scanline and perform alignment
            for m in range(sm.nr):
                # Refine score by moving the origin of this scanline
                orTest = sm.scanOr[k, :, m][:, None] + dxy*scanOrStep[k, m]

                # If required, force ordering of points
                if flagPointOrder:
                    vTest = nn[0]*orTest[0, :] + nn[1]*orTest[1, :]

                    if m == 0:
                        # no lower bound?
                        vBound = np.array([-np.inf, vParam[m+1]])
                    elif m == sm.nr-1:
                        # no upper bound?
                        vBound = np.array([vParam[m-1], np.inf])
                    else:
                        vBound = np.array([vParam[m-1], vParam[m+1]])

                    # check out of bound entries?
                    for p in range(dxy.shape[1]):
                        if vTest[p] < vBound[0]:
                            orTest[:, p] += nn*(vBound[0]-vTest[p])
                        elif vTest[p] > vBound[1]:
                            orTest[:, p] += nn*(vBound[1]-vTest[p])

                # Loop through origin tests
                inds = np.arange(1, sm.nc+1)
                score = np.zeros(dxy.shape[1])
                for p in range(dxy.shape[1]):
                    xInd = orTest[0, p] + inds*sm.scanDir[k, 0]
                    yInd = orTest[1, p] + inds*sm.scanDir[k, 1]

                    # Prevent pixels from leaving image boundaries
                    xInd = np.clip(xInd, 0, sm.imageSize[0]-2).ravel()
                    yInd = np.clip(yInd, 0, sm.imageSize[1]-2).ravel()

                    # Bilinear coordinates
                    xF = np.floor(xInd).astype(int)
                    yF = np.floor(yInd).astype(int)
                    dx = xInd - xF
                    dy = yInd - yF

                    # scanLines indices switched
                    score[p] = calcScore(imageAlign, xF, yF, dx, dy,
                                         sm.scanLines[k, m, :])

                # Note that if moving origin does not change score, dxy = (0,0)
                # will be selected (ind = 0).
                ind = np.argmin(score)
                if ind == 0:
                    # Reduce the step size for this origin
                    scanOrStep[k, m] *= stepSizeReduce
                else:
                    pshift = np.linalg.norm(orTest[:,ind] - sm.scanOr[k, :, m])
                    pixelsMoved += pshift
                    # change sMerge!
                    sm.scanOr[k, :, m] = orTest[:,ind]

                #TODO add progress bar tqdm later



        # If required, compute moving average of origins using KDE.
        if originWindowAverage > 0:
            _compute_KDE_moving_average(sm, KDEorigin, KDEnorm, basisOr, scanOrLinear)

        # If pixels moved is below threshold, halt refinement
        if (pixelsMoved/sm.numImages) < pixelsMovedThreshold:
            alignStep = refineMaxSteps + 1
        else:
            alignStep += 1

    # Remake images for plotting
    print('Recomputing images and plotting ...')
    for k in range(sm.numImages):
        # sMerge changed!
        sm = SPmakeImage(sm, k)

    # Get final stats (instead of just before plotting)
    # Get mean absolute difference as a fraction of the mean scanline intensity.
    imgT_mean = sm.imageTransform.mean(axis=0)
    Idiff = np.abs(sm.imageTransform - imgT_mean).mean(axis=0)
    dmask = sm.imageDensity.min(axis=0) > densityCutoff
    img_mean = np.abs(sm.scanLines).mean()
    meanAbsDiff = Idiff[dmask].mean() / img_mean
    sm.stats[alignStep-1, :] = np.array([alignStep-1, meanAbsDiff])

    if flagPlot:
        _plot(sm)


    return sm


def calcScore(image, xF, yF, dx, dy, intMeas):

    imgsz = image.shape

    rind1 = np.ravel_multi_index((xF, yF), imgsz)
    rind2 = np.ravel_multi_index((xF+1, yF), imgsz)
    rind3 = np.ravel_multi_index((xF, yF+1), imgsz)
    rind4 = np.ravel_multi_index((xF+1, yF+1), imgsz)

    int1 = image.ravel()[rind1] * (1-dx) * (1-dy)
    int2 = image.ravel()[rind2] * dx * (1-dy)
    int3 = image.ravel()[rind3] * (1-dx) * dy
    int4 = image.ravel()[rind4] * dx * dy

    imageSample = int1 + int2 + int3 + int4

    score = np.abs(imageSample - intMeas).sum()

    return score

def _get_kernel_moving_origins(sm, originAverage):
    # Make kernel for moving average of origins
    r = np.ceil(3*originAverage)
    v = np.arange(-r, r+1)
    KDEorigin = np.exp(-v**2/(2*originAverage**2))

    KDEnorm = 1 / convolve(np.ones(sm.scanOr.shape), KDEorigin[:, None, None].T, 'same')

    # need to offset 1 here??
    basisOr = np.vstack([np.zeros(sm.nr), np.arange(0, sm.nr)]) + 1

    scanOrLinear = np.zeros(sm.scanOr.shape)

    return KDEorigin, KDEnorm, basisOr, scanOrLinear

def _compute_KDE_moving_average(sm, KDEorigin, KDEnorm, basisOr, scanOrLinear):

    # Linear fit to scanlines
    for k in range(sm.numImages):
        # need to offset 1 here for scanOr?
        ppx, *_ = np.linalg.lstsq(basisOr.T, sm.scanOr[k, 0, :]+1, rcond=None)
        ppy, *_ = np.linalg.lstsq(basisOr.T, sm.scanOr[k, 1, :]+1, rcond=None)
        scanOrLinear[k, 0, :] = basisOr.T @ ppx
        scanOrLinear[k, 1, :] = basisOr.T @ ppy

    # Subtract linear fit
    sm.scanOr -= scanOrLinear

    # Moving average of scanlines using KDE
    sm.scanOr = convolve(sm.scanOr, KDEorigin[:, None, None].T, 'same') * KDEnorm

    # Add linear fit back into to origins, and/or linear weighting
    sm.scanOr += scanOrLinear

def _initial_refinement(sm, initialRefineSteps, distStart,
                        densityCutoff, initialShiftMaximum,
                        originInitialAverage, KDEorigin, KDEnorm, basisOr, scanOrLinear):

    for _ in range(initialRefineSteps):
        sm.scanActive = np.zeros((sm.numImages, sm.nr), dtype=bool)

        indStart = _get_starting_scanline(sm, distStart)

        # Rough initial alignment of scanline origins, to nearest pixel
        inds = np.arange(sm.nr)
        dxy = np.array([[0,1,-1,0,0], [0,0,0,1,-1]])
        score = np.zeros(dxy.shape[1])
        for k in range(sm.numImages):
            # Determine which image to align to, based on orthogonality
            ortho = (sm.scanDir[k, :] * sm.scanDir).sum(axis=1)
            indAlign = np.argmin(np.abs(ortho))

            if sm.imageRef is None:
                # sMerge changed!
                sm = SPmakeImage(sm, indAlign, sm.scanActive[indAlign, :])

                imageAlign = sm.imageTransform[indAlign, ...] * (sm.imageDensity[indAlign, ...]>densityCutoff)

            else:
                imageAlign = sm.imageRef

            # align origins
            dOr = sm.scanOr[k, :, 1:] - sm.scanOr[k, :, :-1]
            xyStep = np.mean(dOr, axis=1)
            
            indAligned = np.zeros(sm.nr, dtype=bool)
            indAligned[int(indStart[k])] = True


            while not indAligned.all():

                # Align selected scanlines
                _align_selected_scanline(sm, k, inds, indAligned, xyStep, dxy, score, imageAlign, initialShiftMaximum)

        # If required, compute moving average of origins using KDE.
        if originInitialAverage > 0:
            _compute_KDE_moving_average(sm, KDEorigin, KDEnorm, basisOr, scanOrLinear)



def _get_starting_scanline(sm, distStart):
    # Get starting scanlines for initial alignment
    indStart = np.zeros(sm.numImages)
    for k in range(sm.numImages):
        # Scan line direction and origins
        v = np.array([-sm.scanDir[k, 1], sm.scanDir[k, 0]])
        or_ = sm.scanOr[k, ...]

        # Determine closest scanline origin from point-line distance
        c = -np.sum(sm.ref*v)
        dist = np.abs(v[0]*or_[0,:] + v[1]*or_[1,:] + c) / np.linalg.norm(v)
        indStart[k] = np.argmin(dist)
        sub = dist < distStart
        sm.scanActive[k, sub] = True

    return indStart




def _align_selected_scanline(sm, k, inds, indAligned, xyStep, dxy, score, imageAlign, initialShiftMaximum):
    # Determine scanline indices to check next
    v = binary_dilation(indAligned)
    v[indAligned] = False
    indMove = inds[v]

    # currently active scanlines
    indsActive = inds[indAligned]

    t = np.arange(1, sm.nc+1)
    for m in range(indMove.size):
        # determine starting point from neighboring scanline
        minDistInd = np.argmin(np.abs(indMove[m] - indsActive))

        # Step perpendicular to scanDir
        indMin = indsActive[minDistInd]
        xyOr = sm.scanOr[k, :, indMin] + xyStep * (indMove[m] - indMin)

        # Refine score by moving origin of this scanline
        xInd = np.floor(xyOr[0] + t*sm.scanDir[k, 0] + 0.5).astype(int)
        yInd = np.floor(xyOr[1] + t*sm.scanDir[k, 1] + 0.5).astype(int)

        # Prevent pixels from leaving image boundaries (here mainly for strange image?)
        # I think better to check?
        xInd = np.clip(xInd, 0, sm.imageSize[0]-2).ravel()
        yInd = np.clip(yInd, 0, sm.imageSize[1]-2).ravel()

        for n in range(dxy.shape[1]):
            nxInd = xInd + dxy[0, n]
            nyInd = yInd + dxy[1, n]

            # Prevent pixels from leaving image boundaries (here mainly for strange image?)
            # I think better to check?
            nxInd = np.clip(nxInd, 0, sm.imageSize[0]-2).ravel()
            nyInd = np.clip(nyInd, 0, sm.imageSize[1]-2).ravel()
            rInd = np.ravel_multi_index((nxInd, nyInd), sm.imageSize)

            # scanLines indices switched to match row
            score[n] = np.abs(imageAlign.ravel()[rInd] - sm.scanLines[k, indMove[m], :]).sum()

        ind = np.argmin(score)

        # change sMerge!
        sm.scanOr[k, :, indMove[m]] = xyOr + dxy[:, ind]*initialShiftMaximum
        indAligned[indMove[m]] = True

        #TODO add progress bar tqdm later

def _global_phase_correlation(sm, scanOrStep, meanAbsDiff, densityCutoff, densityDist,
                              flagGlobalShiftIncrease,
                              minGlobalShift, refineInitialStep, alignStep):

    # save current origins, step size and score
    scanOrCurrent = sm.scanOr.copy();
    scanOrStepCurrent = scanOrStep.copy();
    meanAbsDiffCurrent = meanAbsDiff.copy();

    # Align to windowed image 0 or imageRef
    intensityMedian = np.median(sm.scanLines)
    cut = sm.imageDensity[0, ...] < densityCutoff
    min_d = np.minimum(distance_transform(cut) / densityDist, 1)
    densityMask = np.sin(min_d * np.pi/2)**2

    if sm.imageRef is None:
        smooth = sm.imageTransform[0,...]*densityMask + (1-densityMask)*intensityMedian
        imageFFT1 = np.fft.fft2(smooth)
        vecAlign = range(1, sm.numImages)
    else:
        smooth = sm.imageRef*densityMask + (1-densityMask)*intensityMedian
        imageFFT1 = np.fft.fft2(smooth)
        vecAlign = range(sm.numImages)

    # Align datasets 1 and higher to dataset 0, or align all images to imageRef
    for k in vecAlign:
        # Simple phase correlation
        cut = sm.imageDensity[k, ...] < densityCutoff
        min_d = np.minimum(distance_transform(cut) / 64, 1)
        densityMask = np.sin(min_d * np.pi/2)**2

        smooth = sm.imageTransform[k,...]*densityMask + (1-densityMask)*intensityMedian
        imageFFT2 = np.fft.fft2(smooth).conj()

        phase = np.angle(imageFFT1*imageFFT2)
        phaseCorr = np.abs(np.fft.ifft2(np.exp(1j*phase)))

        # Get peak maximum
        xInd, yInd = np.unravel_index(phaseCorr.argmax(), phaseCorr.shape)

        # Compute relative shifts. No -1 shift needed here.
        nr, nc = sm.imageSize
        dx = (xInd + nr/2) % nr - nr/2
        dy = (yInd + nc/2) % nc - nc/2

        # Only apply shift if it is larger than 2 pixels
        if (abs(dx) + abs(dy)) > minGlobalShift:
            # apply global origin shift, if possible
            xNew = sm.scanOr[k, 0, :] + dx
            yNew = sm.scanOr[k, 1, :] + dy

            # Verify shifts are within image boundaries
            withinBoundary = (xNew.min() >= 0) & (xNew.max() < nr-2) &\
                             (yNew.min() >= 0) & (yNew.max() < nc-2)
            if withinBoundary:
                # sMerge changed!
                sm.scanOr[k, 0, :] = xNew
                sm.scanOr[k, 1, :] = yNew

                # Recompute image with new origins
                # sMerge changed!
                sm = SPmakeImage(sm, k)

                # Reset search values for this image
                scanOrStep[k, :] = refineInitialStep

        if not flagGlobalShiftIncrease:
            # Verify global shift did not make mean abs. diff. increase.
            imgT_mean = sm.imageTransform.mean(axis=0)
            Idiff = np.abs(sm.imageTransform - imgT_mean).mean(axis=0)
            dmask = sm.imageDensity.min(axis=0) > densityCutoff
            img_mean = np.abs(sm.scanLines).mean()
            meanAbsDiffNew = Idiff[dmask].mean() / img_mean

            # sMerge changed!
            if meanAbsDiffNew < meanAbsDiffCurrent:
                # If global shift decreased mean absolute different, keep.
                sm.stats[alignStep-1, :] = np.array([alignStep-1, meanAbsDiff])
            else:
                # If global shift incresed mean abs. diff., return origins
                # and step sizes to previous values.
                sm.scanOr = scanOrCurrent
                scanOrStep = scanOrStepCurrent


def _plot(sm):
    imagePlot = (sm.imageTransform*sm.imageDensity).sum(axis=0)
    dens = sm.imageDensity.sum(axis=0)
    mask = dens > 0
    imagePlot[mask] /= dens[mask]

    # Scale intensity of image
    mask = dens > 0.5
    imagePlot -= imagePlot[mask].mean()
    imagePlot /= np.sqrt(np.mean(imagePlot[mask]**2))

    fig, ax = plt.subplots()
    ax.matshow(imagePlot, cmap='gray')

    # RGB colours
    cvals = np.array([[1, 0, 0],
                      [0, 0.7, 0],
                      [0, 0.6, 1],
                      [1, 0.7, 0],
                      [1, 0, 1],
                      [0, 0, 1]])

    # put origins on plot
    for k in range(sm.numImages):
        x = sm.scanOr[k, 1, :]
        y = sm.scanOr[k, 0, :]
        c = cvals[k % cvals.shape[0], :]

        ax.plot(x, y, marker='.', markersize=12, linestyle='None', color=c)

    # Plot statistics
    if sm.stats.shape[0] > 1:
        fig, ax = plt.subplots()
        ax.plot(sm.stats[:, 0], sm.stats[:, 1]*100, color='red', linewidth=2)
        ax.set_xlabel('Iteration [Step Number]')
        ax.set_ylabel('Mean Absolute Difference [%]')

    return
