"""The file contains the _globbal_phase_correlation function
"""

import numpy as np

from scanning_drift_corr.SPmakeImage import SPmakeImage
from scanning_drift_corr.tools import distance_transform


def _globbal_phase_correlation(sm, scanOrStep, meanAbsDiff, densityCutoff,
                               densityDist,flagGlobalShiftIncrease,
                               minGlobalShift, refineInitialStep, alignStep,
                               flagReportProgress):
    """to prevent unit cell hopping
    """

    # save current origins, step size and score
    scanOrCurrent = sm.scanOr.copy()
    scanOrStepCurrent = scanOrStep.copy()
    meanAbsDiffCurrent = meanAbsDiff.copy()

    # Align to windowed image 0 or imageRef
    smooth, imageFFT1, vecAlign = _get_ref(sm, densityCutoff, densityDist)

    # Align datasets 1 and higher to dataset 0, or align all images to imageRef
    for k in vecAlign:
        # simple phase correlation
        phaseCorr = _phase_correlation(sm, k, densityCutoff, imageFFT1)

        # Get peak maximum
        xInd, yInd = np.unravel_index(phaseCorr.argmax(), phaseCorr.shape)

        # Compute relative shifts
        nr, nc = sm.imageSize
        dx = (xInd + nr/2) % nr - nr/2
        dy = (yInd + nc/2) % nc - nc/2

        # Only apply shift if it is larger than 2 pixels (dx+dy)
        if (abs(dx) + abs(dy)) > minGlobalShift:
            shiftApplied = _apply_shift(sm, k, dx, dy)

            # Reset search values for this image if it is globally shifted
            if shiftApplied:
                scanOrStep[k, :] = refineInitialStep

        if not flagGlobalShiftIncrease:
            # Verify global shift did not make mean abs. diff. increase.
            meanAbsDiffNew = _fraction_MD(sm, densityCutoff)

            if meanAbsDiffNew < meanAbsDiffCurrent:
                # If global shift decreased mean absolute different, keep.
                sm.stats[alignStep-1, :] = np.array([alignStep-1, meanAbsDiff])
            else:
                # If global shift incresed mean abs. diff., return origins
                # and step sizes to previous values.
                sm.scanOr = scanOrCurrent
                scanOrStep = scanOrStepCurrent


def _get_ref(sm, densityCutoff, densityDist):
    # Align to windowed image 0 or imageRef
    intensityMedian = np.median(sm.scanLines)
    cut = sm.imageDensity[0, ...] < densityCutoff
    min_d = np.minimum(distance_transform(cut) / densityDist, 1)
    densityMask = np.sin(min_d * np.pi/2)**2

    if sm.imageRef is None:
        smooth = sm.imageTransform[0,...]*densityMask +\
            (1-densityMask)*intensityMedian
        imageFFT1 = np.fft.fft2(smooth)
        vecAlign = range(1, sm.numImages)
    else:
        smooth = sm.imageRef*densityMask + (1-densityMask)*intensityMedian
        imageFFT1 = np.fft.fft2(smooth)
        vecAlign = range(sm.numImages)

    return smooth, imageFFT1, vecAlign


def _phase_correlation(sm, k, densityCutoff, imageFFT1):
    """correlate the phase of current image with reference image
    """

    # Simple phase correlation
    intensityMedian = np.median(sm.scanLines)
    cut = sm.imageDensity[k, ...] < densityCutoff
    min_d = np.minimum(distance_transform(cut) / 64, 1)
    densityMask = np.sin(min_d * np.pi/2)**2

    smooth = sm.imageTransform[k,...]*densityMask +\
        (1-densityMask)*intensityMedian
    imageFFT2 = np.fft.fft2(smooth).conj()

    phase = np.angle(imageFFT1*imageFFT2)
    phaseCorr = np.abs(np.fft.ifft2(np.exp(1j*phase)))

    return phaseCorr

def _apply_shift(sm, k, dx, dy):
    """apply the shift dx and dy, check if within image after global shift
    """

    # apply global origin shift, if possible
    xNew = sm.scanOr[k, 0, :] + dx
    yNew = sm.scanOr[k, 1, :] + dy

    # Verify shifts are within image boundaries
    nr, nc = sm.imageSize
    withinBoundary = (xNew.min() >= 0) & (xNew.max() < nr-2) &\
                     (yNew.min() >= 0) & (yNew.max() < nc-2)
    if withinBoundary:
        sm.scanOr[k, 0, :] = xNew
        sm.scanOr[k, 1, :] = yNew

        # Recompute image with new origins
        sm = SPmakeImage(sm, k)

    return withinBoundary

def _fraction_MD(sm, densityCutoff):
    """Get mean absolute difference as a fraction of the mean scanline
    intensity.
    """

    imgT_mean = sm.imageTransform.mean(axis=0)
    Idiff = np.abs(sm.imageTransform - imgT_mean).mean(axis=0)
    dmask = sm.imageDensity.min(axis=0) > densityCutoff
    img_mean = np.abs(sm.scanLines).mean()
    meanAbsDiff = Idiff[dmask].mean() / img_mean

    return meanAbsDiff
