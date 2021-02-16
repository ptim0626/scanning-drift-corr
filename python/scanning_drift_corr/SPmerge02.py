"""The file contains the SPmerge02 function
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from tqdm import tqdm

from scanning_drift_corr.SPmakeImage import SPmakeImage
from scanning_drift_corr.SPmerge02_initial import SPmerge02_initial
from scanning_drift_corr.SPmerge02_final import SPmerge02_final
from scanning_drift_corr.SPmerge02_phase_correlation import _globbal_phase_correlation

def SPmerge02(sm, refineMaxSteps=None, initialRefineSteps=None, **kwargs):
    """Refinement function for scanning probe image

    Parameters
    ----------
    sm : sMerge object
        the sMerge object contains all the data.
    refineMaxSteps : int, optional
        maximum number of refinement steps. Default to None, set to 32.
    initialRefineSteps : int, optional
        number of initial alignment steps. Default to None, set to 8 if it has
        not been performed, or set to 0 if it has been performed

    ------------------ For initial alignment ------------------
    densityCutoff : float, optional
        density cutoff for image boundaries (norm. to 1). Default to 0.8.
    distStart : float, optional
        radius of # of scanlines used for initial alignment. Default to
        mean of raw data divided by 16.
    initialShiftMaximum : float, optional
        maximum number of pixels shifted per line for the initial alignment
        step. This value should have a maximum of 1, but can be set lower
        to stabilize initial alignment. Default to 0.25.
    originInitialAverage : float, optional
        window sigma in px for initial smoothing. Default to mean of raw data
        divided by 16.

    ------------------ For final alignment ------------------
    refineInitialStep : float, optional
        initial step size for final refinement, in pixels. Default to 0.5.
    pixelsMovedThreshold : float, optional
        if number of pixels shifted (per image) is below this value,
        refinement will be halted. Default to 0.1.
    stepSizeReduce : float, optional
        when a scanline origin does not move, step size will be reduced by
        this factor. Default to 0.5.
    flagPointOrder : bool, optional
        use this flag to force origins to be ordered, i.e. disallow points
        from changing their order. Default to True.
    originWindowAverage : float, optional
        window sigma in px for smoothing scanline origins. Set this value to
        zero to not use window avg. This window is relative to linear steps.
        Default to 1.
    parallel : bool, optional
        whether to parallelise the alignment for scan lines.
        The default is True.

    ------------------ For global phase correlation ------------------
    flagGlobalShift : bool, optional
        if this flag is true, a global phase correlation, performed each
        final iteration (This is meant to fix unit cell shifts and similar
        artifacts). This option is highly recommended! Default to False.
    flagGlobalShiftIncrease : bool, optional
        if this option is true, the global scoring function is allowed to
        increase after global phase correlation step. (false is more stable)
        Default to False.
    minGlobalShift : float, optional
        global shifts only if shifts > this value (pixels). Default to 1.
    densityDist : float, optional
         density mask edge threshold. To generate a moving average along the
         scanline origins (make scanline steps more linear). Default to mean
         of raw data divided by 32.

    ------------------ General behaviour ------------------
    flagRemakeImage : bool, optional
        whether to recompute the image. Default to True.
    flagReportProgress : bool, optional
        whether to show progress bars or not. Default to True.
    flagPlot : bool
        to plot the aligned images of not. Default to True.
    """

    # ignore unknown input arguments
    _args_list = ['densityCutoff', 'distStart', 'initialShiftMaximum',
                  'originInitialAverage', 'refineInitialStep',
                  'pixelsMovedThreshold', 'stepSizeReduce', 'flagPointOrder',
                  'originWindowAverage', 'flagGlobalShift', 'parallel',
                  'flagGlobalShiftIncrease', 'minGlobalShift', 'densityDist',
                  'flagRemakeImage', 'flagPlot', 'flagReportProgress']
    for key in kwargs.keys():
        if key not in _args_list:
            msg = "The argument '{}' is not recognised, and it is ignored."
            warnings.warn(msg.format(key), RuntimeWarning)

    # if number of final alignment not provided, set to 32
    if refineMaxSteps is None:
        refineMaxSteps = 32

    # if number of initial alignment not provided, set to 8 if the scanActive
    # attribute is None (i.e. no initial alignment has been performed, so do
    # it), else skip initial alignment
    if initialRefineSteps is None:
        if sm.scanActive is None:
            initialRefineSteps = 8
        else:
            initialRefineSteps = 0


    # set default values or from input arguments
    meanScanLines = np.mean(sm.scanLines.shape[1:])

    # for initial alignment
    densityCutoff = kwargs.get('densityCutoff', 0.8)
    distStart = kwargs.get('distStart', meanScanLines/16)
    initialShiftMaximum = kwargs.get('initialShiftMaximum', 1/4)
    originInitialAverage = kwargs.get('originInitialAverage', meanScanLines/16)

    # for final alignment
    refineInitialStep = kwargs.get('refineInitialStep', 1/2)
    pixelsMovedThreshold = kwargs.get('pixelsMovedThreshold', 0.1)
    stepSizeReduce = kwargs.get('stepSizeReduce', 1/2)
    flagPointOrder = kwargs.get('flagPointOrder', True)
    originWindowAverage = kwargs.get('originWindowAverage', 1)
    parallel = kwargs.get('parallel', True)

    # for global phase correlation
    flagGlobalShift = kwargs.get('flagGlobalShift', False)
    flagGlobalShiftIncrease = kwargs.get('flagGlobalShiftIncrease', False)
    minGlobalShift = kwargs.get('minGlobalShift', 1)
    densityDist = kwargs.get('densityDist', meanScanLines/32)

    # general use
    flagRemakeImage = kwargs.get('flagRemakeImage', True)
    flagPlot = kwargs.get('flagPlot', True)
    flagReportProgress = kwargs.get('flagReportProgress', True)

    # if required, perform initial alignment
    if initialRefineSteps > 0:
        for _ in tqdm(range(initialRefineSteps), desc='Initial refinement',
                      leave=False, disable=not flagReportProgress):
            SPmerge02_initial(sm, densityCutoff=densityCutoff,
                              distStart=distStart,
                              initialShiftMaximum=initialShiftMaximum)

            # If required, compute moving average of origins using KDE.
            if originInitialAverage > 0:
                _kernel_on_origin(sm, originInitialAverage)

    # Main alignment steps
    # initialisation
    scanOrStep = np.ones((sm.numImages, sm.nr)) * refineInitialStep
    alignStep = 1
    sm.stats = np.zeros((refineMaxSteps+1, 2))

    # create progress bar for the while loop
    pbar = tqdm(total=refineMaxSteps, desc='Final refinement', leave=False,
                disable=not flagReportProgress)

    while alignStep <= refineMaxSteps:
        # Compute all images from current origins
        for k in range(sm.numImages):
            sm = SPmakeImage(sm, k)

        # Get mean absolute difference as a fraction of the mean scanline
        # intensity.
        meanAbsDiff = _fraction_MD(sm, densityCutoff)
        sm.stats[alignStep-1, :] = np.array([alignStep-1, meanAbsDiff])

        # If required, check for global alignment of images
        if flagGlobalShift:
            pbar.set_description('Checking global alignment..')
            _globbal_phase_correlation(sm, scanOrStep, meanAbsDiff,
                                       densityCutoff, densityDist,
                                       flagGlobalShiftIncrease,
                                       minGlobalShift, refineInitialStep,
                                       alignStep, flagReportProgress)
           # restore the progress bar msg
            pbar.set_description('Final refinement')

        # final alignment
        stopRefine = SPmerge02_final(sm, scanOrStep,
                                     densityCutoff=densityCutoff,
                                     pixelsMovedThreshold=pixelsMovedThreshold,
                                     stepSizeReduce=stepSizeReduce,
                                     flagPointOrder=flagPointOrder,
                                     parallel=parallel)

        # If required, compute moving average of origins using KDE.
        if originWindowAverage > 0:
            _kernel_on_origin(sm, originWindowAverage)

        # If pixels moved is below threshold, halt refinement
        if stopRefine:
            alignStep = refineMaxSteps + 1
        else:
            alignStep += 1

        # update progress
        pbar.update(1)


    # close the progress bar of the while loop
    pbar.close()

    # Remake images for plotting
    if flagRemakeImage:
        for k in (tqdm(range(sm.numImages), desc='Recomputing images',
                       leave=False) if flagReportProgress else
                  range(sm.numImages)):
            sm = SPmakeImage(sm, k)

    # Get final stats
    meanAbsDiff = _fraction_MD(sm, densityCutoff)
    sm.stats[alignStep-1, :] = np.array([alignStep-1, meanAbsDiff])

    if flagPlot:
        _plot(sm)

    return sm

def _kernel_on_origin(sm, originAverage):
    """Make kernel for moving average of origins
    """

    r = np.ceil(3*originAverage)
    v = np.arange(-r, r+1)
    KDEorigin = np.exp(-v**2/(2*originAverage**2))

    KDEnorm = 1 / convolve(np.ones(sm.scanOr.shape),
                           KDEorigin[:, None, None].T, 'same')

    # need to offset 1 here??
    basisOr = np.vstack([np.zeros(sm.nr), np.arange(0, sm.nr)]) + 1

    scanOrLinear = np.zeros(sm.scanOr.shape)
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
    sm.scanOr = convolve(sm.scanOr,
                         KDEorigin[:, None, None].T, 'same') * KDEnorm

    # Add linear fit back into to origins, and/or linear weighting
    sm.scanOr += scanOrLinear

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
