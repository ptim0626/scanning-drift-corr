"""The file contains the SPmerge01linear function
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scanning_drift_corr.sMerge import sMerge
from scanning_drift_corr.SPmakeImage import SPmakeImage

def SPmerge01linear(scanAngles, *images, **kwargs):
    """
    New version of SPmerge01.m - This script now searches over linear drift
    vectors, aligned to first two images.  This search is performed twice.

    Merge multiple scanning probe images.  Assume scan origin is upper left
    corner of the image, and that the scan direction for zero degrees is
    horizontal (along MATLAB columns).  All input images must have fast scan
    direction along the array rows (horizontal direction).  Original data is
    stored in 3D arrray sMerge.scanLines

    Parameters
    ----------
    scanAngles : array-like
        the scan angles in degrees, the same order as provided images.
    images : array-like
        provided images, can be a sequence of images (e.g. img1, img2, img3,
        im4) or as a stack (three-dimensional structure with navigation index
        first).
    linearSearch : array-like
        a 1 dimensional array contains the initial guess for the alignment.
        The default is [-0.02, -0.01,  0.  ,  0.01,  0.02].
    paddingScale : float, optional
        the scale to pad around the images, this cannot be too small
        (i.e. too close to 1). The default is 1.125.
    niter : int, optional
        the number of linear drift search to be performed. The default is 2.
    flagReportProgress : bool, optional
        whether to show progress bars or not. Default to True.
    flagPlot : bool, optional
        whether to show plot after linear drift correction. The default is True.

    Returns
    -------
    sm : sMerge object
        the sMerge object contains all the data.
    """

    # do nothing if no image is provided
    if len(images) == 0:
        return

    # ignore unknown input arguments
    _args_list = ['linearSearch', 'paddingScale', 'flagReportProgress',
                  'flagPlot', 'niter']
    for key in kwargs.keys():
        if key not in _args_list:
            msg = "The argument '{}' is not recognised, and it is ignored."
            warnings.warn(msg.format(key), RuntimeWarning)

    # set default values or from input arguments
    linearSearch = np.asarray(kwargs.get('linearSearch',
                                         np.linspace(-0.02, 0.02, num=2*2+1)))
    paddingScale = kwargs.get('paddingScale', 1.125)
    flagReportProgress = kwargs.get('flagReportProgress', True)
    flagPlot = kwargs.get('flagPlot', True)
    niter = kwargs.get('niter', 2)

    # initialise the sMerge object
    scanAngles = np.asarray(scanAngles)
    sm = sMerge(scanAngles, images,  paddingScale=paddingScale)

    # get testing linear drifts
    linearSearch *= sm.nr
    inds = np.linspace(-0.5, 0.5, num=sm.nr)[:, None]

    # get linear drift, using the first two images
    xdrift, ydrift = _get_linear_drift(sm, linearSearch, flagReportProgress,
                                       inds, niter)
    sm.xyLinearDrift = np.array([xdrift, ydrift])

    # apply linear drift to all images
    xyShift = np.hstack([inds*xdrift, inds*ydrift]).T
    _shift_origins(sm, xyShift[None, ...])

    # estimate initial alignment
    dxy = _get_initial_alignment(sm)

    # apply alignments and regenerate images
    _shift_origins(sm, dxy[..., None])

    if flagPlot:
        _plot(sm)

    return sm

def _get_linear_drift(sm, linearSearch, flagReportProgress, inds, niter=2):

    # matrix to store all correlation scores
    scores = np.empty((niter, linearSearch.size, linearSearch.size))

    # drift the image with testing drifts, refine along the way
    xRefine, yRefine = linearSearch, linearSearch
    for k in range(niter - 1):
        # get alignment score for specific drifts, first the linearSearch
        # then the refined value
        score = _get_linear_alignment_score(sm, linearSearch, inds,
                                            flagReportProgress, xRefine,
                                            yRefine)

        # record the score for this set of drift
        scores[k, ...] = score

        # refine the size of drift
        xInd, yInd = np.unravel_index(score.argmax(), score.shape)
        xstep = np.diff(xRefine)[0]
        ystep = np.diff(yRefine)[0]
        xRefine = xRefine[xInd] + np.linspace(-0.5, 0.5, xRefine.size)*xstep
        yRefine = yRefine[yInd] + np.linspace(-0.5, 0.5, yRefine.size)*ystep

    # get final score
    score = _get_linear_alignment_score(sm, linearSearch, inds,
                                        flagReportProgress, xRefine, yRefine)
    scores[niter-1, ...] = score
    sm.linearSearchScores = scores

    # get the drifts, xInd and yInd are the indices of drfits resulting
    # in the highest linearSearchScore
    xInd, yInd = np.unravel_index(score.argmax(), score.shape)
    xDrift, yDrift = np.meshgrid(xRefine, yRefine)

    r1, c1 = np.unravel_index(xInd, xDrift.shape)
    r2, c2 = np.unravel_index(yInd, yDrift.shape)
    xdrift = xDrift[r1, c1]
    ydrift = yDrift[r2, c2]

    return xdrift, ydrift

def _get_linear_alignment_score(sm, linearSearch, inds, flagReportProgress,
                                xcoord=None, ycoord=None):
    """Perform linear alignment

    Parameters
    ----------
    xcoord : array-like, optional
        the candidates for x drifts. The default is None, set to the
        default linearSearch.
    ycoord : array-like, optional
        the candidates for y drifts. The default is None, set to the
        default linearSearch.

    Returns
    -------
    linearSearchScore : ndarray
        the array storing the score for correlation for different drifts.
    """

    # xcoord, ycoord set the value/size of meshgrid
    if xcoord is None:
        xcoord = linearSearch
    else:
        xcoord = np.asarray(xcoord)

    if ycoord is None:
        ycoord = linearSearch
    else:
        ycoord = np.asarray(ycoord)

    # construct x and y drifts and initialise score array
    yDrift, xDrift = np.meshgrid(ycoord, xcoord)
    linearSearchScore = np.zeros((linearSearch.size, linearSearch.size))

    # create progress bar for nested for loop
    pbar = tqdm(total=linearSearch.size**2, desc='Linear Drift Search',
                leave=False, disable=not flagReportProgress)

    for a0 in range(linearSearch.size):
        for a1 in range(linearSearch.size):
            # calculate time dependent linear drift
            xyShift = np.hstack([inds*xDrift[a0,a1], inds*yDrift[a0,a1]])

            # apply linear drift to first two images
            # linear drift should be same for all images
            sm.scanOr[:2, ...] += xyShift.T

            # generate trial images after applying specific drifts
            sm = SPmakeImage(sm, 0)
            sm = SPmakeImage(sm, 1)

            # measure alignment score with hybrid correlation
            Icorr = _correlation(sm, 0, 1)
            linearSearchScore[a0, a1] = Icorr.max()

            # restore the first two image by removing applied linear drift
            sm.scanOr[:2, ...] -= xyShift.T

            # update progress
            pbar.update(1)

        # update progress
        pbar.update(1)

    # close the progress bar of the nested for loop
    pbar.close()

    return linearSearchScore

def _hanning_weight(sm):
    """Get the Hanning window for smoothing before Fourier transform
    """

    # chop off 0 to be consistent with the MATLAB hanningLocal
    # N = sm.scanLines.shape
    hanning = np.hanning(sm.nc + 2)[1:-1] * np.hanning(sm.nr + 2)[1:-1][:, None]
    padw = sm.imageSize - sm.img_shape
    shifts = np.floor(padw / 2 + 0.5).astype(int)
    padded = np.pad(hanning, ((0, padw[0]), (0, padw[1])),
                    mode='constant', constant_values=0)
    w2 = np.roll(padded, shifts, axis=(0,1))

    return w2

def _correlation(sm, ind1, ind2):
    """measure alignment score with hybrid correlation, using images indices
    of ind1 and ind2
    """

    w2 = _hanning_weight(sm)
    m1 = np.fft.fft2(w2 * sm.imageTransform[ind1, ...])
    m2 = np.fft.fft2(w2 * sm.imageTransform[ind2, ...])

    m = m1 * m2.conj()
    magnitude = np.sqrt(np.abs(m))
    phase = np.exp(1j*np.angle(m))
    Icorr = np.fft.ifft2(magnitude * phase).real

    return Icorr

def _shift_origins(sm, dxy):
    """shift origins of the rows by dxy and remake images, where dxy is an
    array contains x and y shifts for every image in sMerge, could be the same
    for every origin or different

    dxy must be broadcastable to scanOr
    """

    # shift the origins
    dxy = np.asarray(dxy)
    sm.scanOr += dxy

    # regenerate every image
    for k in range(sm.numImages):
        sm = SPmakeImage(sm, k)

    return

def _get_initial_alignment(sm):
    """Estimate initial alignment dxy
    """

    # first image set at (0, 0)
    dxy = np.zeros((sm.numImages, 2))

    for k in range(1, sm.numImages):
        # measure alignment score with hybrid correlation
        Icorr = _correlation(sm, k-1, k)
        dx, dy = np.unravel_index(Icorr.argmax(), Icorr.shape)

        # check if it wraps over
        nr, nc = Icorr.shape
        dx = (dx + nr/2) % nr - nr/2
        dy = (dy + nc/2) % nc - nc/2

        # alignment relative the one the was correlated
        dxy[k, :] = dxy[k-1, :] + np.array([dx, dy])

    # normalise??
    dxy[:, 0] -= dxy[:, 0].mean()
    dxy[:, 1] -= dxy[:, 1].mean()

    return dxy

def _plot(sm):
    """Plot images and their density after linear drift correction, also
    showing the reference point for later alignment step
    """
    # Plot results, image with scanline origins overlaid
    imagePlot = sm.imageTransform.mean(axis=0)
    dens = sm.imageDensity.prod(axis=0)

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

    # put a nice cross at the reference point
    ax.plot(sm.ref[1], sm.ref[0], marker='+',
            markersize=20, color=(1,1,0), markeredgewidth=5)
    ax.plot(sm.ref[1], sm.ref[0], marker='+',
            markersize=20, color='k', markeredgewidth=3)

    return
