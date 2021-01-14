"""The file contains the SPmerge01linear function
"""

import numpy as np

from scanning_drift_corr.sMerge import sMerge

def SPmerge01linear(scanAngles, *images, niter=2, plot=True):
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
    niter : int, optional
        the number of linear drift search to be performed. The default is 2.
    plot : bool, optional
        whether to show plot after linear drift correction. The default is True.

    Returns
    -------
    sm : sMerge object
        the sMerge object contains all the data.
    """

    if len(images) == 0:
        return

    scanAngles = np.asarray(scanAngles)

    # initialise the sMerge object
    sm = sMerge(scanAngles, images)

    # get linear drift
    scores, xdrift, ydrift = _get_linear_drift(sm, niter)

    # set scores and linear drift
    sm.linearSearchScores = scores
    sm.xyLinearDrift = np.array([xdrift, ydrift])

    # Apply linear drift to all images
    sm.apply_linear_drift()

    # Estimate initial alignment
    dxy = sm.estimate_initial_alignment()

    # Apply alignments and regenerate images
    sm.apply_estimated_alignment(dxy)

    if plot:
        sm.plot_linear_drift_correction()

    return sm

def _get_linear_drift(sm, niter=2):

    # matrix to store all correlation scores
    scores = np.empty((niter, sm.linearSearch.size, sm.linearSearch.size))

    xRefine, yRefine = sm.linearSearch, sm.linearSearch
    for k in range(niter - 1):

        score, xInd, yInd = sm.linear_alignment(xRefine, yRefine, return_index=True)
        xstep = np.diff(xRefine)[0]
        ystep = np.diff(yRefine)[0]

        xRefine = xRefine[xInd] + np.linspace(-0.5, 0.5, xRefine.size)*xstep
        yRefine = yRefine[yInd] + np.linspace(-0.5, 0.5, yRefine.size)*ystep
        scores[k, ...] = score

    # finally, get drifts
    score, xdrift, ydrift = sm.linear_alignment(xRefine, yRefine)
    scores[niter-1, ...] = score

    return scores, xdrift, ydrift

# def _get_linear_drift(sm):

#     # First linear alignment, search over possible linear drift vectors.
#     score1, xInd, yInd = sm.linear_alignment(return_index=True)

#     # Second linear alignment, refine possible linear drift vectors.
#     step = np.diff(sm.linearSearch)[0]
#     xRefine = sm.linearSearch[xInd] + np.linspace(-0.5, 0.5, sm.linearSearch.size)*step
#     yRefine = sm.linearSearch[yInd] + np.linspace(-0.5, 0.5, sm.linearSearch.size)*step

#     score2, xdrift, ydrift = sm.linear_alignment(xcoord=xRefine, ycoord=yRefine)

#     return score1, score2, xdrift, ydrift
