"""The file contains the SPmerge02_initial function
"""

import warnings

import numpy as np
from scipy.ndimage.morphology import binary_dilation

from scanning_drift_corr.SPmakeImage import SPmakeImage


def SPmerge02_initial(sm, **kwargs):
    """Initial alignment

    Parameters
    ----------
    sm : sMerge object
        the sMerge object contains all the data.
    densityCutoff : float, optional
        density cutoff for image boundaries (norm. to 1). Default to 0.8.
    distStart : float, optional
        radius of # of scanlines used for initial alignment. Default to
        mean of raw data divided by 16.
    initialShiftMaximum : float, optional
        maximum number of pixels shifted per line for the initial alignment
        step. This value should have a maximum of 1, but can be set lower
        to stabilize initial alignment. Default to 0.25.
    """

    # ignore unknown input arguments
    _args_list = ['densityCutoff', 'distStart', 'initialShiftMaximum']
    for key in kwargs.keys():
        if key not in _args_list:
            msg = "The argument '{}' is not recognised, and it is ignored."
            warnings.warn(msg.format(key), RuntimeWarning)

    meanScanLines = np.mean(sm.scanLines.shape[1:])
    densityCutoff = kwargs.get('densityCutoff', 0.8)
    distStart = kwargs.get('distStart', meanScanLines/16)
    initialShiftMaximum = kwargs.get('initialShiftMaximum', 1/4)

    # Rough initial alignment of scanline origins, to nearest pixel
    sm.scanActive = np.zeros((sm.numImages, sm.nr), dtype=bool)
    indStart = _get_starting_scanlines(sm, distStart)
    for k in range(sm.numImages):
        # get alignment image for the current image, based on orthogonality
        imageAlign = _get_reference_image(sm, k, densityCutoff)
        imageAlign = imageAlign.ravel()

        # align origins and get the step size
        dOr = sm.scanOr[k, :, 1:] - sm.scanOr[k, :, :-1]
        xyStep = np.mean(dOr, axis=1)

        # set the aligned indices for this image
        indAligned = np.zeros(sm.nr, dtype=bool)
        indAligned[indStart[k]] = True

        # start the alignment and stop until all have been aligned
        dxy = np.array([[0,1,-1,0,0], [0,0,0,1,-1]])
        while not indAligned.all():
            # Determine scanline indices to check next
            # indMove contains the indices of scanline to check
            # indsActive contains the indices of currently active scanlines
            inds = np.arange(sm.nr)
            v = binary_dilation(indAligned)
            v[indAligned] = False
            indsMove = inds[v]
            indsActive = inds[indAligned]

            # loop over each selected scan line
            for m in indsMove:
                # determine starting point from neighboring scanline
                xyOr = _get_xyOr(sm, k, m, indsActive, xyStep)

                # score each of the moved selected scan line against the
                # reference image (imageAlign)
                score = np.zeros(dxy.shape[1])
                raw_scanline = sm.scanLines[k, m, :]
                for p in range(dxy.shape[1]):
                    xymove = dxy[:, p]
                    score[p] = _get_score(sm, imageAlign, xyOr, k, xymove,
                                          raw_scanline)

                # move the scan line
                ind = np.argmin(score)
                sm.scanOr[k, :, m] = xyOr + dxy[:, ind]*initialShiftMaximum
                indAligned[m] = True

    return

def _get_starting_scanlines(sm, distStart):
    """ Get starting scanlines for initial alignment
    indStart is an array containing the index of the starting scanline for
    each image
    """

    indStart = np.zeros(sm.numImages, dtype=int)
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

def _get_reference_image(sm, k, densityCutoff):
    """Generate alignment image, use the most orthogonal image to current one
    unless user has specified a reference image.
    """

    # sum of row of ortho is 0 if they are exact orthogonal
    ortho = (sm.scanDir[k, :] * sm.scanDir).sum(axis=1)
    indAlign = np.argmin(np.abs(ortho))

    if sm.imageRef is None:
        sm = SPmakeImage(sm, indAlign, sm.scanActive[indAlign, :])
        dens_cut = sm.imageDensity[indAlign, ...] > densityCutoff
        imageAlign = sm.imageTransform[indAlign, ...] * dens_cut
    else:
        imageAlign = sm.imageRef

    return imageAlign

def _get_xyOr(sm, k, m, indsActive, xyStep):
    """Determine starting point from neighboring scanline
    """

    minDistInd = np.argmin(np.abs(m - indsActive))

    # Step perpendicular to scanDir (orthogonality)
    indMin = indsActive[minDistInd]
    xyOr = sm.scanOr[k, :, indMin] + xyStep * (m - indMin)

    return xyOr

def _get_score(sm, imageAlign, xyOr, k, xymove, raw_scanline):
    """Refine score by moving origin of this scanline
    """

    t = np.arange(1, sm.nc+1)
    xInd = np.floor(xyOr[0] + t*sm.scanDir[k, 0] + 0.5).astype(int)
    yInd = np.floor(xyOr[1] + t*sm.scanDir[k, 1] + 0.5).astype(int)

    # move the scan line
    dx, dy = xymove
    nxInd = xInd + dx
    nyInd = yInd + dy

    # Prevent pixels from leaving image boundaries
    nxInd = np.core.umath.clip(nxInd, 0, sm.imageSize[0]-2).ravel()
    nyInd = np.core.umath.clip(nyInd, 0, sm.imageSize[1]-2).ravel()    
    # same as np.ravel_multi_index((nxInd, nyInd), sm.imageSize)
    # but quicker, why?
    rInd = nyInd + nxInd*sm.imageSize[-1]

    # calculate the score after moving the scanline
    score = np.abs(imageAlign[rInd] - raw_scanline).sum()

    return score
