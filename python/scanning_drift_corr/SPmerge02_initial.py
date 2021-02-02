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
    indStart = _get_starting_scanline(sm, distStart)
    for k in range(sm.numImages):
        # Determine which image to align to, based on orthogonality
        # sum of row of ortho is 0 if they are exact orthogonal
        ortho = (sm.scanDir[k, :] * sm.scanDir).sum(axis=1)
        indAlign = np.argmin(np.abs(ortho))

        if sm.imageRef is None:
            # no reference image, use the most orthogonal image to current one
            sm = SPmakeImage(sm, indAlign, sm.scanActive[indAlign, :])
            imageAlign = sm.imageTransform[indAlign, ...] *\
                (sm.imageDensity[indAlign, ...]>densityCutoff)
        else:
            imageAlign = sm.imageRef

        # align origins
        dOr = sm.scanOr[k, :, 1:] - sm.scanOr[k, :, :-1]
        xyStep = np.mean(dOr, axis=1)

        # set the aligned indices for this image
        indAligned = np.zeros(sm.nr, dtype=bool)
        indAligned[int(indStart[k])] = True

        # align selected scanlines
        while not indAligned.all():
            _align_selected_scanline(sm, k, indAligned, xyStep, imageAlign,
                                     initialShiftMaximum)

    return


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


def _align_selected_scanline(sm, k, indAligned, xyStep, imageAlign,
                             initialShiftMaximum):
    """ndarray indAligned could be mutated.
    """

    inds = np.arange(sm.nr)
    dxy = np.array([[0,1,-1,0,0], [0,0,0,1,-1]])
    score = np.zeros(dxy.shape[1])

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

        # Prevent pixels from leaving image boundaries
        # I think better to check?
        xInd = np.clip(xInd, 0, sm.imageSize[0]-2).ravel()
        yInd = np.clip(yInd, 0, sm.imageSize[1]-2).ravel()

        for n in range(dxy.shape[1]):
            nxInd = xInd + dxy[0, n]
            nyInd = yInd + dxy[1, n]

            # Prevent pixels from leaving image boundaries
            # I think better to check?
            nxInd = np.clip(nxInd, 0, sm.imageSize[0]-2).ravel()
            nyInd = np.clip(nyInd, 0, sm.imageSize[1]-2).ravel()
            rInd = np.ravel_multi_index((nxInd, nyInd), sm.imageSize)

            # scanLines indices switched to match row
            score[n] = np.abs(imageAlign.ravel()[rInd] -
                              sm.scanLines[k, indMove[m], :]).sum()

        ind = np.argmin(score)

        # move the scan line
        sm.scanOr[k, :, indMove[m]] = xyOr + dxy[:, ind]*initialShiftMaximum
        indAligned[indMove[m]] = True

    return
