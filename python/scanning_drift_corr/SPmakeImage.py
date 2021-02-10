"""This file contains the function SPmakeImage
"""

import numpy as np

from scanning_drift_corr.tools import distance_transform, \
    bilinear_interpolation, apply_KDE

def SPmakeImage(sMerge, indImage, indLines=None, delta=0):
    """
    This function generates a resampled scanning probe image with dimensions
    of imageSize, from a an array of N scan lines given in scaneLines,
    (lines specified as image rows), from an array of Nx2 origins in scanOr.
    scanDir is a 2 element vector specifying the direction of the scan.
    All arrays are stored inside struct sMerge.  ind specified update index.
    indLines is a vector of binary values specifying which lines to include.

    Parameters
    ----------
    sMerge : sMerge object
        the sMerge object.
    indImage : int
        the index of the image to be transformed.
    indLines : ndarray, optional
        an array of binary values specifying which lines to include.
        The default is None, set to use all rows.
    cheat : float, optional
        Developer's use, for normal usage, should be 0. A small number to
        the 'count' array to be consistent with MATLAB while checking the
        logical of implementation.

    Returns
    -------
    sMerge : sMerge object
        the sMerge object.
    """

    # perform bilinear interpolation
    scanLines = sMerge.scanLines[indImage, ...]
    scanOr = sMerge.scanOr[indImage, ...]
    scanDir = sMerge.scanDir[indImage, :]
    imageSize = sMerge.imageSize
    sig, count = bilinear_interpolation(scanLines, scanOr, scanDir, imageSize,
                                        indLines=indLines)

    # Apply KDE
    sig = apply_KDE(sig, sMerge.KDEsigma)
    count = apply_KDE(count, sMerge.KDEsigma)

    # cheat mode!
    if delta:
        count += delta

    # the precision(?) in MATLAB sometimes results in edge value being evaluated
    # as zero while it is not (shouldn't be worried?)
    sub = count > 0
    sig[sub] /= count[sub]
    sMerge.imageTransform[indImage, ...] = sig

    # Estimate sampling density
    bound = count == 0
    bound[[0,-1], :] = True
    bound[:, [0, -1]] = True

    # MATLAB bwdist calculates 'the distance between that pixel and the
    # nearest nonzero pixel', scipy version is more conventional, which is
    # the reverse, use a wrapper to handle this
    dt = distance_transform(bound)
    dtmin = np.minimum(dt/sMerge.edgeWidth, 1)
    sMerge.imageDensity[indImage, ...] = np.sin(dtmin*np.pi/2)**2

    return sMerge
