"""This file contains the function SPmakeImage
"""

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt

def SPmakeImage(sMerge, indImage, indLines=None):
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

    Returns
    -------
    sMerge : sMerge object
        the sMerge object.
    """

    nr = sMerge.nr
    if indLines is None:
        # use all rows
        indLines = np.ones(nr, dtype=bool)

    # perform bilinear interpolation
    sig, count = _bilinear_interpolation(sMerge, indImage, indLines)

    # Apply KDE
    sig, count = _apply_KDE(sMerge, sig, count)

    # cheat mode
    count += 1e-10

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
    # the reverse, hence the negation
    dt = distance_transform_edt(~bound)
    dtmin = np.minimum(dt/sMerge.edgeWidth, 1)
    sMerge.imageDensity[indImage, ...] = np.sin(dtmin*np.pi/2)**2

    return sMerge

def _bilinear_interpolation(sMerge, indImage, indLines):
    """Bilinear interpolation, ported from MATLAB
    """

    nc = sMerge.nc

    # Expand coordinates
    t = np.arange(1, nc+1)
    x0 = sMerge.scanOr[indImage, 0, indLines][:,None]
    y0 = sMerge.scanOr[indImage, 1, indLines][:,None]
    xInd = x0 + t * sMerge.scanDir[indImage, 0]
    yInd = y0 + t * sMerge.scanDir[indImage, 1]

    # Prevent pixels from leaving image boundaries
    # in MATLAB is column vector, here is row vector
    # cap at indices 1 lower than MATLAB
    xInd = np.clip(xInd, 0, sMerge.imageSize[0]-2).ravel()
    yInd = np.clip(yInd, 0, sMerge.imageSize[1]-2).ravel()

    # Convert to bilinear interpolants and weights
    # xAll/yAll have 4 rows, each represent the interpolants of the pixel of
    # the image which as are column vec (column size is raw data size)
    xIndF = np.floor(xInd).astype(int)
    yIndF = np.floor(yInd).astype(int)
    xAll = np.vstack([xIndF, xIndF+1, xIndF, xIndF+1])
    yAll = np.vstack([yIndF, yIndF, yIndF+1, yIndF+1])
    dx = xInd - xIndF
    dy = yInd - yIndF

    # indAll in MATLAB is from sub2ind
    w = np.vstack([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy])
    indAll = np.ravel_multi_index((xAll, yAll), sMerge.imageSize)

    # raw image
    sL = sMerge.scanLines[indImage, indLines, :]

    # weigh the raw image for interpolation
    wsig = w * sL.ravel()
    wcount = w

    # Generate image
    sig = np.bincount(indAll.ravel(), weights=wsig.ravel(),
                      minlength=sMerge.imageSize.prod()).reshape(sMerge.imageSize)
    count = np.bincount(indAll.ravel(), weights=wcount.ravel(),
                      minlength=sMerge.imageSize.prod()).reshape(sMerge.imageSize)

    return sig, count

def _apply_KDE(sMerge, sig, count):
    """Apply KDE
    """

    # r min at 5
    r = np.maximum(np.ceil(sMerge.KDEsigma*3), 5)

    # the parameters match the behaviour of convolving a normalised Gaussian
    # kernel in MATLAB
    fargs = {'sigma' : sMerge.KDEsigma,
             'mode' : 'constant',
             'cval' : 0,
             'truncate' : r / sMerge.KDEsigma}

    sig = gaussian_filter(sig, **fargs)
    count = gaussian_filter(count, **fargs)

    return sig, count
