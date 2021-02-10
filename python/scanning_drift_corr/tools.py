"""The file contains the some utility functions
"""

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt

def distance_transform(binary_image):
    """ Same as bwdist in MATLAB,  computes the Euclidean distance transform
    of the binary image. For each pixel, the distance transform assigns a
    number that is the distance between that pixel and the nearest nonzero
    pixel of the binary image.

    Parameters
    ----------
    binary_image : array-like
        the binary image

    Returns
    -------
    ndarray
        the distance transform.
    """

    binary_image = np.asarray(binary_image, dtype=bool)

    if np.any(binary_image):
        return distance_transform_edt(~binary_image)
    else:
        return np.full(binary_image.shape, np.inf)

def bilinear_interpolation(scanLines, scanOr, scanDir, imageSize,
                           indLines=None, upsampleFactor=1):
    """Bilinear interpolation, ported from MATLAB

    Parameters
    ----------
    scanLines : array-like
        the original raw image at this scanning direction
    scanOr : array-like
        the scan line origins of the image
    scanDir : array-like
        the scanning direction
    imageSize : array-like
        the new image size
    indLines : array-like
        contains position of active scan line origins
    upsampleFactor : float
        the oversampling ratio

    Returns
    -------
    sig : ndarray
        the interpolated signal
    count : ndarray
        the count of weights in interpolating signal
    """

    scanLines = np.asarray(scanLines)
    scanOr = np.asarray(scanOr)
    scanDir = np.asarray(scanDir)
    imageSize = np.asarray(imageSize)

    nr, nc = scanLines.shape
    if indLines is None:
        # use all rows
        indLines = np.ones(nr, dtype=bool)
    else:
        indLines = np.asarray(indLines, dtype=bool)

    # Expand coordinates
    t = np.arange(1, nc+1)
    x0 = scanOr[0, indLines][:,None]
    y0 = scanOr[1, indLines][:,None]

    # plus here to shift in Python's coordinate system
    xInd = x0*upsampleFactor + (upsampleFactor-1)/2 +\
        (t*scanDir[0])*upsampleFactor
    yInd = y0*upsampleFactor + (upsampleFactor-1)/2 +\
        (t*scanDir[1])*upsampleFactor

    # Prevent pixels from leaving image boundaries
    xInd = np.clip(xInd, 0, (imageSize[0]*upsampleFactor)-2).ravel()
    yInd = np.clip(yInd, 0, (imageSize[1]*upsampleFactor)-2).ravel()

    imgsize = imageSize*upsampleFactor
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
    indAll = np.ravel_multi_index((xAll, yAll), imgsize)

    # get the active scan line from the raw image
    image = scanLines[indLines, :]

    # weigh the raw image for interpolation
    wsig = w * image.ravel()
    wcount = w

    # Generate image and density
    sig = np.bincount(indAll.ravel(),
                      weights=wsig.ravel(),
                      minlength=imgsize.prod()).reshape(imgsize)
    count = np.bincount(indAll.ravel(),
                        weights=wcount.ravel(),
                        minlength=imgsize.prod()).reshape(imgsize)

    return sig, count

def apply_KDE(img, KDEsigma, rmin=5):
    """Apply KDE

    Parameters
    ----------
    img : array-like
        the image to be convolved
    KDEsigma : float
        the sigma value of the Gaussian kernel
    rmin : int
        the minimum radius to be truncated after convolution. The default is 5.

    Returns
    -------
    imgconv : ndarray
        the convolved image
    """

    # set the truncated width
    r = np.maximum(np.ceil(KDEsigma*3), rmin)

    # the parameters match the behaviour of convolving a normalised Gaussian
    # kernel in MATLAB
    fargs = {'sigma' : KDEsigma,
             'mode' : 'constant',
             'cval' : 0,
             'truncate' : r / KDEsigma}

    imgconv = gaussian_filter(img, **fargs)

    return imgconv
