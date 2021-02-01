"""The file contains the some utility functions
"""

import numpy as np
from scipy.ndimage import distance_transform_edt

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
    
def bilinear_interpolation(sMerge, indImage, indLines=None, upsampleFactor=1):
    """Bilinear interpolation, ported from MATLAB
    """

    nc = sMerge.nc
    nr = sMerge.nr
    
    if indLines is None:
        # use all rows
        indLines = np.ones(nr, dtype=bool)

    # Expand coordinates
    t = np.arange(1, nc+1)
    x0 = sMerge.scanOr[indImage, 0, indLines][:,None]
    y0 = sMerge.scanOr[indImage, 1, indLines][:,None]

    # plus here to shift in Python's coordinate system
    xInd = x0*upsampleFactor + (upsampleFactor-1)/2 +\
        (t*sMerge.scanDir[indImage, 0])*upsampleFactor
    yInd = y0*upsampleFactor + (upsampleFactor-1)/2 +\
        (t*sMerge.scanDir[indImage, 1])*upsampleFactor

    # Prevent pixels from leaving image boundaries
    xInd = np.clip(xInd, 0, (sMerge.imageSize[0]*upsampleFactor)-2).ravel()
    yInd = np.clip(yInd, 0, (sMerge.imageSize[1]*upsampleFactor)-2).ravel()
    
    imgsize = sMerge.imageSize*upsampleFactor
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

    # raw image
    image = sMerge.scanLines[indImage, indLines, :]

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






        







