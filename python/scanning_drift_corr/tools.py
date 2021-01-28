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
    
    