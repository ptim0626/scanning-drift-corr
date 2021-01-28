"""The file contains the some utility functions
"""

import numpy as np
from scipy.ndimage import distance_transform_edt


def distance_transform(binary_image):
    # just to be consistent with MATLAB output
    # distance_transform_edt returns 0 but MATLAB bwdist return inf when
    # there are no nonzero entries
    
    if np.any(binary_image):
        return distance_transform_edt(~binary_image)
    else:
        return np.full(binary_image.shape, np.inf)
    
    