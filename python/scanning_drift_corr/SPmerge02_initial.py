"""The file contains the SPmerge02_initial function
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.ndimage.morphology import binary_dilation

from scanning_drift_corr.SPmakeImage import SPmakeImage
from scanning_drift_corr.tools import distance_transform

def SPmerge02_initial(sm, **kwargs):
    """

    Parameters
    ----------
    sm : TYPE
        DESCRIPTION.
    densityCutoff : float, optional
        density cutoff for image boundaries (norm. to 1). Default to 0.8.
    distStart : float, optional
        radius of # of scanlines used for initial alignment. Default to
        mean of raw data divided by 16.
    initialShiftMaximum : float, optional
        maximum number of pixels shifted per line for the initial alignment 
        step. This value should have a maximum of 1, but can be set lower
        to stabilize initial alignment. Default to 0.25.
    originInitialAverage : float, optional
        window sigma in px for initial smoothing. Default to mean of raw data 
        divided by 16.
        
    Returns
    -------
    None.

    """
    
    
    # ignore unknown input arguments
    _args_list = ['densityCutoff', 'distStart', 'initialShiftMaximum', 
                  'originInitialAverage', 'flagReportProgress']
    for key in kwargs.keys():
        if key not in _args_list:
            msg = "The argument '{}' is not recognised, and it is ignored."
            warnings.warn(msg.format(key), RuntimeWarning)
    
    meanScanLines = np.mean(sm.scanLines.shape[1:])
    densityCutoff = kwargs.get('densityCutoff', 0.8)
    distStart = kwargs.get('distStart', meanScanLines/16)
    initialShiftMaximum = kwargs.get('initialShiftMaximum', 1/4)
    originInitialAverage = kwargs.get('originInitialAverage', meanScanLines/16)
    
    
    
    
    return
