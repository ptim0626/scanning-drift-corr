"""The file contains the SPmerge02 function
"""

import numpy as np
from scipy.signal import convolve

# from scanning_drift_corr.sMerge import sMerge

def SPmerge02(sm, refineMaxSteps=None, initialRefineSteps=None):
    
    flagPlot = 1

    # Set to true to see updates on console.
    flagReportProgress = 1

    # density cutoff for image boundaries (norm. to 1).
    densityCutoff = 0.8

    # Radius of # of scanlines used for initial alignment.
    distStart = np.mean(sm.scanLines.shape[1:]) / 16
    
    # Max number of pixels shifted per line for the
    # initial alignment step.  This value should
    # have a maximum of 1, but can be set lower
    # to stabilize initial alignment.    
    initialShiftMaximum = 1/4

    # Initial step size for main refinement, in pixels.    
    refineInitialStep = 1/2

    # If number of pixels shifted (per image) is 
    # below this value, refinement will be halted.
    pixelsMovedThreshold = 0.1

    # When a scanline origin does not move,
    # step size will be reduced by this factor.
    stepSizeReduce = 1/2

    # Use this flag to force origins to be ordered, i.e.
    # disallow points from changing their order.
    flagPointOrder = 1

    # If this flag is true, a global phase correlation
    # performed each primary iteration (This is meant to
    # fix unit cell shifts and similar artifacts).
    # This option is highly recommended!
    flagGlobalShift = 1*0

    # If this option is true, the global scoring 
    # function is allowed to increase after global
    # phase correlation step. (false is more stable)
    flagGlobalShiftIncrease = 0

    # Global shifts only if shifts > this value (pixels)
    minGlobalShift = 1

    # density mask edge threshold 
    # To generate a moving average along the scanline origins
    # (make scanline steps more linear), use the settings below:
    densityDist =  np.mean(sm.scanLines.shape[1:]) / 32
    
    # Window sigma in px for smoothing scanline origins.
    # Set this value to zero to not use window avg.
    # This window is relative to linear steps.
    originWindowAverage = 1

    # Window sigma in px for initial smoothing.
    originInitialAverage = np.mean(sm.scanLines.shape[1:]) / 16
    
    # Set this value to true to redo initial alignment.    
    resetInitialAlignment = False
    
    # Default number of iterations if user does not provide values
    nargs = 3
    if refineMaxSteps is None:
        refineMaxSteps = 32
        nargs -= 1
    if initialRefineSteps is None:
        initialRefineSteps = 0
        nargs -= 1

    # Make kernel for moving average of origins
    if originInitialAverage > 0:
        r = np.ceil(3*originInitialAverage)
        v = np.arange(-r, r+1)
        KDEorigin = np.exp(-v**2/(2*originInitialAverage**2))

        KDEnorm = 1 / convolve(np.ones(sm.scanOr.shape), KDEorigin[:, None, None].T, 'same')
        sz = sm.scanLines.shape[1]
        basisOr = np.vstack([np.zeros(sz), np.arange(0, sz)])
        scanOrLinear = np.zeros(sm.scanOr.shape)

    flag = ((sm.scanActive is None) | resetInitialAlignment | 
            (initialRefineSteps > 0)) & (nargs == 3)
    if flag:
        _initial_refinement(sm, initialRefineSteps, distStart)

    
    return sm


def _initial_refinement(sm, initialRefineSteps, distStart):

    for _ in range(initialRefineSteps):
        sz = sm.scanLines.shape[1]
        sm.scanActive = np.zeros((sm.numImages, sz), dtype=bool)
        
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
            
            print(dist)
            
            
            