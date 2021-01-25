"""The file contains the SPmerge02 function
"""

import numpy as np
from scipy.signal import convolve
from scipy.ndimage.morphology import binary_dilation

# from scanning_drift_corr.sMerge import sMerge
from scanning_drift_corr.SPmakeImage import SPmakeImage

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
        _initial_refinement(sm, initialRefineSteps, distStart, densityCutoff, initialShiftMaximum)

    
    return sm


def _initial_refinement(sm, initialRefineSteps, distStart, densityCutoff, initialShiftMaximum):

    for _ in range(initialRefineSteps):
        sz = sm.scanLines.shape[1]
        sm.scanActive = np.zeros((sm.numImages, sz), dtype=bool)
        
        indStart = _get_starting_scanline(sm, distStart)
            
        # Rough initial alignment of scanline origins, to nearest pixel
        inds = np.arange(sz)
        dxy = np.array([[0,1,-1,0,0], [0,0,0,1,-1]])
        score = np.zeros(dxy.shape[1])
        for k in range(sm.numImages):
            # Determine which image to align to, based on orthogonality
            ortho = (sm.scanDir[k, :] * sm.scanDir).sum(axis=1)
            indAlign = np.argmin(np.abs(ortho))
            
            if sm.imageRef is None:

                sm = SPmakeImage(sm, indAlign, sm.scanActive[indAlign, :])

                imageAlign = sm.imageTransform[indAlign, ...] * (sm.imageDensity[indAlign, ...]>densityCutoff)

            else:
                imageAlign = sm.imageRef
        
            # align origins
            dOr = sm.scanOr[k, :, 1:] - sm.scanOr[k, :, :-1]
            sz = sm.scanLines.shape[1]
            xyStep = np.mean(dOr, axis=1)
            indAligned = np.zeros(sz, dtype=bool)
            indAligned[int(indStart[k])] = True
            
            
            while not indAligned.all():

                # Align selected scanlines
                _align_selected_scanline(sm, k, inds, indAligned, xyStep, dxy, score, imageAlign, initialShiftMaximum)


        
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
    
        
        
        
def _align_selected_scanline(sm, k, inds, indAligned, xyStep, dxy, score, imageAlign, initialShiftMaximum):
    # Determine scanline indices to check next
    v = binary_dilation(indAligned)
    v[indAligned] = False
    indMove = inds[v]

    # currently active scanlines
    indsActive = inds[indAligned]
    
    for m in range(indMove.size):
        # determine starting point from neighboring scanline
        minDistInd = np.argmin(np.abs(indMove[m] - indsActive))
        
        # Step perpendicular to scanDir
        indMin = indsActive[minDistInd]
        xyOr = sm.scanOr[k, :, indMin] + xyStep * (indMove[m] - indMin)

        # Refine score by moving origin of this scanline
        xInd = np.floor(xyOr[0] + 1 + (inds+1)*sm.scanDir[k, 0] + 0.5).astype(int) - 1
        yInd = np.floor(xyOr[1] + 1 + (inds+1)*sm.scanDir[k, 1] + 0.5).astype(int) - 1
        
        # Prevent pixels from leaving image boundaries (here mainly for strange image?)
        xInd = np.clip(xInd, 0, sm.imageSize[0]-2).ravel()
        yInd = np.clip(yInd, 0, sm.imageSize[1]-2).ravel()
        
        for n in range(dxy.shape[1]):

            nxInd = xInd + dxy[0, n]
            nyInd = yInd + dxy[1, n]
            # Prevent pixels from leaving image boundaries (here mainly for strange image?)
            nxInd = np.clip(nxInd, 0, sm.imageSize[0]-2).ravel()
            nyInd = np.clip(nyInd, 0, sm.imageSize[1]-2).ravel()


            pp = np.ravel_multi_index((nxInd, nyInd), sm.imageSize)

            # scanLines indices switched?
            score[n] = np.abs(imageAlign.ravel()[pp] - sm.scanLines[k, indMove[m], :]).sum()

        ind = np.argmin(score)
        
        # change sMerge!
        sm.scanOr[k, :, indMove[m]] = xyOr + dxy[:, ind]*initialShiftMaximum
        indAligned[indMove[m]] = True

        #TODO add progress bar tqdm later

