"""The file contains the SPmerge02_final function
"""

import warnings

import numpy as np
from scipy.signal import convolve

from scanning_drift_corr.SPmakeImage import SPmakeImage
from scanning_drift_corr.SPmerge02_initial import SPmerge02_initial
from scanning_drift_corr.tools import distance_transform


def SPmerge02_final(sm, scanOrStep, **kwargs):
    """
    
    Parameters
    ----------
    sm : sMerge object
        the sMerge object contains all the data.    
    densityCutoff : float, optional
        density cutoff for image boundaries (norm. to 1). Default to 0.8.
    refineInitialStep : float, optional
        initial step size for final refinement, in pixels. Default to 0.5.        
    stepSizeReduce : float, optional
        when a scanline origin does not move, step size will be reduced by
        this factor. Default to 0.5.
    flagPointOrder : bool, optional
        use this flag to force origins to be ordered, i.e. disallow points
        from changing their order. Default to True.


    flagGlobalShift : bool, optional
        if this flag is true, a global phase correlation, performed each
        final iteration (This is meant to fix unit cell shifts and similar
        artifacts). This option is highly recommended! Default to False.
    flagGlobalShiftIncrease : bool, optional
        if this option is true, the global scoring function is allowed to
        increase after global phase correlation step. (false is more stable)
        Default to False.
    minGlobalShift : float, optional
        global shifts only if shifts > this value (pixels). Default to 1.
    densityDist : float, optional
         density mask edge threshold. To generate a moving average along the
         scanline origins (make scanline steps more linear). Default to mean
         of raw data divided by 32.
    """
 
    # ignore unknown input arguments
    _args_list = ['densityCutoff', 'stepSizeReduce', 'flagPointOrder']
    for key in kwargs.keys():
        if key not in _args_list:
            msg = "The argument '{}' is not recognised, and it is ignored."
            warnings.warn(msg.format(key), RuntimeWarning)


    # for final alignment
    densityCutoff = kwargs.get('densityCutoff', 0.8)
    stepSizeReduce = kwargs.get('stepSizeReduce', 1/2)
    flagPointOrder = kwargs.get('flagPointOrder', True)

    
    # scanOrStep = np.ones((sm.numImages, sm.nr)) * refineInitialStep
    dxy = np.array([[0,1,-1,0,0], [0,0,0,1,-1]])
    
    
    # Reset pixels moved count
    pixelsMoved = 0



    # Refine each image in turn, against the sum of all other images
    for k in range(sm.numImages):
        # Generate alignment image, mean of all other scanline datasets,
        # unless user has specified a reference image.
        if sm.imageRef is None:
            indsAlign = np.arange(sm.numImages, dtype=int)
            indsAlign = indsAlign[indsAlign != k]

            dens_cut = sm.imageDensity[indsAlign, ...] > densityCutoff
            imageAlign = (sm.imageTransform[indsAlign, ...] * dens_cut).sum(axis=0)
            dens = dens_cut.sum(axis=0)
            sub = dens > 0
            imageAlign[sub] = imageAlign[sub] / dens[sub]
            imageAlign[~sub] = np.mean(imageAlign[sub])
        else:
            imageAlign = sm.imageRef


        # If ordering is used as a condition, determine parametric positions
        if flagPointOrder:
            # Use vector perpendicular to scan direction (negative 90 deg)
            nn = np.array([sm.scanDir[k, 1], -sm.scanDir[k, 0]])
            vParam = nn[0]*sm.scanOr[k, 0, :] + nn[1]*sm.scanOr[k, 1, :]

        # Loop through each scanline and perform alignment
        for m in range(sm.nr):
            # Refine score by moving the origin of this scanline
            orTest = sm.scanOr[k, :, m][:, None] + dxy*scanOrStep[k, m]

            # If required, force ordering of points
            if flagPointOrder:
                vTest = nn[0]*orTest[0, :] + nn[1]*orTest[1, :]

                if m == 0:
                    # no lower bound?
                    vBound = np.array([-np.inf, vParam[m+1]])
                elif m == sm.nr-1:
                    # no upper bound?
                    vBound = np.array([vParam[m-1], np.inf])
                else:
                    vBound = np.array([vParam[m-1], vParam[m+1]])

                # check out of bound entries?
                for p in range(dxy.shape[1]):
                    if vTest[p] < vBound[0]:
                        orTest[:, p] += nn*(vBound[0]-vTest[p])
                    elif vTest[p] > vBound[1]:
                        orTest[:, p] += nn*(vBound[1]-vTest[p])

            # Loop through origin tests
            inds = np.arange(1, sm.nc+1)
            score = np.zeros(dxy.shape[1])
            for p in range(dxy.shape[1]):
                xInd = orTest[0, p] + inds*sm.scanDir[k, 0]
                yInd = orTest[1, p] + inds*sm.scanDir[k, 1]

                # Prevent pixels from leaving image boundaries
                xInd = np.clip(xInd, 0, sm.imageSize[0]-2).ravel()
                yInd = np.clip(yInd, 0, sm.imageSize[1]-2).ravel()

                # Bilinear coordinates
                xF = np.floor(xInd).astype(int)
                yF = np.floor(yInd).astype(int)
                dx = xInd - xF
                dy = yInd - yF

                # scanLines indices switched
                score[p] = calcScore(imageAlign, xF, yF, dx, dy,
                                     sm.scanLines[k, m, :])

            # Note that if moving origin does not change score, dxy = (0,0)
            # will be selected (ind = 0).
            ind = np.argmin(score)
            if ind == 0:
                # Reduce the step size for this origin
                scanOrStep[k, m] *= stepSizeReduce
            else:
                pshift = np.linalg.norm(orTest[:,ind] - sm.scanOr[k, :, m])
                pixelsMoved += pshift
                # change sMerge!
                sm.scanOr[k, :, m] = orTest[:,ind]

            #TODO add progress bar tqdm later   
    
    
    
    
    return pixelsMoved




def calcScore(image, xF, yF, dx, dy, intMeas):

    imgsz = image.shape

    rind1 = np.ravel_multi_index((xF, yF), imgsz)
    rind2 = np.ravel_multi_index((xF+1, yF), imgsz)
    rind3 = np.ravel_multi_index((xF, yF+1), imgsz)
    rind4 = np.ravel_multi_index((xF+1, yF+1), imgsz)

    int1 = image.ravel()[rind1] * (1-dx) * (1-dy)
    int2 = image.ravel()[rind2] * dx * (1-dy)
    int3 = image.ravel()[rind3] * (1-dx) * dy
    int4 = image.ravel()[rind4] * dx * dy

    imageSample = int1 + int2 + int3 + int4

    score = np.abs(imageSample - intMeas).sum()

    return score


