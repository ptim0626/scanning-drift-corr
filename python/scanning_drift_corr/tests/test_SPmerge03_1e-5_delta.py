"""The file contains tests for the SPmerge03 function

NOTE: For this to work, manually change the 'delta' argument in SPmakeImage
to 1e-5. The testing dataset from MATLAB was generated with the same
modification to avoid floating-point errors. To regenerate the MATLAB testing
dataset, manually add the number to the corresponding line.

Bear in mind the images here DO NOT MAKE SENSE, this is just to check the 
implementation logics based on same-input-same-output principle.
"""

import numpy as np
import scipy.io as sio
import pytest

from scanning_drift_corr.SPmerge01linear import SPmerge01linear
from scanning_drift_corr.SPmerge02 import SPmerge02
from scanning_drift_corr.SPmerge03 import SPmerge03


def test_returns_1en5_delta(MATLAB_simulated_images):

    im1, im2 = MATLAB_simulated_images
    scanAngles = (0, 90)
    
    sm = SPmerge01linear(scanAngles, im1, im2)
    sm = SPmerge02(sm, 32, 8)
    imageFinal, signalArray, densityArray = SPmerge03(sm)
    
    mfile = 'SPmerge03_simulated_0_90_1en5_delta.mat'
    mstruct = sio.loadmat('matlab_result/SPmerge03/'+mfile)
    imageFinal_m = mstruct['imageFinal']
    signalArray_m = mstruct['signalArray'].T
    densityArray_m = mstruct['densityArray'].T


    assert np.isclose(imageFinal, imageFinal_m).all()
    assert np.isclose(signalArray[0,...], signalArray_m[0,...].T, atol=1e-5).all()
    assert np.isclose(signalArray[1,...], signalArray_m[1,...].T, atol=1e-5).all()
    assert np.isclose(densityArray[0,...], densityArray_m[0,...].T, atol=1e-5).all()
    assert np.isclose(densityArray[1,...], densityArray_m[1,...].T, atol=1e-5).all()
    
