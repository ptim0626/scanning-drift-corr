import numpy as np
import scipy.io as sio
import pytest

from scanning_drift_corr.SPmerge01linear import SPmerge01linear
from scanning_drift_corr.SPmerge02 import SPmerge02



def test_initial_alignment_simulated_data_1en5_delta(MATLAB_simulated_images):

    im1, im2 = MATLAB_simulated_images
    scanAngles = (0, 90)
    

    sm = SPmerge01linear(scanAngles, im1, im2)
    sm = SPmerge02(sm, 0, 8, only_initial_refinemen=True)
    
    
    imgtrans0 = sm.imageTransform[0, ...]
    imgtrans1 = sm.imageTransform[1, ...]
    imgden0 = sm.imageDensity[0, ...]
    imgden1 = sm.imageDensity[1, ...]
    scanOr = sm.scanOr
    scanAc = sm.scanActive

    mfile = 'SPmerge02_initial_alignment_simulated_0_90.mat'
    mstruct = sio.loadmat('matlab_result/SPmerge02/'+mfile)
    imgtrans0_m = mstruct['it1']
    imgtrans1_m = mstruct['it2']
    imgden0_m = mstruct['id1']
    imgden1_m = mstruct['id2']
    scanOr_m = mstruct['scOr'].T - 1
    scanAc_m = mstruct['scAc'].T

    assert np.isclose(imgtrans0, imgtrans0_m).all()
    assert np.isclose(imgtrans1, imgtrans1_m).all()
    assert np.isclose(imgden0, imgden0_m).all()
    assert np.isclose(imgden1, imgden1_m).all()
    assert np.isclose(scanOr, scanOr_m).all()
    assert np.isclose(scanAc, scanAc_m).all()
