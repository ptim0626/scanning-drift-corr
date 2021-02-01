"""The file contains tests for the SPmerge02 function

NOTE: For this to work, manually change the 'delta' argument in SPmakeImage
to 1e-3. The testing dataset from MATLAB was generated with the same
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

def test_final_alignment_WITH_global_phase_corr_1en3_delta(MATLAB_simulated_images):

    im1, im2 = MATLAB_simulated_images
    scanAngles = (0, 90)

    sm = SPmerge01linear(scanAngles, im1, im2)
    sm = SPmerge02(sm, 32, 8)

    imgtrans0 = sm.imageTransform[0, ...]
    imgtrans1 = sm.imageTransform[1, ...]
    imgden0 = sm.imageDensity[0, ...]
    imgden1 = sm.imageDensity[1, ...]
    scanOr = sm.scanOr
    scanAc = sm.scanActive
    stats = sm.stats

    mfile = 'SPmerge02_final_alignment_WITH_global_corr_simulated_0_90_1en3_delta.mat'
    mstruct = sio.loadmat('matlab_result/SPmerge02/'+mfile)
    imgtrans0_m = mstruct['it1']
    imgtrans1_m = mstruct['it2']
    imgden0_m = mstruct['id1']
    imgden1_m = mstruct['id2']
    scanOr_m = mstruct['scOr'].T - 1
    scanAc_m = mstruct['scAc'].T
    stats_m = mstruct['stats']

    assert np.isclose(imgtrans0, imgtrans0_m).all()
    assert np.isclose(imgtrans1, imgtrans1_m).all()
    assert np.isclose(imgden0, imgden0_m).all()
    assert np.isclose(imgden1, imgden1_m).all()
    assert np.isclose(scanOr, scanOr_m).all()
    assert np.isclose(scanAc, scanAc_m).all()
    assert np.isclose(stats, stats_m).all()

def test_final_alignment_rectangle_matrices_WITH_global_phase_corr_1en3_delta(small_delta_rectangle):

    im1, im2 = small_delta_rectangle
    scanAngles = (10, 100)

    sm = SPmerge01linear(scanAngles, im1, im2)
    sm = SPmerge02(sm, 2, 1)

    imgtrans0 = sm.imageTransform[0, ...]
    imgtrans1 = sm.imageTransform[1, ...]
    imgden0 = sm.imageDensity[0, ...]
    imgden1 = sm.imageDensity[1, ...]
    scanOr = sm.scanOr
    scanAc = sm.scanActive
    stats = sm.stats

    mfile = 'SPmerge02_final_alignment_rectangle_matrices_WITH_global_corr_simulated_10_100_1en3_delta.mat'
    mstruct = sio.loadmat('matlab_result/SPmerge02/'+mfile)
    imgtrans0_m = mstruct['it1']
    imgtrans1_m = mstruct['it2']
    imgden0_m = mstruct['id1']
    imgden1_m = mstruct['id2']
    scanOr_m = mstruct['scOr'].T - 1
    scanAc_m = mstruct['scAc'].T
    stats_m = mstruct['stats']

    assert np.isclose(imgtrans0, imgtrans0_m).all()
    assert np.isclose(imgtrans1, imgtrans1_m).all()
    assert np.isclose(imgden0, imgden0_m).all()
    assert np.isclose(imgden1, imgden1_m).all()
    assert np.isclose(scanOr, scanOr_m).all()
    assert np.isclose(scanAc, scanAc_m).all()
    assert np.isclose(stats, stats_m).all()
