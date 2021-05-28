import numpy as np
import scipy.io as sio
import pytest

from scanning_drift_corr.SPmerge01linear import SPmerge01linear, _get_linear_drift
from scanning_drift_corr.sMerge import sMerge
from scanning_drift_corr.tools import _hanning_weight

def test_no_provided_image():
    scanAngles = (30, 120)
    sm = SPmerge01linear(scanAngles)

    assert sm is None

def test_result_small_delta(small_delta_matrix):
    im1, im2 = small_delta_matrix
    scanAngles = (0, 90)

    sm = SPmerge01linear(scanAngles, im1, im2)
    imgtrans0 = sm.imageTransform[0, ...]
    imgtrans1 = sm.imageTransform[1, ...]
    imgden0 = sm.imageDensity[0, ...]
    imgden1 = sm.imageDensity[1, ...]
    scanOr = sm.scanOr

    mstruct = sio.loadmat('matlab_result/SPmerge01linear_result_small_delta.mat')
    imgtrans0_m = mstruct['it1']
    imgtrans1_m = mstruct['it2']
    imgden0_m = mstruct['id1']
    imgden1_m = mstruct['id2']
    scanOr_m = mstruct['scOr'].T - 1

    assert np.isclose(imgtrans0, imgtrans0_m).all()
    assert np.isclose(imgtrans1, imgtrans1_m).all()
    assert np.isclose(imgden0, imgden0_m).all()
    assert np.isclose(imgden1, imgden1_m).all()
    assert np.isclose(scanOr, scanOr_m).all()

def test_result_small_delta_stack(small_delta_matrix_stack):
    stack = small_delta_matrix_stack
    scanAngles = (0, 90)

    sm = SPmerge01linear(scanAngles, stack)
    imgtrans0 = sm.imageTransform[0, ...]
    imgtrans1 = sm.imageTransform[1, ...]
    imgden0 = sm.imageDensity[0, ...]
    imgden1 = sm.imageDensity[1, ...]
    scanOr = sm.scanOr

    mstruct = sio.loadmat('matlab_result/SPmerge01linear_result_small_delta.mat')
    imgtrans0_m = mstruct['it1']
    imgtrans1_m = mstruct['it2']
    imgden0_m = mstruct['id1']
    imgden1_m = mstruct['id2']
    scanOr_m = mstruct['scOr'].T - 1

    assert np.isclose(imgtrans0, imgtrans0_m).all()
    assert np.isclose(imgtrans1, imgtrans1_m).all()
    assert np.isclose(imgden0, imgden0_m).all()
    assert np.isclose(imgden1, imgden1_m).all()
    assert np.isclose(scanOr, scanOr_m).all()

@pytest.mark.parametrize('angles, mfile',
                         [((45, 135), 'SPmerge01linear_result_45_135_simulated_data.mat'),
                          ((1, 91), 'SPmerge01linear_result_1_91_simulated_data.mat'),
                          ((0, 90), 'SPmerge01linear_result_0_90_simulated_data.mat'),
                          ((-90, -180), 'SPmerge01linear_result_m90_m180_simulated_data.mat'),
                          ])
def test_result_ignore_edge_simulated_data(angles, mfile, MATLAB_simulated_images):

    im1, im2 = MATLAB_simulated_images
    scanAngles = angles
    bd = int(0.25*im1.shape[0]) # the start and end indices for comparison

    sm = SPmerge01linear(scanAngles, im1, im2)
    imgtrans0 = sm.imageTransform[0, ...]
    imgtrans1 = sm.imageTransform[1, ...]
    imgden0 = sm.imageDensity[0, ...]
    imgden1 = sm.imageDensity[1, ...]
    scanOr = sm.scanOr

    mstruct = sio.loadmat('matlab_result/'+mfile)
    imgtrans0_m = mstruct['it1']
    imgtrans1_m = mstruct['it2']
    imgden0_m = mstruct['id1']
    imgden1_m = mstruct['id2']
    scanOr_m = mstruct['scOr'].T - 1

    assert np.isclose(imgtrans0[bd:-bd, bd:-bd], imgtrans0_m[bd:-bd, bd:-bd]).all()
    assert np.isclose(imgtrans1[bd:-bd, bd:-bd], imgtrans1_m[bd:-bd, bd:-bd]).all()
    assert np.isclose(imgden0[bd:-bd, bd:-bd], imgden0_m[bd:-bd, bd:-bd]).all()
    assert np.isclose(imgden1[bd:-bd, bd:-bd], imgden1_m[bd:-bd, bd:-bd]).all()
    assert np.isclose(scanOr, scanOr_m).all()

def test_result_ignore_edge_4_images_simulated_data(MATLAB_simulated_images):

    im1, im2 = MATLAB_simulated_images
    scanAngles = (0, 90, 45, 135)
    bd = int(0.25*im1.shape[0]) # the start and end indices for comparison

    sm = SPmerge01linear(scanAngles, im1, im2, im1, im2, parallel=False)
    imgtrans0 = sm.imageTransform[0, ...]
    imgtrans1 = sm.imageTransform[1, ...]
    imgtrans2 = sm.imageTransform[2, ...]
    imgtrans3 = sm.imageTransform[3, ...]
    imgden0 = sm.imageDensity[0, ...]
    imgden1 = sm.imageDensity[1, ...]
    imgden2 = sm.imageDensity[2, ...]
    imgden3 = sm.imageDensity[3, ...]
    scanOr = sm.scanOr

    mstruct = sio.loadmat('matlab_result/'+'SPmerge01linear_result_0_90_45_135_simulated_data.mat')
    imgtrans0_m = mstruct['it1']
    imgtrans1_m = mstruct['it2']
    imgtrans2_m = mstruct['it3']
    imgtrans3_m = mstruct['it4']
    imgden0_m = mstruct['id1']
    imgden1_m = mstruct['id2']
    imgden2_m = mstruct['id3']
    imgden3_m = mstruct['id4']
    scanOr_m = mstruct['scOr'].T - 1

    assert np.isclose(imgtrans0[bd:-bd, bd:-bd], imgtrans0_m[bd:-bd, bd:-bd]).all()
    assert np.isclose(imgtrans1[bd:-bd, bd:-bd], imgtrans1_m[bd:-bd, bd:-bd]).all()
    assert np.isclose(imgtrans2[bd:-bd, bd:-bd], imgtrans2_m[bd:-bd, bd:-bd]).all()
    assert np.isclose(imgtrans3[bd:-bd, bd:-bd], imgtrans3_m[bd:-bd, bd:-bd]).all()
    assert np.isclose(imgden0[bd:-bd, bd:-bd], imgden0_m[bd:-bd, bd:-bd]).all()
    assert np.isclose(imgden1[bd:-bd, bd:-bd], imgden1_m[bd:-bd, bd:-bd]).all()
    assert np.isclose(imgden2[bd:-bd, bd:-bd], imgden2_m[bd:-bd, bd:-bd]).all()
    assert np.isclose(imgden3[bd:-bd, bd:-bd], imgden3_m[bd:-bd, bd:-bd]).all()
    assert np.isclose(scanOr, scanOr_m).all()

def test_result_small_delta_rectangle(small_delta_rectangle):
    im1, im2 = small_delta_rectangle
    scanAngles = (10, 100)

    sm = SPmerge01linear(scanAngles, im1, im2)
    imgtrans0 = sm.imageTransform[0, ...]
    imgtrans1 = sm.imageTransform[1, ...]
    imgden0 = sm.imageDensity[0, ...]
    imgden1 = sm.imageDensity[1, ...]
    scanOr = sm.scanOr

    mstruct = sio.loadmat('matlab_result/SPmerge01linear_result_small_delta_rectangle.mat')
    imgtrans0_m = mstruct['it1']
    imgtrans1_m = mstruct['it2']
    imgden0_m = mstruct['id1']
    imgden1_m = mstruct['id2']
    scanOr_m = mstruct['scOr'].T - 1

    assert np.isclose(imgtrans0, imgtrans0_m).all()
    assert np.isclose(imgtrans1, imgtrans1_m, atol=1e-4).all() # weird!
    assert np.isclose(imgden0, imgden0_m).all()
    assert np.isclose(imgden1, imgden1_m).all()
    assert np.isclose(scanOr, scanOr_m).all()

def test_get_linear_drift(small_delta_matrix):
    im1, im2 = small_delta_matrix

    # set default values or from input arguments
    linearSearch = np.linspace(-0.02, 0.02, num=2*2+1)
    paddingScale = 1.125
    flagReportProgress = True
    parallel = True
    niter = 2

    # initialise the sMerge object
    scanAngles = (0, 90)
    sm = sMerge(scanAngles, (im1, im2),  paddingScale=paddingScale)

    # get testing linear drifts
    linearSearch *= sm.nr
    inds = np.linspace(-0.5, 0.5, num=sm.nr)[:, None]

    # get linear drift, using the first two images
    xdrift, ydrift = _get_linear_drift(sm, linearSearch, flagReportProgress,
                                       parallel, inds, niter)

    mstruct = sio.loadmat('matlab_result/sMerge_linalig_small_delta_provide_xycoord.mat')
    score_m = mstruct['score']
    xdrift_m, ydrift_m = mstruct['drifts'][0]

    assert np.isclose(sm.linearSearchScores[-1,...], score_m).all()
    assert np.isclose(xdrift, xdrift_m).all()
    assert np.isclose(ydrift, ydrift_m).all()

def test_hanning_weights(small_delta_matrix_sm):
    sm = small_delta_matrix_sm

    padxy = sm.imageSize - sm.img_shape
    nr, nc = sm.img_shape
    hw = _hanning_weight(nr, nc, padxy)

    mstruct = sio.loadmat('matlab_result/sMerge_hanning_weights_small_delta.mat')
    hw_m = mstruct['w2']

    assert np.isclose(hw, hw_m).all()
