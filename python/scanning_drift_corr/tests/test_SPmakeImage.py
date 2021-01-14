import numpy as np
import scipy.io as sio
import pytest

from scanning_drift_corr.sMerge import sMerge
from scanning_drift_corr.SPmakeImage import SPmakeImage, _bilinear_interpolation, _apply_KDE

@pytest.mark.parametrize('angles, mfile',
                         [((10, 100), 'SPmakeimage_result_10_100_small_delta.mat'),
                          ((-20, 70), 'SPmakeimage_result_-20_70_small_delta.mat'),
                          pytest.param((30, 120), 'SPmakeimage_result_30_120_small_delta.mat', marks=pytest.mark.xfail(strict=True)),
                          ])
def test_transformed_image_small_delta_matrix(angles, mfile, small_delta_matrix):
    scanAngles = angles
    sm = sMerge(scanAngles, small_delta_matrix)

    # use a particular xyShift [1,3], match the testSPmakeimage func (2,4)
    yDrift, xDrift = np.meshgrid(sm.linearSearch, sm.linearSearch)
    xyShift = np.hstack([sm.inds*xDrift[1,3], sm.inds*yDrift[1,3]])
    sm.scanOr[:2, ...] += xyShift.T

    # Generate trial images
    sm = SPmakeImage(sm, 0)
    sm = SPmakeImage(sm, 1)

    # Python results
    imgtrans0 = sm.imageTransform[0, ...]
    imgtrans1 = sm.imageTransform[1, ...]
    imgden0 = sm.imageDensity[0, ...]
    imgden1 = sm.imageDensity[1, ...]

    # MATLAB results
    mstruct = sio.loadmat('matlab_result/'+mfile)
    imgtrans0_m = mstruct['it1']
    imgtrans1_m = mstruct['it2']
    imgden0_m = mstruct['id1']
    imgden1_m = mstruct['id2']

    assert np.isclose(imgtrans0, imgtrans0_m).all()
    assert np.isclose(imgtrans1, imgtrans1_m).all()
    assert np.isclose(imgden0, imgden0_m).all()
    assert np.isclose(imgden1, imgden1_m).all()

def test_after_bilinear_interpolation_angle_30_120(small_delta_matrix):
    scanAngles = (30, 120)
    sm = sMerge(scanAngles, small_delta_matrix)

    # use a particular xyShift [1,3], match the testSPmakeimage func (2,4)
    yDrift, xDrift = np.meshgrid(sm.linearSearch, sm.linearSearch)
    xyShift = np.hstack([sm.inds*xDrift[1,3], sm.inds*yDrift[1,3]])
    sm.scanOr[:2, ...] += xyShift.T

    # perform bilinear interpolation
    indLines = np.ones(sm.nr, dtype=bool)
    sig0, count0 = _bilinear_interpolation(sm, 0, indLines)
    sig1, count1 = _bilinear_interpolation(sm, 1, indLines)

    # MATLAB results
    mstruct = sio.loadmat('matlab_result/SPmakeimage_rotated_sig_count_ang30_small_delta.mat')
    sig0_m = mstruct['sig']
    count0_m = mstruct['count']
    mstruct = sio.loadmat('matlab_result/SPmakeimage_rotated_sig_count_ang120_small_delta.mat')
    sig1_m = mstruct['sig']
    count1_m = mstruct['count']

    assert np.isclose(sig0, sig0_m).all()
    assert np.isclose(count0, count0_m).all()
    assert np.isclose(sig1, sig1_m).all()
    assert np.isclose(count1, count1_m).all()

def test_after_bilinear_interpolation_KDE_angle_30_120(small_delta_matrix):
    scanAngles = (30, 120)
    sm = sMerge(scanAngles, small_delta_matrix)

    # use a particular xyShift [1,3], match the testSPmakeimage func (2,4)
    yDrift, xDrift = np.meshgrid(sm.linearSearch, sm.linearSearch)
    xyShift = np.hstack([sm.inds*xDrift[1,3], sm.inds*yDrift[1,3]])
    sm.scanOr[:2, ...] += xyShift.T

    # perform bilinear interpolation and apply KDE
    indLines = np.ones(sm.nr, dtype=bool)
    sig0, count0 = _bilinear_interpolation(sm, 0, indLines)
    sig0, count0 = _apply_KDE(sm, sig0, count0)
    sig1, count1 = _bilinear_interpolation(sm, 1, indLines)
    sig1, count1 = _apply_KDE(sm, sig1, count1)

    # MATLAB results
    mstruct = sio.loadmat('matlab_result/SPmakeimage_rotated_KDE_sig_count_ang30_small_delta.mat')
    sig0_m = mstruct['sig']
    count0_m = mstruct['count']
    mstruct = sio.loadmat('matlab_result/SPmakeimage_rotated_KDE_sig_count_ang120_small_delta.mat')
    sig1_m = mstruct['sig']
    count1_m = mstruct['count']

    assert np.isclose(sig0, sig0_m).all()
    assert np.isclose(count0, count0_m).all()
    assert np.isclose(sig1, sig1_m).all()
    assert np.isclose(count1, count1_m).all()

"""direct reason of failing test_transformed_image_small_delta_matrix with (30, 120)"""
@pytest.mark.xfail(strict=True)
def test_sub_angle_30_120(small_delta_matrix):
    scanAngles = (30, 120)
    sm = sMerge(scanAngles, small_delta_matrix)

    # use a particular xyShift [1,3], match the testSPmakeimage func (2,4)
    yDrift, xDrift = np.meshgrid(sm.linearSearch, sm.linearSearch)
    xyShift = np.hstack([sm.inds*xDrift[1,3], sm.inds*yDrift[1,3]])
    sm.scanOr[:2, ...] += xyShift.T

    # perform bilinear interpolation
    indLines = np.ones(sm.nr, dtype=bool)
    sig0, count0 = _bilinear_interpolation(sm, 0, indLines)
    sig0, count0 = _apply_KDE(sm, sig0, count0)
    sub0 = count0 > 0
    sig1, count1 = _bilinear_interpolation(sm, 1, indLines)
    sig1, count1 = _apply_KDE(sm, sig1, count1)
    sub1 = count1 > 0

    # MATLAB results
    mstruct = sio.loadmat('matlab_result/SPmakeimage_sub_ang30_small_delta.mat')
    sub0_m = mstruct['sub']
    mstruct = sio.loadmat('matlab_result/SPmakeimage_sub_ang120_small_delta.mat')
    sub1_m = mstruct['sub']

    assert np.isclose(sub0, sub0_m).all()
    assert np.isclose(sub1, sub1_m).all()

@pytest.mark.parametrize('angles, mfile',
                         [((0, 90), ('SPmakeimage_rotated_sig_count_ang0_simu.mat', 'SPmakeimage_rotated_sig_count_ang90_simu.mat')),
                          ((125, 35), ('SPmakeimage_rotated_sig_count_ang125_simu.mat', 'SPmakeimage_rotated_sig_count_ang35_simu.mat')),
                          ])
def test_after_bilinear_interpolation_simulated_data(angles, mfile, MATLAB_simulated_images):
    scanAngles = angles
    sm = sMerge(scanAngles, MATLAB_simulated_images)

    # use a particular xyShift [1,3], match the testSPmakeimage func (2,4)
    yDrift, xDrift = np.meshgrid(sm.linearSearch, sm.linearSearch)
    xyShift = np.hstack([sm.inds*xDrift[1,3], sm.inds*yDrift[1,3]])
    sm.scanOr[:2, ...] += xyShift.T

    # perform bilinear interpolation and apply KDE
    indLines = np.ones(sm.nr, dtype=bool)
    sig0, count0 = _bilinear_interpolation(sm, 0, indLines)
    sig1, count1 = _bilinear_interpolation(sm, 1, indLines)

    # MATLAB results
    m1, m2 = mfile
    mstruct = sio.loadmat('matlab_result/'+m1)
    sig0_m = mstruct['sig']
    count0_m = mstruct['count']
    mstruct = sio.loadmat('matlab_result/'+m2)
    sig1_m = mstruct['sig']
    count1_m = mstruct['count']

    assert np.isclose(sig0, sig0_m).all()
    assert np.isclose(count0, count0_m).all()
    assert np.isclose(sig1, sig1_m).all()
    assert np.isclose(count1, count1_m).all()

@pytest.mark.parametrize('angles, mfile',
                         [((30, 120), ('SPmakeimage_rotated_KDE_sig_count_ang30_simu.mat', 'SPmakeimage_rotated_KDE_sig_count_ang120_simu.mat')),
                          ((-50, 40), ('SPmakeimage_rotated_KDE_sig_count_angm50_simu.mat', 'SPmakeimage_rotated_KDE_sig_count_ang40_simu.mat')),
                          ])
def test_after_bilinear_interpolation_KDE_simulated_data(angles, mfile, MATLAB_simulated_images):
    scanAngles = angles
    sm = sMerge(scanAngles, MATLAB_simulated_images)

    # use a particular xyShift [1,3], match the testSPmakeimage func (2,4)
    yDrift, xDrift = np.meshgrid(sm.linearSearch, sm.linearSearch)
    xyShift = np.hstack([sm.inds*xDrift[1,3], sm.inds*yDrift[1,3]])
    sm.scanOr[:2, ...] += xyShift.T

    # perform bilinear interpolation and apply KDE
    indLines = np.ones(sm.nr, dtype=bool)
    sig0, count0 = _bilinear_interpolation(sm, 0, indLines)
    sig0, count0 = _apply_KDE(sm, sig0, count0)
    sig1, count1 = _bilinear_interpolation(sm, 1, indLines)
    sig1, count1 = _apply_KDE(sm, sig1, count1)

    # MATLAB results
    m1, m2 = mfile
    mstruct = sio.loadmat('matlab_result/'+m1)
    sig0_m = mstruct['sig']
    count0_m = mstruct['count']
    mstruct = sio.loadmat('matlab_result/'+m2)
    sig1_m = mstruct['sig']
    count1_m = mstruct['count']

    assert np.isclose(sig0, sig0_m).all()
    assert np.isclose(count0, count0_m).all()
    assert np.isclose(sig1, sig1_m).all()
    assert np.isclose(count1, count1_m).all()
