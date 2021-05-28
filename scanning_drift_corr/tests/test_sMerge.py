import numpy as np
import scipy.io as sio
import pytest

from scanning_drift_corr.sMerge import sMerge


class TestInitialisation:

    def test_scanAngles_seq(self, dummy_sequential_sm):
        sm = dummy_sequential_sm
        assert sm.scanAngles[0] == 35
        assert sm.scanAngles[1] == 125
        assert sm.scanAngles[2] == 60
        assert sm.scanAngles[3] == 150
        assert sm.scanAngles.size == 4

    def test_numImages_seq(self, dummy_sequential_sm):
        sm = dummy_sequential_sm
        assert sm.numImages == 4

    def test_imageSize_seq(self, dummy_sequential_sm):
        sm = dummy_sequential_sm
        assert (sm.imageSize == [36, 36]).all()

    def test_scanLines_seq(self, dummy_sequential, dummy_sequential_sm):
        im1, im2, im3, im4 = dummy_sequential
        sm = dummy_sequential_sm
        assert (sm.scanLines[0,...] == im1).all()
        assert (sm.scanLines[1,...] == im2).all()
        assert (sm.scanLines[2,...] == im3).all()
        assert (sm.scanLines[3,...] == im4).all()
        assert sm.scanLines.shape == (4,*im1.shape)

    def test_scanOr_seq(self, dummy_sequential, dummy_sequential_sm):
        sm = dummy_sequential_sm

        mstruct = sio.loadmat('matlab_result/init_scanOr_idummy_sequential.mat')
        scanOr_m = mstruct['scOr'].T - 1

        assert sm.scanOr.shape == (4, 2, dummy_sequential[0].shape[0])
        assert np.isclose(sm.scanOr, scanOr_m).all()

    def test_scanOr_0_90_180_270_seq(self, dummy_sequential):
        scanAngles = (0, 90, 180, 270)
        sm = sMerge(scanAngles, dummy_sequential)

        mstruct = sio.loadmat('matlab_result/init_scanOr_0_90_180_270_dummy_sequential.mat')
        scanOr_m = mstruct['scOr'].T - 1

        assert np.isclose(sm.scanOr, scanOr_m).all()

    def test_scanDir_seq(self, dummy_sequential_sm):
        sm = dummy_sequential_sm
        assert np.isclose(np.around(sm.scanDir[0,:], 4), [-0.5736, 0.8192]).all()
        assert np.isclose(np.around(sm.scanDir[1,:], 4), [-0.8192, -0.5736]).all()
        assert np.isclose(np.around(sm.scanDir[2,:], 4), [-0.8660, 0.5000]).all()
        assert np.isclose(np.around(sm.scanDir[3,:], 4), [-0.5000, -0.8660]).all()

    def test_num_angles_images_not_match(self, dummy_sequential):
        scanAngles = (25, 115)

        with pytest.raises(ValueError):
            sMerge(scanAngles, dummy_sequential)

    def test_provided_images_shape_not_equal(self):
        im1 = np.arange(32*32, dtype=float).reshape(32,32)
        im2 = np.arange(33*33, dtype=float).reshape(33,33)
        scanAngles = (25, 115)

        with pytest.raises(ValueError):
            sMerge(scanAngles, (im1, im2))
