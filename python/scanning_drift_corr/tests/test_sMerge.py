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
        im1 = np.arange(32*32, dtype=np.float).reshape(32,32)
        im2 = np.arange(33*33, dtype=np.float).reshape(33,33)       
        scanAngles = (25, 115)
        
        with pytest.raises(ValueError):
            sMerge(scanAngles, (im1, im2))    


class TestLinearAlignmentSmallDeltaMatrix:
    
    def test_return_index_true(self, small_delta_matrix_sm):
        sm = small_delta_matrix_sm
        score, xInd, yInd = sm.linear_alignment(return_index=True)
        
        mstruct = sio.loadmat('matlab_result/sMerge_linalig_small_delta_index.mat')
        score_m = mstruct['score']
        xInd_m = mstruct['xInd']
        yInd_m = mstruct['yInd']
                    
        assert np.isclose(score, score_m).all()
        assert xInd == xInd_m - 1
        assert yInd == yInd_m - 1

    def test_return_index_false(self, small_delta_matrix_sm):
        sm = small_delta_matrix_sm
        score, xdrift, ydrift = sm.linear_alignment(return_index=False)
        
        mstruct = sio.loadmat('matlab_result/sMerge_linalig_small_delta_noindex.mat')
        score_m = mstruct['score']
        xdrift_m, ydrift_m = mstruct['drifts'][0]
        
        assert np.isclose(score, score_m).all()
        assert np.isclose(xdrift, xdrift_m).all()
        assert np.isclose(ydrift, ydrift_m).all()

    def test_provide_xycoord(self, small_delta_matrix_sm):
        sm = small_delta_matrix_sm
        
        # do first and second alignment
        # easier comparison with MATLAB
        score1, xInd, yInd = sm.linear_alignment(return_index=True)
        sm.linearSearchScore1 = score1

        step = np.diff(sm.linearSearch)[0]
        xRefine = sm.linearSearch[xInd] + np.linspace(-0.5, 0.5, 
                                                      sm.linearSearch.size)*step
        yRefine = sm.linearSearch[yInd] + np.linspace(-0.5, 0.5, 
                                                      sm.linearSearch.size)*step

        score2, xdrift, ydrift = sm.linear_alignment(xcoord=xRefine, ycoord=yRefine)

        mstruct = sio.loadmat('matlab_result/sMerge_linalig_small_delta_provide_xycoord.mat')
        score_m = mstruct['score']
        xdrift_m, ydrift_m = mstruct['drifts'][0]
        
        assert np.isclose(score2, score_m).all()
        assert np.isclose(xdrift, xdrift_m).all()
        assert np.isclose(ydrift, ydrift_m).all()

    def test_progress_bar_true(self, small_delta_matrix_sm):
        sm = small_delta_matrix_sm
        sm.flagReportProgress = True
        _ = sm.linear_alignment(return_index=False)

def test_hanning_weights(small_delta_matrix_sm):
    sm = small_delta_matrix_sm
    hw = sm._hanning_weight()

    mstruct = sio.loadmat('matlab_result/sMerge_hanning_weights_small_delta.mat')
    hw_m = mstruct['w2']

    assert np.isclose(hw, hw_m).all()

class TestApplyLinearDirftSmallDeltaMatrix:
            
    
    def test_set_xyshift_value(self, small_delta_matrix_sm):
        sm = small_delta_matrix_sm
        sm.xyLinearDrift = np.array([-0.02, 0.15])
        
        sm.apply_linear_drift()
        
        mstruct = sio.loadmat('matlab_result/sMerge_apply_linear_drift_small_delta.mat')
        scanOr0_m = mstruct['scanOr1'].T - 1
        scanOr1_m = mstruct['scanOr2'].T - 1
        imgtrans0_m = mstruct['imgtrans1']
        imgtrans1_m = mstruct['imgtrans2']
        
        assert np.isclose(sm.scanOr[0, ...], scanOr0_m).all()
        assert np.isclose(sm.scanOr[1, ...], scanOr1_m).all()
        assert np.isclose(sm.imageTransform[0, ...], imgtrans0_m).all()
        assert np.isclose(sm.imageTransform[1, ...], imgtrans1_m).all()
        
    def test_no_xylineardrift(self, small_delta_matrix_sm):
        sm = small_delta_matrix_sm
        
        with pytest.raises(ValueError):
            sm.apply_linear_drift()
            

class TestEstimateInitialAlignmentSmallDeltaMatrix:
            
    def test_set_xyshift_value(self, small_delta_matrix_sm):
        sm = small_delta_matrix_sm
        sm.xyLinearDrift = np.array([-0.02, 0.15])
        
        sm.apply_linear_drift()
        dxy = sm.estimate_initial_alignment()

        mstruct = sio.loadmat('matlab_result/sMerge_estimated_alignmment_small_delta.mat')
        dxy_m = mstruct['dxy']  
    
        assert np.isclose(dxy, dxy_m).all()
    
    def test_apply_estimated_alignment(self, small_delta_matrix_sm):
        sm = small_delta_matrix_sm
        sm.xyLinearDrift = np.array([-0.02, 0.15])
        
        sm.apply_linear_drift()
        dxy = sm.estimate_initial_alignment()
        sm.apply_estimated_alignment(dxy)

        mstruct = sio.loadmat('matlab_result/sMerge_apply_estimated_alignment_small_delta.mat')
        imgtrans0_m = mstruct['imgtrans1']
        imgtrans1_m = mstruct['imgtrans2']

        assert np.isclose(sm.imageTransform[0, ...], imgtrans0_m).all()
        assert np.isclose(sm.imageTransform[1, ...], imgtrans1_m).all()

def test_plot_linear_drift_correction(small_delta_matrix_sm):
    sm = small_delta_matrix_sm
    sm.xyLinearDrift = np.array([-0.02, 0.15])
    
    sm.apply_linear_drift()
    dxy = sm.estimate_initial_alignment()
    sm.apply_estimated_alignment(dxy)
    
    sm.plot_linear_drift_correction()
