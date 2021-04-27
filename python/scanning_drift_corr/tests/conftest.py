import numpy as np
import h5py
import pytest

from scanning_drift_corr.sMerge import sMerge

@pytest.fixture
def dummy_sequential():
    im1 = np.arange(32*32, dtype=float).reshape(32,32)
    im2 = im1 * 3
    im3 = im1[::-1, :]*2
    im4 = im1[:, ::-1]*4

    return im1, im2, im3, im4

@pytest.fixture
def dummy_sequential_sm(dummy_sequential):
    scanAngles = (35, 125, 60, 150)
    sm = sMerge(scanAngles, dummy_sequential)

    return sm

@pytest.fixture
def small_delta_matrix():
    # angle [0,90]
    n = 9

    im1 = np.ones((n,n))
    im1[4,6] = 10
    im1[2,3] = 10

    im2 = np.ones((n,n))
    im2[5,2] = 10
    im2[2,4] = 10

    return im1, im2

@pytest.fixture
def small_delta_matrix_sm(small_delta_matrix):
    scanAngles = (0, 90)
    sm = sMerge(scanAngles, small_delta_matrix)

    return sm

@pytest.fixture
def small_delta_matrix_stack():
    # angle [0,90]
    n = 9

    im1 = np.ones((n,n))
    im1[4,6] = 10
    im1[2,3] = 10

    im2 = np.ones((n,n))
    im2[5,2] = 10
    im2[2,4] = 10

    stack = np.empty((2, 9, 9))
    stack[0, ...] = im1
    stack[1, ...] = im2

    return stack

@pytest.fixture
def MATLAB_simulated_images():
    with h5py.File('nonlinear_drift_correction_synthetic_dataset_for_testing.mat', 'r') as f:
        image00deg = np.array(f['image00deg']).T
        image90deg = np.array(f['image90deg']).T
        #imageIdeal = np.array(f['imageIdeal']).T

    return image00deg, image90deg

@pytest.fixture
def MATLAB_simulated_images_sm(MATLAB_simulated_images):
    scanAngles = (0, 90)
    sm = sMerge(scanAngles, MATLAB_simulated_images)

    return sm

@pytest.fixture
def MATLAB_simulated_images_stack():
    with h5py.File('nonlinear_drift_correction_synthetic_dataset_for_testing.mat', 'r') as f:
        image00deg = np.array(f['image00deg']).T
        image90deg = np.array(f['image90deg']).T
        #imageIdeal = np.array(f['imageIdeal']).T

    stack = np.empty((2, 512, 512))
    stack[0, ...] = image00deg
    stack[1, ...] = image90deg

    return stack

@pytest.fixture
def small_delta_rectangle():
    # angle [0,90]
    rect1 = np.ones((4,10))
    rect2 = np.ones((4,10))
    rect1[2,5] = 10
    rect1[3,8] = 10
    rect2[1,5] = 10
    rect2[0,2] = 10

    return rect1, rect2

@pytest.fixture
def small_delta_rectangle_sm(small_delta_rectangle):
    scanAngles = (0, 90)
    sm = sMerge(scanAngles, small_delta_rectangle)

    return sm
