import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio

from scanning_drift_corr.SPmerge01linear import SPmerge01linear

# read the simulated data
with h5py.File('../data_examples/nonlinear_drift_correction_synthetic_dataset_for_testing.mat', 'r') as f:
    image00deg = np.array(f['image00deg']).T
    image90deg = np.array(f['image90deg']).T
    imageIdeal = np.array(f['imageIdeal']).T

# the Python interface
scanAngles = (0, 90)
sm = SPmerge01linear(scanAngles, image00deg, image90deg)

# ------------------
# the following just compares MATLAB and Python results
# ------------------
# python results
img00deg_py = sm.imageTransform[0, ...]
img90deg_py = sm.imageTransform[1, ...]
dens00deg_py = sm.imageDensity[0, ...]
dens90deg_py = sm.imageDensity[1, ...]

# MATLAB results
fpath = 'scanning_drift_corr/tests/matlab_result/'
mstruct = sio.loadmat(fpath + 'SPmerge01linear_result_0_90_simulated_data.mat')
img00deg_m = mstruct['it1']
img90deg_m = mstruct['it2']
dens00deg_m = mstruct['id1']
dens90deg_m = mstruct['id2']

# show transformed image from python and MATLAB
fig, ax = plt.subplots(1,2)
ax[0].matshow(img00deg_py, cmap='gray')
ax[1].matshow(img00deg_m, cmap='gray')
ax[0].set_title('Python 0 deg')
ax[1].set_title('MATLAB 0 deg')

fig, ax = plt.subplots(1,2)
ax[0].matshow(img90deg_py, cmap='gray')
ax[1].matshow(img90deg_m, cmap='gray')
ax[0].set_title('Python 90 deg')
ax[1].set_title('MATLAB 90 deg')

# show their differences
# note the colour scale!
# beside the edges, the difference is from numeric noise
fig, ax = plt.subplots()
ax.matshow(np.abs(img00deg_py-img00deg_m), cmap='gray', vmax=1e-14)
ax.set_title('Intensity difference 0 deg')

fig, ax = plt.subplots()
ax.matshow(np.abs(img90deg_py-img90deg_m), cmap='gray', vmax=1e-14)
ax.set_title('Intensity difference 90 deg')

# show the image density differences
# the reason for differences in edges
fig, ax = plt.subplots()
ax.matshow(np.abs(dens00deg_py-dens00deg_m), cmap='gray')
ax.set_title('Image density difference 0 deg')

fig, ax = plt.subplots()
ax.matshow(np.abs(dens90deg_py-dens90deg_m), cmap='gray')
ax.set_title('Image density difference 90 deg')
