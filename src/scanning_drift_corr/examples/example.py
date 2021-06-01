import numpy as np
import matplotlib.pyplot as plt
import h5py

import scanning_drift_corr.api as sdc

# read the simulated data
with h5py.File('../tests/nonlinear_drift_correction_synthetic_dataset_for_testing.mat', 'r') as f:
    image00deg = np.array(f['image00deg']).T
    image90deg = np.array(f['image90deg']).T
    imageIdeal = np.array(f['imageIdeal']).T

# ===================================
# the Python interface
# ===================================
scanAngles = (0, 90)
sm = sdc.SPmerge01linear(scanAngles, image00deg, image90deg)

sm = sdc.SPmerge02(sm, 32, 8)

imageFinal, signalArray, densityArray = sdc.SPmerge03(sm)
# ===================================
# the Python interface
# ===================================

# compare with the ideal image
fig, ax = plt.subplots(1,2, figsize=(16,9))
ax[0].matshow(imageFinal, cmap='gray', vmin=5.5, vmax=7)
ax[1].matshow(imageIdeal, cmap='gray', vmin=5.5, vmax=7)
ax[0].set_title('Corrected image')
ax[1].set_title('Ideal image')

# estimated image on each scan
fig, ax = plt.subplots(1,2, figsize=(16,9))
ax[0].matshow(image00deg, cmap='gray')
ax[1].matshow(signalArray[0,...], cmap='gray')
ax[0].set_title('Original 0 deg image')
ax[1].set_title('Estimated 0 deg image')

fig, ax = plt.subplots(1,2, figsize=(16,9))
ax[0].matshow(image90deg, cmap='gray')
ax[1].matshow(signalArray[1,...], cmap='gray')
ax[0].set_title('Original 90 deg image')
ax[1].set_title('Estimated 90 deg image at 0 deg')

# estimated density on each scan
fig, ax = plt.subplots(1,2, figsize=(16,9))
ax[0].matshow(densityArray[0,...], cmap='gray')
ax[1].matshow(densityArray[1,...], cmap='gray')
ax[0].set_title('Estimated density at 0 deg')
ax[1].set_title('Estimated denstiy at 90 deg')
