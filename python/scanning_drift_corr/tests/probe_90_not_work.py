import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy.io as sio

from scanning_drift_corr.sMerge import sMerge
from scanning_drift_corr.SPmakeImage import SPmakeImage, _bilinear_interpolation, _apply_KDE
from scanning_drift_corr.SPmerge01linear import SPmerge01linear

# easier debug with matlab
np.set_printoptions(precision=4, suppress=True, linewidth=np.nan)

with h5py.File('nonlinear_drift_correction_synthetic_dataset_for_testing.mat', 'r') as f:
    image00deg = np.array(f['image00deg']).T
    image90deg = np.array(f['image90deg']).T
    imageIdeal = np.array(f['imageIdeal']).T

# scanAngles = (0, 90)
# sm = sMerge(scanAngles, (image00deg, image90deg))

# scInit = sm.scanOr.copy()



# mstruct = sio.loadmat('matlab_result/xy_init.mat')
# xy_m = mstruct['xy'].T
# print(xy_m)

# assert 0

# # use a particular xyShift
# yDrift, xDrift = np.meshgrid(sm.linearSearch, sm.linearSearch)
# xyShift = np.hstack([sm.inds*xDrift[1,3], sm.inds*yDrift[1,3]])
# sm.scanOr[:2, ...] += xyShift.T



# def _rotate_local(sMerge, indImage, indLines):
#     """Bilinear interpolation, ported from MATLAB
#     """

#     nc = sMerge.nc

#     # Expand coordinates
#     t = np.arange(1, nc+1)
#     x0 = sMerge.scanOr[indImage, 0, indLines][:,None] 
#     y0 = sMerge.scanOr[indImage, 1, indLines][:,None] 

#     xInd = x0 + t * sMerge.scanDir[indImage, 0]
#     yInd = y0 + t * sMerge.scanDir[indImage, 1]

#     # print(x0)
#     # print(sMerge.scanDir[indImage, 1])
#     # print()


#     # Prevent pixels from leaving image boundaries
#     # in MATLAB is column vector, here is row vector
#     # cap at indices 1 lower than MATLAB
#     xInd = np.clip(xInd, 0, sMerge.imageSize[0]-2).ravel()
#     yInd = np.clip(yInd, 0, sMerge.imageSize[1]-2).ravel()

#     # Convert to bilinear interpolants and weights
#     # xAll/yAll have 4 rows, each represent the interpolants of the pixel of
#     # the image which as are column vec (column size is raw data size)
#     xIndF = np.floor(xInd).astype(int)
#     yIndF = np.floor(yInd).astype(int)
    
    
#     xAll = np.vstack([xIndF, xIndF+1, xIndF, xIndF+1])
#     yAll = np.vstack([yIndF, yIndF, yIndF+1, yIndF+1])
    
#     dx = xInd - xIndF
#     dy = yInd - yIndF
    

#     # indAll in MATLAB is from sub2ind
#     w = np.vstack([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy])
#     # w = np.vstack([dx*dy, dy*(1-dx), (1-dy)*dx, (1-dx)*(1-dy)])
    
    
    
    
#     indAll = np.ravel_multi_index((xAll, yAll), sMerge.imageSize)
    
#     # indAll += 1

#     # raw image
#     sL = sMerge.scanLines[indImage, indLines, :]

#     # weigh the raw image for interpolation
#     wsig = w * sL.ravel()
#     wcount = w

#     # Generate image
#     sig = np.bincount(indAll.ravel(), weights=wsig.ravel(),
#                       minlength=sMerge.imageSize.prod()).reshape(sMerge.imageSize)
#     count = np.bincount(indAll.ravel(), weights=wcount.ravel(),
#                       minlength=sMerge.imageSize.prod()).reshape(sMerge.imageSize)

#     return sig, count
    
    # return sL, indAll, w



# indLines = np.ones(sm.nr, dtype=bool)

# sL1, indAll1, w1 = _rotate_local(sm, 1, indLines)

# # MATLAB results
# mstruct = sio.loadmat('matlab_result/SPmakeimage_sL_indAll_w_ang30_simu.mat')
# sL1_m = mstruct['sL']
# indAll1_m = mstruct['indAll'].T
# w1_m = mstruct['w'].T


# fig, ax = plt.subplots()
# ax.matshow(w1, cmap='gray')
# fig, ax = plt.subplots()
# ax.matshow(w1_m, cmap='gray')

# perform bilinear interpolation and apply KDE
# indLines = np.ones(sm.nr, dtype=bool)
# sig0, count0 = _rotate_local(sm, 0, indLines)
# sig1, count1 = _rotate_local(sm, 1, indLines)


# # ../scan_distortion/scan_distortion/tests/
# # MATLAB results
# mstruct = sio.loadmat('matlab_result/SPmakeimage_rotated_sig_count_ang0_simu.mat')
# sig0_m = mstruct['sig']
# count0_m = mstruct['count']
# mstruct = sio.loadmat('matlab_result/SPmakeimage_rotated_sig_count_ang91_simu.mat')
# sig1_m = mstruct['sig']
# count1_m = mstruct['count']




# # fig, ax = plt.subplots()
# # ax.matshow(np.abs(sig0-sig0_m), cmap='gray')
# print(np.abs(sig0-sig0_m).max())


# # fig, ax = plt.subplots()
# # ax.matshow(np.abs(sig1-sig1_m), cmap='gray')
# print(np.abs(sig1-sig1_m).max())


eps = np.finfo(float).eps
scanAngles = (0, 90)

sm = SPmerge01linear(scanAngles, image00deg, image90deg)




imgtrans0 = sm.imageTransform[0, ...]
imgtrans1 = sm.imageTransform[1, ...]
imgden0 = sm.imageDensity[0, ...]
imgden1 = sm.imageDensity[1, ...]
scanOr = sm.scanOr

mstruct = sio.loadmat('matlab_result/SPmerge01linear_result_0_90_simulated_data.mat')
imgtrans0_m = mstruct['it1']
imgtrans1_m = mstruct['it2']
imgden0_m = mstruct['id1']
imgden1_m = mstruct['id2']
scanOr_m = mstruct['scOr'].T - 1


bd = int(0.25*512)

fig, ax = plt.subplots()
ax.matshow(np.abs(imgtrans0[bd:-bd, bd:-bd]-imgtrans0_m[bd:-bd, bd:-bd]), cmap='gray')
# print(np.abs(sig0-sig0_m).max())

print(np.abs(imgtrans0[bd:-bd, bd:-bd]-imgtrans0_m[bd:-bd, bd:-bd]).max())
print(np.abs(imgtrans1[bd:-bd, bd:-bd]-imgtrans1_m[bd:-bd, bd:-bd]).max())

fig, ax = plt.subplots()
ax.matshow(np.abs(imgtrans0-imgtrans0_m), cmap='gray', vmax=1e-14)

fig, ax = plt.subplots()
ax.matshow(np.abs(imgtrans1-imgtrans1_m), cmap='gray', vmax=1e-14)

