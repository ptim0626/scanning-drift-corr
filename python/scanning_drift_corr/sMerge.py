"""The file contains the sMerge class
"""

from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
from scanning_drift_corr.SPmakeImage import SPmakeImage

class sMerge:
    """The sMerge class, corresponds to the sMerge struct
    """

    def __init__(self, scanAngles, images, paddingScale=1.125,
                 flagReportProgress=False):

        self.KDEsigma = 1 / 2
        self.edgeWidth = 1 / 128
        self.linearSearch = np.linspace(-0.02, 0.02, num=2*2+1)
        self.flagReportProgress = flagReportProgress

        self._input_validation(scanAngles, images)
        self.nr, self.nc = self.img_shape

        self.imageSize = np.floor(self.img_shape * paddingScale/4 + 0.5).astype(int) * 4

        # coordinates of origin of rows
        # second dim is coordinate (row, col)
        # third dim is number of rows
        self.scanOr = np.zeros((self.numImages, 2, self.nr))

        # scan direction
        self.scanDir = np.zeros((self.numImages, 2))

        # save raw data to scanLines
        if self.isStack:
            # TODO: ensure navigation index first
            self.scanLines = images
        else:
            self.scanLines = np.empty((self.numImages, *self.img_shape))
            for k, im in enumerate(images):
                self.scanLines[k, :, :] = im

        self._set_scanOr_scanDir()

        self.imageTransform = np.zeros((self.numImages, *self.imageSize))
        self.imageDensity = np.zeros((self.numImages, *self.imageSize))
        self.inds = np.linspace(-0.5, 0.5, num=self.nr)[:, None]
        self.linearSearch *= self.nr
        self.linearSearchScores = None
        self.xyLinearDrift = None
        self.ref = np.floor(self.imageSize/2 + 0.5).astype(int) - 1
        self.scanActive = None
        self.imageRef = None
        self.stats = None

    def _input_validation(self, scanAngles, images):
        """Determine whether provided images is a stack, the shapes of them
        and number of images, and some checks on input data
        """

        # ensure tuple if not passed from multiple args
        images = tuple(images)

        if len(images) == 1:
            # 3D stack, navigation index first
            images = images[0]
            if images.ndim != 3:
                raise ValueError('A stack of image is expected.')

            self.isStack = True
            self.img_shape = np.asarray(images.shape[1:])
            self.numImages = images.shape[0]
        if len(images) > 1:
            # image sequence
            shapes = np.asarray([arr.shape for arr in images])
            shape_equal = (shapes[0,0] == shapes[:, 0]).all() &\
                (shapes[0,1] == shapes[:, 1]).all()
            if not shape_equal:
                raise ValueError('The provided images are not of the same shape')

            self.isStack = False
            self.img_shape = shapes[0,:]
            self.numImages = len(images)

        self.scanAngles = np.asarray(scanAngles)
        if self.scanAngles.size != self.numImages:
            msg = ('The number of scanning angles ({}) does not match the '
                   'number of images ({})')
            raise ValueError(msg.format(self.scanAngles.size, self.numImages))

        return

    def _set_scanOr_scanDir(self):
        """Set pixel origins and scan direction
        """

        scanAngles_rad = np.deg2rad(self.scanAngles)

        for k in range(self.numImages):
            # initialise origins of rows
            # zero in Python
            xy = np.zeros((2, self.nr))
            xy[0, :] = np.arange(self.nr)

            # coordinates offset by half before rotation
            xy[0, :] -= self.img_shape[0] / 2
            xy[1, :] -= self.img_shape[1] / 2

            # rotate the coordinates above the origin
            # the 'origin' is different in MATLAB due to 0 and 1 indexing
            # to accommodate this the points are translated by 1 in rotation
            ang = scanAngles_rad[k]
            rotM = np.array([[np.cos(ang), -np.sin(ang)],
                             [np.sin(ang), np.cos(ang)]])
            xy = rotM @ (xy+1) - 1

            # cancel the offset after rotation
            xy[0, :] += self.imageSize[0] / 2
            xy[1, :] += self.imageSize[1] / 2

            # shift coordinates by fractional part of the first one
            # ensure first coordinate always integers (why?)
            xy[0, :] -= xy[0, 0] % 1
            xy[1, :] -= xy[1, 0] % 1

            self.scanOr[k, ...] = xy
            self.scanDir[k, :] = [np.cos(ang+np.pi/2), np.sin(ang+np.pi/2)]

    def linear_alignment(self, xcoord=None, ycoord=None, return_index=False):
        """Perform linear alignment

        Parameters
        ----------
        xcoord : array-like, optional
            the candidates for x drifts. The default is None, set to the
            default linearSearch.
        ycoord : array-like, optional
            the candidates for y drifts. The default is None, set to the
            default linearSearch.
        return_index : bool, optional
            whether to return the indices of drfits resulting in the highest
            linearSearchScore or return the drifts resulting it.
            The default is False.

        Returns
        -------
        linearSearchScore : ndarray
            the array storing the score for correlation for different drifts.
        xInd, yInd : int
            the indices of drfits resulting in the highest linearSearchScore
        xdrift, ydrift : float
            the drifts resulting in the highest linearSearchScore.
        """

        # xcoord, ycoord set the value/size of meshgrid
        if xcoord is None:
            xcoord = self.linearSearch
        else:
            xcoord = np.asarray(xcoord)

        if ycoord is None:
            ycoord = self.linearSearch
        else:
            ycoord = np.asarray(ycoord)

        yDrift, xDrift = np.meshgrid(ycoord, xcoord)
        linearSearchScore = np.zeros((self.linearSearch.size, self.linearSearch.size))
        w2 = self._hanning_weight()

        # set the iterator depending on whether it reports progress
        if self.flagReportProgress:
            a0_iter = trange(len(self.linearSearch), desc='Linear Drift Search')
            # just use one bar to avoid confusion
            a1_iter = range(len(self.linearSearch))
        else:
            a0_iter = range(len(self.linearSearch))
            a1_iter = range(len(self.linearSearch))

        for a0 in a0_iter:
            for a1 in a1_iter:
                # Calculate time dependent linear drift
                xyShift = np.hstack([self.inds*xDrift[a0,a1],
                                     self.inds*yDrift[a0,a1]])

                # Apply linear drift to first two images
                # linear drift should be same for all
                self.scanOr[:2, ...] += xyShift.T

                # Generate trial images
                self = SPmakeImage(self, 0)
                self = SPmakeImage(self, 1)

                # Measure alignment score with hybrid correlation
                m1 = np.fft.fft2(w2 * self.imageTransform[0,...])
                m2 = np.fft.fft2(w2 * self.imageTransform[1,...])

                m = m1 * m2.conj()
                magnitude = np.sqrt(np.abs(m))
                phase = np.exp(1j*np.angle(m))
                Icorr = np.fft.ifft2(magnitude * phase).real

                linearSearchScore[a0, a1] = Icorr.max()

                # Remove linear drift from first two images
                self.scanOr[:2, ...] -= xyShift.T

        xInd, yInd = np.unravel_index(linearSearchScore.argmax(),
                                      linearSearchScore.shape)

        if return_index:
            return linearSearchScore, xInd, yInd
        else:
            r1, c1 = np.unravel_index(xInd, xDrift.shape, order='F')
            r2, c2 = np.unravel_index(yInd, yDrift.shape, order='F')

            xdrift = xDrift[r1, c1]
            ydrift = yDrift[r2, c2]

            return linearSearchScore, xdrift, ydrift

    def _hanning_weight(self):
        """Get the Hanning window for smoothing before Fourier transform
        """

        # chop off 0 to be consistent with the MATLAB hanningLocal
        N = self.scanLines.shape
        hanning = np.hanning(N[2] + 2)[1:-1] * np.hanning(N[1] + 2)[1:-1][:, None]
        padw = self.imageSize - np.asarray(N)[1:]
        shifts = np.floor(padw / 2 + 0.5).astype(int)
        padded = np.pad(hanning, ((0, padw[0]), (0, padw[1])),
                        mode='constant', constant_values=0)
        w2 = np.roll(padded, shifts, axis=(0,1))

        return w2

    def apply_linear_drift(self):
        """Apply linear drift

        Raises
        ------
        ValueError
            if xyLinearDrift is None.
        """

        if self.xyLinearDrift is None:
            raise ValueError('No x and y linear drift values stored. '
                             'Determine the drifts or set attribute xyLinearDrift.')

        xyShift = np.hstack([self.inds*self.xyLinearDrift[0],
                             self.inds*self.xyLinearDrift[1]])

        for k in range(self.numImages):
            self.scanOr[k, ...] += xyShift.T
            self = SPmakeImage(self, k)

        return

    def estimate_initial_alignment(self):
        """Estimate initial alignment

        Returns
        -------
        dxy : ndarray
            the estimated alignment.
        """

        dxy = np.zeros((self.numImages, 2))
        w2 = self._hanning_weight()
        G1 = np.fft.fft2(w2 * self.imageTransform[0,...])

        for k in range(1, self.numImages):
            G2 = np.fft.fft2(w2 * self.imageTransform[k,...])

            m = G1 * G2.conj()
            magnitude = np.sqrt(np.abs(m))
            phase = np.exp(1j*np.angle(m))
            Icorr = np.fft.ifft2(magnitude * phase).real

            dx, dy = np.unravel_index(Icorr.argmax(), Icorr.shape)

            # no need to shift indices here, compared to MATLAB
            nr, nc = Icorr.shape
            dx = (dx + nr/2) % nr - nr/2
            dy = (dy + nc/2) % nc - nc/2
            dxy[k, :] = dxy[k-1, :] + np.array([dx, dy])

            G1 = G2

        dxy[:, 0] -= dxy[:, 0].mean()
        dxy[:, 1] -= dxy[:, 1].mean()

        return dxy

    def apply_estimated_alignment(self, dxy):
        """Apply estimated alignment

        dxy : array-like
            the estimated alignment.
        """

        dxy = np.asarray(dxy)

        for k in range(self.numImages):
            self.scanOr[k, 0, :] += dxy[k, 0]
            self.scanOr[k, 1, :] += dxy[k, 1]
            self = SPmakeImage(self, k)

        return

    def plot_linear_drift_correction(self):
        """Plot images and their density after linear drift correction
        """
        # Plot results, image with scanline origins overlaid
        imagePlot = self.imageTransform.mean(axis=0)
        dens = self.imageDensity.prod(axis=0)

        # Scale intensity of image
        mask = dens > 0.5
        imagePlot -= imagePlot[mask].mean()
        imagePlot /= np.sqrt(np.mean(imagePlot[mask]**2))

        fig, ax = plt.subplots()
        ax.matshow(imagePlot, cmap='gray')

        # RGB colours
        cvals = np.array([[1, 0, 0],
                          [0, 0.7, 0],
                          [0, 0.6, 1],
                          [1, 0.7, 0],
                          [1, 0, 1],
                          [0, 0, 1]])

        # put origins on plot
        for k in range(self.numImages):
            x = self.scanOr[k, 1, :]
            y = self.scanOr[k, 0, :]
            c = cvals[k % cvals.shape[0], :]

            ax.plot(x, y, marker='.', markersize=12, linestyle='None', color=c)

        # put a nice cross at the reference point
        ax.plot(self.ref[1], self.ref[0], marker='+',
                markersize=20, color=(1,1,0), markeredgewidth=5)
        ax.plot(self.ref[1], self.ref[0], marker='+',
                markersize=20, color='k', markeredgewidth=3)

        return
