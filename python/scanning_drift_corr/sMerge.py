"""The file contains the sMerge class
"""

import numpy as np

from scanning_drift_corr.SPmakeImage import SPmakeImage

class sMerge:
    """The sMerge class, corresponds to the sMerge struct
    """

    def __init__(self, scanAngles, images, paddingScale=1.125,
                 flagReportProgress=False):

        # properties to be use d in SPmakeimage
        self.KDEsigma = 1 / 2
        self.edgeWidth = 1 / 128
        
        # self.linearSearch = np.linspace(-0.02, 0.02, num=2*2+1)

        # self.flagReportProgress = flagReportProgress

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
        # self.inds = np.linspace(-0.5, 0.5, num=self.nr)[:, None]
        # self.linearSearch *= self.nr
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



