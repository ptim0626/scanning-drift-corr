"""The file contains the sMerge class
"""

import numpy as np

class sMerge:
    """The sMerge class, corresponds to the sMerge struct

    Attributes
    ----------
    isStack : bool
        whether the provided images are in a stack or not
    img_shape : ndarray
        the shape of the provided images
    numImages : int
        the number of input images
    scanAngles : ndarray
        the direction of the input images, same order with the provided images
    nr, nc : int
        the number of rows and columns of the input images
    imageSize : ndarray
        the shape of the output images
    scanOr : ndarray
        the xy points define the origins of the scan lines. Three dimensional
        array, with the first dim the number of images, second dim (size 2)
        the x and y, and third dim the number of rows of input images.
    scanDir : ndarray
        contains the xy vectors of which all rotated images. Two dimensional
        array, with the first dim the number of images and second dim the
        vectors (size 2, with x and y)
    scanLines : ndarray
        the input images as a stack. Three dimensional array, with the first
        dim the number of images, second and third dim the row and column
    imageTransform : ndarray
        the regenerated image with the current scan line origins. Three
        dimensional, the first dim the number of images, the second and third
        dim store the regenerated image with shape imageSize
    imageDensity : ndarray
        the density of the regenerated image during interpolation with the raw
        images. Three dimensional, the first dim the number of images, the
        second and third dim store the density with shape imageSize
    linearSearchScores : ndarray
        the correlation score during the search of linear drifts. Three
        dimensional, the first dim the number of images, the second and third
        dim correspond to the search grid position
    xyLinearDrift : ndarray
        two values, the x and y linear drift found
    ref : ndarray
        two values, the x and y coordinates of the reference point during
        alignment
    scanActive : ndarray
        contains the active position (bool array) of scan line used for
        alignment
    stats : ndarray
        the mean absolute difference of each alignment steps during final
        alignment. Two dimensional, first dim the number of alignment and
        second dim the mean absolute difference.
    """

    def __init__(self, scanAngles, images, KDEsigma=1/2, edgeWidth=1/128,
                 paddingScale=1.125, imageRef=None):
        """
        Parameters
        ----------
        scanAngles : array-like
            the scan angles in degrees, the same order as provided images.
        images : array-like
            provided images, can be a sequence of images (e.g. img1, img2,
            img3, im4) or as a stack (three-dimensional structure with
            navigation index first). When a stack is provided, no check is
            performed to ensure the first index is the navigation.
        KDEsigma : float, optional
            the smoothing between pixels when regenerating images for KDE. The
            default is 1/2.
        edgeWidth : float, optional
            size of edge blending relative to input images. The default is
            1/128.
        paddingScale : float, optional
            padding amount for scaling of the output.
        imageRef : array-like, optional
            a reference image to compare with when performing alignment. The
            default is None.
        """

        self.KDEsigma = KDEsigma
        self.edgeWidth = edgeWidth
        self.paddingScale = paddingScale
        if imageRef is None:
            self.imageRef = None
        else:
            self.imageRef = np.asarray(imageRef)

        # validate input
        self._input_validation(scanAngles, images)
        self.nr, self.nc = self.img_shape

        # set the size of the output images
        self.imageSize = np.floor(self.img_shape *
                                  self.paddingScale/4 + 0.5).astype(int) * 4

        # initialise scanOr and scanDir
        self.scanOr = np.zeros((self.numImages, 2, self.nr))
        self.scanDir = np.zeros((self.numImages, 2))

        # save raw data to scanLines
        if self.isStack:
            self.scanLines = images[0]
        else:
            self.scanLines = np.empty((self.numImages, *self.img_shape))
            for k, im in enumerate(images):
                self.scanLines[k, :, :] = im

        # calculate the scan line origins
        self._set_scanOr_scanDir()

        self.imageTransform = np.zeros((self.numImages, *self.imageSize))
        self.imageDensity = np.zeros((self.numImages, *self.imageSize))
        self.linearSearchScores = None
        self.xyLinearDrift = None
        self.ref = np.floor(self.imageSize/2 + 0.5).astype(int) - 1
        self.scanActive = None
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
        elif len(images) > 1:
            # image sequence
            shapes = np.asarray([arr.shape for arr in images])
            shape_equal = (shapes[0,0] == shapes[:, 0]).all() &\
                (shapes[0,1] == shapes[:, 1]).all()
            if not shape_equal:
                msg = 'The provided images are not of the same shape'
                raise ValueError(msg)

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
