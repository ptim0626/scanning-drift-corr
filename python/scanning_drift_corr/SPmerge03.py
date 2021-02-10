"""The file contains the SPmerge03 function
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt

from scanning_drift_corr.tools import distance_transform, bilinear_interpolation


def SPmerge03(sm, **kwargs):
    """ Final scanning probe merge script. This script uses KDE and Fourier
    filtering to produce a combined image from all component scans. The
    Fourier weighting used (if flag is enabled) is cos(theta)^2, where
    theta is the angle from the scan direction. This essentially zeros out
    the slow scan direction, while weighting the fast scan direction at 100%.

    Parameters
    ----------
    sm : sMerge object
        the sMerge object contains all the data.
    KDEsigma : float, optional
        Gaussian sigma value used in kernel density estimator (pixels)
    upsampleFactor : int, optional
        upsampling factor used in image generation, a large upsampleFactor
        can be very slow
    sigmaDensity : float, optional
        smoothing sigma value for density estimation (pixels)
    boundary : int
        thickness of windowed boundary (pixels)
    flagFourierWeighting : bool
        set to true to enable cos(theta)^2 Fourier weights
    flagDownsampleOutput : bool
        set to true to downsample output to the original resolution, as
        opposed to that of upsampleFactor
    flagPlot : bool
        to plot the final image of not

    Returns
    -------
    imageFinal : ndarray
        final combined image.
    signalArray : ndarray
        image stack containing estimated image from each scan, with axis 0
        the navigation index
    densityArray : ndarray
        image stack containing estimated density of each scan, with axis 0
        the navigation index
    """

    # ignore unknown input arguments
    _args_list = ['KDEsigma', 'upsampleFactor', 'sigmaDensity', 'boundary',
                  'flagFourierWeighting', 'flagDownsampleOutput', 'flagPlot']
    for key in kwargs.keys():
        if key not in _args_list:
            msg = "The argument '{}' is not recognised, and it is ignored."
            warnings.warn(msg.format(key), RuntimeWarning)

    # set default values or from input arguments
    KDEsigma = kwargs.get('KDEsigma', 1.5)
    upsampleFactor = kwargs.get('upsampleFactor', 2)
    sigmaDensity = kwargs.get('sigmaDensity', 8/2)
    boundary = kwargs.get('boundary', 8)
    flagFourierWeighting = kwargs.get('flagFourierWeighting', True)
    flagDownsampleOutput = kwargs.get('flagDownsampleOutput', True)
    flagPlot = kwargs.get('flagPlot', True)

    # Initialize arrays
    upsampledSize = sm.imageSize*upsampleFactor
    signalArray = np.zeros((sm.numImages, *upsampledSize));
    densityArray = np.zeros((sm.numImages, *upsampledSize));

    # construct k space coordinates
    nr, nc = sm.imageSize
    x = np.fft.fftfreq(upsampleFactor*nr, 1/nr)[:, None]
    y = np.fft.fftfreq(upsampleFactor*nc, 1/nc)

    # kernel generation in upsampled coordinates
    kernel = np.fft.fft2(np.exp(-x**2/(2*KDEsigma**2)) * np.exp(-y**2/(2*KDEsigma**2)))
    imageSize = sm.imageSize
    for k in range(sm.numImages):
        # Loop over scans and create images / densities
        scanLines = sm.scanLines[k, ...]
        scanOr = sm.scanOr[k, ...]
        scanDir = sm.scanDir[k, :]
        ret = bilinear_interpolation(scanLines, scanOr, scanDir, imageSize,
                                     upsampleFactor=upsampleFactor)

        # Apply KDE to both arrays
        signalArray[k, ...] = conv_fft(ret[0], kernel)
        densityArray[k, ...] = conv_fft(ret[1], kernel)


    # Normalize image intensity by sampling density
    sub = densityArray > 1e-8
    signalArray[sub] /= densityArray[sub]


    # Calculate smooth density estimate, set max value to 2 to reduce edge
    # effects, apply to images.
    smoothDensityEstimate = np.fft.fft2(np.exp(-x**2/(2*sigmaDensity**2)) *
                                        np.exp(-y**2/(2*sigmaDensity**2)) /
                                        (2*np.pi*sigmaDensity**2*upsampleFactor**2))
    intensityMedian = np.median(sm.scanLines)
    for k in range(sm.numImages):
        minDensity = np.minimum(densityArray[k,...], 2)
        densityArray[k,...] = conv_fft(minDensity, smoothDensityEstimate)

        denMask = densityArray[k,...] < 0.5
        minDist = np.minimum((distance_transform(denMask) / (boundary*upsampleFactor)), 1)
        densityArray[k,...] = np.sin(minDist * np.pi/2)**2

        # Apply mask to each image
        signalArray[k, ...] = density_masking(signalArray[k, ...],
                                              densityArray[k, ...],
                                              intensityMedian)


    # generate final image by combining scans at different directions
    density = densityArray.prod(axis=0)
    if flagFourierWeighting:
        imageFinal = fourier_weight(sm, signalArray, upsampleFactor)

        # apply global density mask
        imageFinal = density_masking(imageFinal, density, intensityMedian)
    else:
        # Density weighted average
        sigavg = signalArray.mean(axis=0)
        imageFinal = density_masking(sigavg, density, intensityMedian)

        if not flagDownsampleOutput:
            # Keep intensity constant
            imageFinal /= upsampleFactor**2


    # Downsample outputs if required
    if flagDownsampleOutput and (upsampleFactor>1):
        xVec, yVec = _get_downsampled_coord(sm, upsampleFactor)

        imageFinal = downsampling(imageFinal, xVec, yVec) / upsampleFactor**2
        signalArray = downsampling(signalArray, xVec, yVec)
        densityArray = downsampling(densityArray, xVec, yVec)


    # plot final image
    if flagPlot:
        fig, ax = plt.subplots()
        ax.matshow(imageFinal, cmap='gray')

    return imageFinal, signalArray, densityArray

def conv_fft(img, kernel):

    imgconv = np.fft.ifft2(np.fft.fft2(img)*kernel).real

    return imgconv

def fourier_weight(sm, signalArray, upsampleFactor):

    upsampledSize = sm.imageSize*upsampleFactor
    imageFinal = np.zeros(upsampledSize)

    # Make Fourier coordinates
    qx = np.fft.fftfreq(imageFinal.shape[0])
    qy = np.fft.fftfreq(imageFinal.shape[1])
    qya, qxa = np.meshgrid(qy, qx)
    qTheta = np.arctan2(qya, qxa)

    # Generate Fourier weighted final image
    weightTotal = np.zeros(imageFinal.shape)
    for k in range(sm.numImages):
        # Filter array
        thetaScan = np.arctan2(sm.scanDir[k,1], sm.scanDir[k,0])
        qWeight = np.cos(qTheta - thetaScan)**2

        # is this to normalise the DC component?
        qWeight[0, 0] = 1

        # Filtered image, combining image in k space
        imageFinal = imageFinal + np.fft.fft2(signalArray[k,...]) * qWeight
        weightTotal += qWeight

    # divide the weight of the added images before getting the image
    imageFinal = np.fft.ifft2(imageFinal/weightTotal).real

    return imageFinal


def density_masking(signal, density, val):

    sigval = signal*density + (1-density)*val

    return sigval


def _get_downsampled_coord(sm, upsampleFactor):
    # Fourier subsets for downsampling
    # imageSize never odd so no need to handle
    xsz = sm.imageSize[0]
    ysz = sm.imageSize[1]
    xszMid = int(xsz/2)
    yszMid = int(ysz/2)

    xVec = np.empty(xsz)
    xVec[:xszMid] = np.arange(0, xszMid)
    xVec[xszMid:] = np.arange(-xszMid, 0) + xsz*upsampleFactor

    yVec = np.empty(ysz)
    yVec[:yszMid] = np.arange(0, yszMid)
    yVec[yszMid:] = np.arange(-yszMid, 0) + ysz*upsampleFactor

    xVec = xVec.astype(int)
    yVec = yVec.astype(int)

    return xVec, yVec

def downsampling(arr, xInd, yInd):

    # Downsample output image or image stack
    if arr.ndim == 2:
        Farr = np.fft.fft2(arr)
        arr_down = np.fft.ifft2(Farr[xInd[:,None], yInd[None,:]]).real
    elif arr.ndim == 3:
        Farr = np.fft.fft2(arr, axes=(-2, -1)) # default but just for clarity
        arr_down = np.fft.ifft2(Farr[:, xInd[:,None], yInd[None,:]]).real


    return arr_down
