"""The file contains the SPmerge03 function
"""

import numpy as np
import matplotlib.pyplot as plt

from scanning_drift_corr.tools import distance_transform


def SPmerge03(sm):
    # return imageFinal, signalArray, densityArray

    flagPlot = True

    # Gaussian sigma value used in kernel density estimator (pixels)
    KDEsigma = 0.5 + 1

    # upsampling factor used in image generation (integer)
    # Using a large upsampleFactor can be very slow.
    upsampleFactor = 2

    # Smoothing sigma value for density estimation (pixels)
    sigmaDensity = 8/2

    # Thickness of windowed boundary (pixels)
    boundary = 8

    # Set to true to enable cos(theta)^2 Fourier weights
    flagFourierWeighting = True

    # Set to true to downsample output to the original
    # resolution, as opposed to that of "upsampleFactor."
    flagDownsampleOutput = True

    # Initialize arrays
    upsampledSize = sm.imageSize*upsampleFactor
    signalArray = np.zeros((sm.numImages, *upsampledSize));
    densityArray = np.zeros((sm.numImages, *upsampledSize));
    imageFinal = np.zeros(upsampledSize)

    # kernel generation in upsampled coordinates
    nr, nc = sm.imageSize
    x = np.fft.fftfreq(upsampleFactor*nr, 1/nr)[:, None]
    y = np.fft.fftfreq(upsampleFactor*nc, 1/nc)
    kernel = np.fft.fft2(np.exp(-x**2/(2*KDEsigma**2)) * np.exp(-y**2/(2*KDEsigma**2)))
    smoothDensityEstimate = np.fft.fft2(np.exp(-x**2/(2*sigmaDensity**2)) *
                                        np.exp(-y**2/(2*sigmaDensity**2)) /
                                        (2*np.pi*sigmaDensity**2*upsampleFactor**2))

    # Loop over scans and create images / densities
    t = np.arange(1, sm.nc+1)
    for k in range(sm.numImages):
        # Expand coordinates
        x0 = sm.scanOr[k, 0, :][:,None]
        y0 = sm.scanOr[k, 1, :][:,None]
        # plus here to shift in Python's coordinate system
        xInd = x0*upsampleFactor + (upsampleFactor-1)/2 + (t*sm.scanDir[k, 0])*upsampleFactor
        yInd = y0*upsampleFactor + (upsampleFactor-1)/2 + (t*sm.scanDir[k, 1])*upsampleFactor

        # Prevent pixels from leaving image boundaries
        xInd = np.clip(xInd, 0, (sm.imageSize[0]*upsampleFactor)-2).ravel()
        yInd = np.clip(yInd, 0, (sm.imageSize[1]*upsampleFactor)-2).ravel()

        imgsize = sm.imageSize*upsampleFactor

        # Create bilinear coordinates
        xIndF = np.floor(xInd).astype(int)
        yIndF = np.floor(yInd).astype(int)
        xAll = np.vstack([xIndF, xIndF+1, xIndF, xIndF+1])
        yAll = np.vstack([yIndF, yIndF, yIndF+1, yIndF+1])
        dx = xInd - xIndF
        dy = yInd - yIndF
        w = np.vstack([(1-dx)*(1-dy), dx*(1-dy), (1-dx)*dy, dx*dy])
        indAll = np.ravel_multi_index((xAll, yAll), imgsize)

        # weigh the raw image for interpolation
        image = sm.scanLines[k, ...]
        wsig = w * image.ravel()
        wcount = w

        # Generate image and density
        signalArray[k, ...] = np.bincount(indAll.ravel(), weights=wsig.ravel(),
                                          minlength=imgsize.prod()).reshape(imgsize)

        densityArray[k, ...] = np.bincount(indAll.ravel(), weights=wcount.ravel(),
                                           minlength=imgsize.prod()).reshape(imgsize)

    # for refactoring later, sepearate the for loop
    # Apply KDE to both arrays
    for k in range(sm.numImages):
        signalArray[k, ...] = np.fft.ifft2(np.fft.fft2(signalArray[k, ...])*kernel).real
        densityArray[k, ...] = np.fft.ifft2(np.fft.fft2(densityArray[k, ...])*kernel).real

    # Normalize image intensity by sampling density
    sub = densityArray > 1e-8
    signalArray[sub] /= densityArray[sub]

    # Calculate smooth density estimate, set max value to 2 to reduce edge
    # effects, apply to images.
    intensityMedian = np.median(sm.scanLines)
    for k in range(sm.numImages):
        minDensity = np.minimum(densityArray[k,...], 2)
        densityArray[k,...] = np.fft.ifft2(np.fft.fft2(minDensity) * smoothDensityEstimate).real

        denMask = densityArray[k,...] < 0.5
        minDist = np.minimum((distance_transform(denMask) / (boundary*upsampleFactor)), 1)
        densityArray[k,...] = np.sin(minDist * np.pi/2)**2

        # Apply mask to each image
        signalArray[k, ...] = signalArray[k, ...]*densityArray[k, ...] + (1-densityArray[k,...])*intensityMedian


    # Combine scans to produce final image
    if flagFourierWeighting:
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

            # Filtered image
            imageFinal = imageFinal + np.fft.fft2(signalArray[k,...]) * qWeight
            weightTotal += qWeight

        imageFinal = np.fft.ifft2(imageFinal/weightTotal).real

        # apply global density mask
        density = densityArray.prod(axis=0)
        imageFinal = imageFinal*density + (1-density)*intensityMedian
    else:
        # Density weighted average
        density = densityArray.prod(axis=0)
        imageFinal = signalArray.mean(axis=0)*density + (1-density)*np.median(sm.scanLines)

        if not flagDownsampleOutput:
            # Keep intensity constant
            imageFinal /= upsampleFactor**2


    # Downsample outputs if required
    if flagDownsampleOutput and (upsampleFactor>1):
        # Fourier subsets for downsampling
        # TODO handle odd number pixel images
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

        # Downsample output image
        imgFFT = np.fft.fft2(imageFinal)
        imageFinal = np.fft.ifft2(imgFFT[xVec[:,None], yVec[None,:]]).real / upsampleFactor**2

        # Get the right region for signal and density array
        sigFFT = np.fft.fft2(signalArray,  axes=(-2, -1))
        signalArray = np.fft.ifft2(sigFFT[:, xVec[:,None], yVec[None,:]]).real
        denFFT = np.fft.fft2(densityArray,  axes=(-2, -1))
        densityArray = np.fft.ifft2(denFFT[:, xVec[:,None], yVec[None,:]]).real


    if flagPlot:
        fig, ax = plt.subplots()
        ax.matshow(imageFinal, cmap='gray')


    return imageFinal, signalArray, densityArray
