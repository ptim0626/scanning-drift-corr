# Changelog
My own modification of MATLAB files from Colin is documented here.

## SPmerge01linear.m
### Line 63-66, the rotation of coordinates
- The precision here in rotation sometimes causes difference between Python
and MATLAB version.
- It impacts on subsequent removal of fractional part.
- e.g. for 512x512 image, the first coordinate is 32.99999999... in MATLAB
but 33.0 in Python. So in MATLAB all coordinates are shifted by 0.9999999...
while it is not shifted in Python.
- Increase the precision here solves the inconsistency.

## SPmerge02.m
### Line 174-176, pull the scanlines correctly
- The `inds` originally is incorrect here, should use size 2.
- Now will work with rectangular matrices.

