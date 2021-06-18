# Scan distortion correction

This package corrects scan distortion by using orthogonal pairs.

**This is the Python translation from the code of Colin Ophus. The MATLAB code
can be found [here](https://github.com/cophus/scanning-drift-corr "Colin Ophus' MATLAB code").
It can also be found [here](https://github.com/ptim0626/scanning-drift-corr/tree/master/matlab).
The implementation is based on [his paper](https://www.sciencedirect.com/science/article/abs/pii/S0304399115300838
"Correcting nonlinear drift distortion of scanning probe and scanning transmission electron microscopies from image pairs with orthogonal scan directions").**

The translation from MATLAB to Python is finished, all `SPmerge01linear`,
`SPmerge02` and `SPmerge03` are availabe to use. The usage is pretty much the
same with MATLAB (see `example.py` or `example.ipynb`). The detail of the usage
can be found in the original [README](./README_original.md) or the [original
repository](https://github.com/cophus/scanning-drift-corr "Colin Ophus' MATLAB code").

The MATLAB-to-Python translation provides a convenient interface where MATLAB
is not available or Python is the preferred language in your analysis pipeline.
The initial translation focuses on consistency between the MATLAB and
Python version, after ensure the correctness then more changes will be made.
A certain degree of parallelism is introduced in the Python version.

The tests compare output from Python to MATLAB version at different breakpoints,
the MATLAB version comparing is in `matlab_modified`. Some modifications are
made during the translation, mainly to address floating-point errors.

## Try it for yourself
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ptim0626/scanning-drift-corr/HEAD?filepath=src%2Fscanning_drift_corr%2Fexamples%2Fexample.ipynb)


## Installation
The package is available in conda-forge:
```
conda install -c conda-forge scanning_drift_corr
```

To install via pip:
```
pip install scanning_drift_corr
```

To install locally in editable mode, clone the repository and in the root
directory run
```
pip install -e .
```

Version `1.0.0` can be used with Python version >= 3.7 but the parallel
implementation is broken in Windows/macOS :( Subsequent version supports Python
version >= 3.8 due to the usage of `shared_memory` in the standard library
`multiprocessing`, and the parallel implementation works in different systems.
However, see [notes](#notes-on-parallel-implementation) below.

## Usage
After installing, you can try to run `example.py`.  This script illustrates
basic usage of the Python interface.

A Jupyter notebook `example.ipynb` is also available.

## Notes on parallel implementation
`SPmerge01linear` and `SPmerge02` can be provided with `parallel=True` for
parallel computation. This works well in Unix system where processes are
forked, but in Windows/macOS processes are spawned and the parallel
implementation is (usually) slower than the serial one. The default behaviour
for Unix system is `parallel=True` and for Windows/macOS is `parallel=False`.
You are welcome to experiment yourself. See [here](https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods)
for details.

## TODO
- [x] refactoring of the codes (currently everything flying around)
- [x] handle rectangular image

