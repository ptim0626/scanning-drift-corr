# Scan distortion correction

This package corrects scan distortion by using orthogonal pairs.

**This is the python translation from the code of Colin Ophus. The MATLAB code
can be found [here](https://github.com/cophus/scanning-drift-corr "Colin Ophus' MATLAB code").
It can also be found [here](./matlab).  The implementation is based on
[his paper](https://www.sciencedirect.com/science/article/abs/pii/S0304399115300838
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

## Installation
To install locally in editable mode, run
```
pip install -e .
```

## Usage
After installing, you can try to run `example.py`.  This script illustrates
basic usage of the Python interface.

A Jupyter notebook `example.ipynb` is also available.

## TODO
- [x] refactoring of the codes (currently everything flying around)
- [x] handle rectangular image

