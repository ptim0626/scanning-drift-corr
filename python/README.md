# Scan distortion correction

This package corrects scan distortion by using orthogonal pairs.

The translation from MATLAB to Python is finished, all `SPmerge01linear`,
`SPmerge02` and `SPmerge03` are availabe to use. The usage is pretty much the
same with MATLAB. (see `example.py` or `example.ipynb`)

The initial translation focuses on consistency betweeh the MATLAB and
Python version, after ensure the correctness then more changes will be made.

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
- [ ] refactoring of the codes (currently everything flying around)
- [ ] handle rectangular image

