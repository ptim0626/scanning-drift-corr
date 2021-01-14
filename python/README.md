# Scan distortion correction

This package corrects scan distortion.

The translation from MATLAB to Python is ongoing, currently `SPmerge01linear` is finished and the work of `SPmerge02` and `SPmerge03` is work in progress. The initial translation focuses on consistency betweeh the MATLAB and Python version, after ensure the correctness then more changes will be made.

The tests compare output from Python to MATLAB version at different breakpoints, the MATLAB version comparing is in `matlab_modified`. Some modifications are made during the translation.


To install locally in editable mode, run
```
pip install -e .
```

After installing, you can try to run `example.py`. This script illustrates basic usage of the Python interface.
