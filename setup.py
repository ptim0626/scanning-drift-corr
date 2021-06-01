import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
name="scanning_drift_corr",
version="1.0.0",
author="Timothy Poon",
author_email="timothy0626@gmail.com",
description="Correct scan drift distortion",
long_description=long_description,
long_description_content_type="text/markdown",
url="https://github.com/ptim0626/scanning-drift-corr",
project_urls={
    "Bug Tracker": "https://github.com/ptim0626/scanning-drift-corr/issues",
    },
package_dir={"": "src"},
packages=setuptools.find_packages(where="src", exclude=['*tests*']),
classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
],
python_requires='>=3.7',
install_requires=[
    'numpy',
    'matplotlib',
    'scipy',
    'tqdm',
    'h5py',
    ],
extras_require={'tests': ['pytest']},
package_data={
    'scanning_drift_corr' : [
        'tests/nonlinear_drift_correction_synthetic_dataset_for_testing.mat',
        'examples/example.py',
        'examples/example.ipynb',
        ],
    },
)
