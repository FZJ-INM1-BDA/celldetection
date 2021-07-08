""" CellDetection is a package for efficient and robust cell detection with PyTorch.

CellDetection is licensed under Apache License, Version 2.0.

"""

DOCSTRING = (__doc__ or '').split("\n")

from setuptools import setup

requirements = [
    'numpy',
    'scikit-image',
    'scipy',
    'opencv-python',
    'torch>=1.9.0',
    'torchvision>=0.10.0',
    'seaborn',
    'matplotlib',
    'tqdm',
    'h5py',
    'setuptools',
]

setup(
    author="Eric Upschulte",
    author_email='e.upschulte@fz-juelich.de',
    name='celldetection',
    version='0.2.0',
    description=DOCSTRING[0],
    long_description="\n".join(DOCSTRING[2:]),
    url='https://celldetection.org',
    packages=['celldetection', 'celldetection.data', 'celldetection.models', 'celldetection.ops', 'celldetection.util',
              'celldetection.visualization'],
    install_requires=requirements,
    license='Apache License, Version 2.0',
    keywords=['cell', 'detection', 'object', 'segmentation', 'pytorch', 'cpn', 'contour', 'proposal', 'network', 'deep',
              'learning', 'unet', 'fzj', 'julich', 'juelich', 'ai']
)
