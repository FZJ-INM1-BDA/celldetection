from setuptools import setup

requirements = [
    'numpy',
    'scikit-image',
    'scipy',
    'opencv-python',
    'torch',
    'torchvision',
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
    version='0.1.0',
    description='Cell Detection with PyTorch',
    url='https://celldetection.org',
    packages=['celldetection', 'celldetection.data', 'celldetection.models', 'celldetection.ops', 'celldetection.util',
              'celldetection.visualization'],
    install_requires=requirements,
    license='Apache License, Version 2.0',
    keywords=['cell', 'detection', 'object', 'segmentation', 'pytorch', 'cpn', 'contour', 'proposal', 'network', 'deep',
              'learning', 'unet', 'fzj', 'julich', 'juelich', 'ai']
)
