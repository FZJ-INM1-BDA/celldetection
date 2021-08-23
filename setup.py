from setuptools import setup
from os.path import join, dirname, abspath


def read_utf8(*args):
    with open(join(*args), encoding="utf-8") as f:
        return f.read()


directory, m = dirname(abspath(__file__)), {}
exec(read_utf8(directory, 'celldetection', '__meta__.py'), m)
requirements = read_utf8(directory, 'requirements.txt').strip().split("\n")
long_description = read_utf8(directory, 'README.md')

setup(
    author=m['__author__'],
    author_email=m['__email__'],
    name=m['__title__'],
    version=m['__version__'],
    description=m['__summary__'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=m['__url__'],
    packages=['celldetection', 'celldetection.data', 'celldetection.data.datasets', 'celldetection.models',
              'celldetection.mpi', 'celldetection.ops', 'celldetection.util', 'celldetection.visualization'],
    package_data={'': ['LICENSE', 'requirements.txt', 'README.md']},
    include_package_data=True,
    install_requires=requirements,
    license=m['__license__'],
    keywords=['cell', 'detection', 'object', 'segmentation', 'pytorch', 'cpn', 'contour', 'proposal', 'network', 'deep',
              'learning', 'unet', 'fzj', 'julich', 'juelich', 'ai'],
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent'
    ]
)
