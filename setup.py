from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    author="Eric Upschulte",
    author_email='e.upschulte@fz-juelich.de',
    name='celldetection',
    version='0.1.0',
    description='Cell Detection with PyTorch',
    url='https://celldetection.org',
    packages=['celldetection', 'celldetection.data', 'celldetection.models', 'celldetection.ops', 'celldetection.util',
              'celldetection.visualization'],
    install_requires=requirements
)
