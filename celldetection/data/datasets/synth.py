from .generic import GenericH5
from glob import glob
from os.path import join, isdir
from os import makedirs
from torchvision import datasets

__all__ = ['download_synth', 'SynthTrain', 'SynthVal', 'SynthTest']


def download_synth(directory, url='https://celldetection.org/data/synth.zip'):
    """Download Synth.

    Download and extract the Synth dataset to given directory.

    Args:
        directory: Root directory.
        url: Download URL (this dataset is distributed in a single zip file).

    """
    makedirs(directory, exist_ok=True)
    datasets.utils.download_and_extract_archive(url, directory)


class _Synth(GenericH5):
    def __init__(self, directory, download, mode: str, cache=True):
        assert mode in ('train', 'val', 'test')

        if download and not isdir(join(directory, mode)):
            download_synth(directory)

        super().__init__(
            filenames=glob(join(directory, mode, '*.h5')),
            keys=('image', 'labels'), cache=cache
        )


class SynthTrain(_Synth):
    def __init__(self, directory, download=False, cache=True):
        """Synth Train data.

        Args:
            directory: Data directory. E.g. ``directory='data/synth'``.
            download: Whether to download data.
            cache: Whether to hold data in memory.
        """
        super().__init__(directory, download, 'train', cache=cache)


class SynthVal(_Synth):
    def __init__(self, directory, download=False, cache=True):
        super().__init__(directory, download, 'val', cache=cache)

    __init__.__doc__ = SynthTrain.__init__.__doc__.replace('Synth Train data.', 'Synth Val data.')


class SynthTest(_Synth):
    def __init__(self, directory, download=False, cache=True):
        super().__init__(directory, download, 'test', cache=cache)

    __init__.__doc__ = SynthTrain.__init__.__doc__.replace('Synth Train data.', 'Synth Test data.')
