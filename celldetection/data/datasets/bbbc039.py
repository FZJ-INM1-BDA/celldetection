from os.path import join
from os import makedirs
from imageio import imread
from skimage import measure
import numpy as np
from ..cpn import labels2contour_list as labels2contours
from ...visualization.images import show_detection
import torchvision

__all__ = ['download_bbbc039', 'BBBC039Train', 'BBBC039Test', 'BBBC039Val']


def read_all(directory, filename):
    if filename is None or directory is None:
        return None
    with open(join(directory, filename), 'r') as f:
        return [i.strip() for i in f.readlines()]


def load(images_directory, masks_directory, names, **label_kwargs):
    if None in (images_directory, masks_directory):
        return None, None, None
    images = [imread(join(images_directory, f.replace('.png', '.tif'))) for f in names]
    masks = [imread(join(masks_directory, f)) for f in names]
    return images, masks, [measure.label(m[:, :, 0], **label_kwargs) for m in masks]


def download_bbbc039(directory):
    """Download BBBC039.

    Download and extract the BBBC039 dataset to given directory.

    References:
        https://bbbc.broadinstitute.org/BBBC039

    Args:
        directory: Root directory.

    """
    makedirs(directory, exist_ok=True)
    for url in [
        'https://data.broadinstitute.org/bbbc/BBBC039/images.zip',
        'https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip',
        'https://data.broadinstitute.org/bbbc/BBBC039/masks.zip'
    ]:
        torchvision.datasets.utils.download_and_extract_archive(url, directory)


class _BBBC039:
    def __init__(self, directory, download, mode: str):
        assert mode in ('train', 'test', 'val')

        meta_directory = join(directory, 'metadata')
        masks_directory = join(directory, 'masks')
        images_directory = join(directory, 'images')

        if download:
            download_bbbc039(directory)

        self.names = read_all(meta_directory, {
            'train': 'training.txt',
            'val': 'validation.txt',
            'test': 'test.txt'
        }[mode])

        self.images, self.masks, self.labels = load(images_directory, masks_directory, self.names)

    def plot(self, num=1, figsize=(20, 15)):
        for i in np.random.randint(0, len(self), num):
            show_detection(image=self.images[i], contours=labels2contours(self.labels[i]),
                           figsize=figsize, contour_linestyle='-')

    def __getitem__(self, item):
        return self.names[item], self.images[item], self.masks[item], self.labels[item]

    def __len__(self):
        return len(self.images)


class BBBC039Train(_BBBC039):
    def __init__(self, directory, download=False):
        """BBBC039 Train.

        Training split of the BBBC039 dataset.

        References:
            https://bbbc.broadinstitute.org/BBBC039

        Args:
            directory: Root directory.
            download: Whether to download the dataset.
        """
        super().__init__(directory, download=download, mode='train')


class BBBC039Val(_BBBC039):
    def __init__(self, directory, download=False):
        """BBBC039 Validation.

        Validation split of the BBBC039 dataset.

        References:
            https://bbbc.broadinstitute.org/BBBC039

        Args:
            directory: Root directory.
            download: Whether to download the dataset.
        """
        super().__init__(directory, download=download, mode='val')


class BBBC039Test(_BBBC039):
    def __init__(self, directory, download=False):
        """BBBC039 Test.

        Test split of the BBBC039 dataset.

        References:
            https://bbbc.broadinstitute.org/BBBC039

        Args:
            directory: Root directory.
            download: Whether to download the dataset.
        """
        super().__init__(directory, download=download, mode='test')
