from os.path import join, dirname, basename, isdir
from os import makedirs
from imageio import imread
from glob import glob
import numpy as np
from ..cpn import masks2labels
import torchvision

__all__ = ['download_bbbc038', 'BBBC038Train']


def download_bbbc038(directory):
    """Download BBBC038.

    Download and extract the BBBC038 dataset to given directory.

    References:
        https://bbbc.broadinstitute.org/BBBC038

    Args:
        directory: Root directory.

    """
    for url in [
        'https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip',
        'https://data.broadinstitute.org/bbbc/BBBC038/stage1_test.zip',
        'https://data.broadinstitute.org/bbbc/BBBC038/stage2_test_final.zip'
    ]:
        directory_ = join(directory, basename(url).split('.')[0])
        makedirs(directory_, exist_ok=True)
        torchvision.datasets.utils.download_and_extract_archive(url, directory, extract_root=directory_)


class BBBC038Train:
    def __init__(self, directory, download=False):
        if download and not isdir(join(directory, 'stage1_train')):
            download_bbbc038(directory)
        self.image_f = sorted(glob(join(directory, 'stage1_train', '*', 'images', '*.*')))
        self.label_f = [sorted(glob(join(dirname(dirname(f)), 'masks', '*.*'))) for f in self.image_f]

    def __getitem__(self, item):
        img_f = self.image_f[item]
        lbl_f = self.label_f[item]

        img = imread(img_f)
        masks = np.stack([imread(f) for f in lbl_f])
        lbl = masks2labels(masks)
        return img, lbl, (img_f, lbl_f)

    def __len__(self):
        return len(self.image_f)
