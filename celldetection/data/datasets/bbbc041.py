from torchvision.datasets import utils
from os.path import join, basename, isfile
from os import makedirs, listdir
from imageio import imread
import numpy as np
import json
from matplotlib import pyplot as plt
import shutil
from ...visualization.images import plot_box, plot_text

__all__ = ['download_bbbc041', 'BBBC041Train', 'BBBC041Test']


def download_bbbc041(directory, url='https://data.broadinstitute.org/bbbc/BBBC041/malaria.zip'):
    """Download BBBC041.

    Download and extract the BBBC041 dataset to given directory.

    References:
        https://bbbc.broadinstitute.org/BBBC041

    Args:
        directory: Root directory.
        url: Download URL (this dataset is distributed in a single zip file).

    """
    makedirs(directory, exist_ok=True)
    utils.download_and_extract_archive(url, directory)
    malaria_dir = join(directory, 'malaria')
    for src in listdir(malaria_dir):
        src = join(malaria_dir, src)
        dst = join(directory, basename(src))
        print(src, '->', dst)
        shutil.move(src, dst)


class _BBBC041:
    def __init__(self, directory, download, mode: str):
        assert mode in ('train', 'test')  # , 'val'

        json_file = join(directory, {
            'train': 'training.json',
            # 'val': 'validation.json',
            'test': 'test.json'
        }[mode])

        if download and not isfile(json_file):
            download_bbbc041(directory)

        with open(json_file, 'r') as f:
            meta = json.load(f)

        self.filenames = []
        self.images = []
        self.boxes = []
        self.categories = []
        for item in meta:
            image_item = item['image']
            object_items = item['objects']
            pn = image_item['pathname']
            if pn[0] == '/':
                pn = pn[1:]
            self.filenames.append(join(directory, pn))
            self.images.append(imread(self.filenames[-1]))
            boxes = []
            categories = []
            for box in object_items:
                bb = box['bounding_box']
                # Boxes as x_min, y_min, x_max, y_max
                boxes.append([bb['minimum']['c'], bb['minimum']['r'], bb['maximum']['c'], bb['maximum']['r']])
                categories.append(box['category'])
            self.boxes.append(boxes)
            self.categories.append(categories)

    def plot(self, item=None, num=1, figsize=(16, 9)):
        if item is None:
            item = np.random.randint(0, len(self), num)
        else:
            item = item,
        for i in item:
            name, img, boxes, c = self[i]
            plt.figure(None, figsize)
            plt.imshow(img)
            for j, box in enumerate(boxes):
                plot_box(*box)
                plot_text(c[j], box[0] + 5, box[1] - 5)
            plt.show()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        return self.filenames[item], self.images[item], self.boxes[item], self.categories[item]


class BBBC041Train(_BBBC041):
    def __init__(self, directory, download=False):
        """BBBC041 Train data.

        References:
            https://bbbc.broadinstitute.org/BBBC041

        Args:
            directory: Data directory.
            download: Whether to download data.
        """
        super().__init__(directory, download=download, mode='train')


class BBBC041Test(_BBBC041):
    def __init__(self, directory, download=False):
        """BBBC041 Test data.

        References:
            https://bbbc.broadinstitute.org/BBBC041

        Args:
            directory: Data directory.
            download: Whether to download data.
        """
        super().__init__(directory, download=download, mode='test')
