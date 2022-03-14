from h5py import File
from os.path import isfile

__all__ = ['GenericH5']


class GenericH5:
    def __init__(self, filenames, keys, cache=False):
        """Generic hdf5 dataset.

        Basic interface for a set of hdf5 files.

        Examples:
            >>> import celldetection as cd
            ... from glob import glob
            ... data = GenericH5(glob('./files/*.h5'), ('image', 'labels'))
            ... name, (img, lbl) = data[0]

        Args:
            filenames: Hdf5 file names.
            keys: Keys. Used to retrieve data from each file.
            cache: Whether to hold data in memory.
        """
        self._filenames = list(filenames)
        self._filenames.sort()
        self.content = list(self._filenames)
        for f in self.content:
            if not isfile(f):
                raise FileNotFoundError(f'File not found: {f}')
        self._single = isinstance(keys, str)
        self.keys = (keys,) if self._single else keys
        self.cache = cache

    def __getitem__(self, item):
        it = self.content[item]
        if isinstance(it, str):
            with File(it, 'r') as h:
                it = [h[k][:] for k in self.keys]
            if self._single:
                it, = it
            if self.cache:
                self.content[item] = it
        return self._filenames[item], it

    def __len__(self):
        return len(self.content)

    def __str__(self):
        return f'{len(self)} hdf5 files'

    __repr__ = __str__
