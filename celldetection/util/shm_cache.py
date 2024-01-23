import hashlib
from os.path import isfile, dirname, basename, isdir, join, splitext, islink, abspath
from os import remove, makedirs, rmdir, unlink, symlink, cpu_count
from shutil import copy2
from concurrent.futures import ThreadPoolExecutor
from warnings import warn
import pytorch_lightning as pl
from .util import compare_file_hashes

__all__ = ['ShmCache']


def get_hash(filename):
    hash_obj = hashlib.sha256()
    hash_obj.update(abspath(filename).encode())
    return hash_obj.hexdigest()


def get_dst(filename, root):
    bn = basename(filename)
    dn = dirname(filename)
    cache_dir = join(root, get_hash(dn))
    dst = join(cache_dir, get_hash(bn)) + splitext(bn)[1]
    return cache_dir, dst


def _resolve_filename(filename, root):
    postfix = ''
    if isinstance(filename, dict):  # allow to pass dict, specifying postfix and filename
        postfix = filename.get('postfix', postfix)
        filename = filename['filename']
    cache_dir, dst = get_dst(filename[:-len(postfix)] if len(postfix) else filename, root)
    dst = dst + postfix
    return cache_dir, dst, filename


def copy_file(filename, root, check_hash=True, verbose=True):
    cache_dir, dst, filename = _resolve_filename(filename, root)
    makedirs(cache_dir, exist_ok=True)
    if verbose:
        print('Copy', filename, '-->', dst)
    copy2(filename, dst)
    if check_hash:
        assert compare_file_hashes(filename, dst), f'File hashes did not match after copying: {filename}, {dst}'
    return dst


def link_file(filename, root, verbose=True):
    cache_dir, dst, filename = _resolve_filename(filename, root)
    makedirs(cache_dir, exist_ok=True)
    if verbose:
        print('Link', abspath(filename), 'as', abspath(dst))
    if not islink(dst):
        try:
            symlink(abspath(filename), abspath(dst))
        except Exception as e:
            print(f"Error creating symlink for {filename} at {dst}: {e}")
            return None
    elif verbose:
        print('Already linked', abspath(filename), 'as', abspath(dst))
    return dst


def delete_file_or_directory(filename, root, verbose=True):
    if filename is None:
        return True
    import os
    assert filename.startswith(root) and '..' not in filename
    try:
        if islink(filename):
            if verbose:
                print('Unlink', filename)
            unlink(filename)
        elif isdir(filename):
            assert os.path.realpath(filename).startswith(root)
            if verbose:
                print('Delete', filename)
            rmdir(filename)
        elif isfile(filename):
            assert os.path.realpath(filename).startswith(root)
            if verbose:
                print('Delete', filename)
            remove(filename)
        elif verbose:
            print('Already gone:', filename)
        return True
    except Exception as e:
        print(f"Error deleting {filename}: {e}")
        return False


class ShmCache:
    def __init__(self, filenames, link_filenames=None, directory='/dev/shm/celldetection', verbose=True):
        """Shared Memory Cache.

        Creates copies or symbolic links for all files in `filenames` and `link_filenames`, respectively.
        This serves as an easy-to-use interface for caching training data.

        Note:
            When using `ShmCache` in a distributed environment, failing processes should call `teardown`
            to remove all cached files before the script terminates. Note that other running processes
            will not find any cached files after `teardown` has been called. Hence, it is expected
            that exceptions from failing ranks are accompanied by `OSError: Unable to open file (open failed)`
            or `FileNotFoundError` from running ranks.

        Args:
            filenames: Filenames (to be copied).
            link_filenames: Filenames (to be symbolically linked).
            directory: Target directory.
            verbose: Verbosity.
        """
        self.filenames = filenames
        self.link_filenames = link_filenames
        self.directory = directory
        self.cached_files = None
        self.linked_files = None
        self.subs = None
        self.verbose = verbose

    @property
    def setup_complete(self):
        return self.subs is not None

    def setup(self, max_workers=None):
        """Setup.

        Copies all files specified in the constructor via `filenames`.
        Also creates symlinks for all files specified in the constructor via `link_filenames`.

        Args:
            max_workers: Max number of workers.

        Returns:
            List of cached file names.
        """
        if max_workers is None:
            max_workers = cpu_count() or 1

        if not isdir(self.directory):
            if self.verbose:
                print('Creating', self.directory)
            makedirs(self.directory, exist_ok=True)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            self.subs = set()
            if self.filenames is not None:
                self.cached_files = list(
                    executor.map(lambda f: copy_file(f, self.directory, verbose=self.verbose), self.filenames))
                self.subs.update({dirname(f) for f in self.cached_files})
            if self.link_filenames is not None:
                self.linked_files = list(
                    executor.map(lambda f: link_file(f, self.directory, verbose=self.verbose), self.link_filenames))
                self.subs.update({dirname(f) for f in self.linked_files})

        return self.cached_files

    def teardown(self, max_workers=None):
        """Teardown.

        Removes all copies, unlinks all symlinks and removes all subdirectories that contained copies or symlinks.

        Args:
            max_workers: Max number of workers.
        """
        if max_workers is None:
            max_workers = cpu_count() or 1

        if self.cached_files is None:
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Remove cached files
            results = list(executor.map(lambda f: delete_file_or_directory(f, self.directory, verbose=self.verbose),
                                        self.cached_files))

            # Remove linked files
            if self.linked_files is not None:
                results = list(executor.map(lambda f: delete_file_or_directory(f, self.directory, verbose=self.verbose),
                                            self.linked_files))

            # Remove cached dirs (must be empty)
            if self.subs is not None:
                results += list(
                    executor.map(lambda f: delete_file_or_directory(f, self.directory, verbose=self.verbose),
                                 self.subs))

        self.cached_files = self.linked_files = self.subs = None

        if not all(results):
            warn(f'Could not remove all cached files in {self.directory}. Removed {sum(results)}/{len(results)}')
