from __future__ import absolute_import  # or menpofit.math causes trouble!
from math import ceil
import sys

try:
    from urllib2 import urlopen  # Py2
except ImportError:
    from urllib.request import urlopen  # Py3

from menpofit.base import menpofit_src_dir_path
from menpo.io import import_pickle

# The remote URL that should be queried to download pretrained models
MENPO_URL = 'http://static.menpo.org'

# All remote models are versioned based on a binary compatibility number.
# This needs to be bumped every time an existing remote model is no longer
# Compatible with this version of menpofit.
MENPOFIT_BINARY_VERSION = 0


def menpofit_data_dir_path():
    r"""
    Returns a path to the ./data directory, creating it if needed.

    Returns
    -------
    path : `Path`
        The path to the data directory (which will be instantiated)
    """
    data_path = menpofit_src_dir_path() / 'data'
    if not data_path.exists():
        data_path.mkdir()
    return data_path


def filename_for_fitter(name):
    r"""
    Returns the filename of a fitter with the identifier ``name`` accounting
    for the Python version and menpofit binary compatibility version number.

    Parameters
    ----------
    name : `str`
        The name of the fitter, e.g. `balanced_frontal_face_aam`

    Returns
    -------
    filename : `str`
        The filename of the fitter
    """
    return '{}_v{}_py{}.pkl'.format(name, MENPOFIT_BINARY_VERSION,
                                    sys.version_info.major)


def url_of_fitter(name):
    r"""
    Returns the remote URL for a fitter with identifier ``name``.

    Parameters
    ----------
    name : `str`
        The identifier of the fitter, e.g. `balanced_frontal_face_aam`

    Returns
    -------
    url : `str`
        The URL that this fitter is stored at for this installation of
        menpofit
    """
    return '{}/menpofit/{}'.format(MENPO_URL, filename_for_fitter(name))


def path_of_fitter(name):
    r"""
    Returns a path to where a fitter with identifier ``name`` can be
    located on disk.

    Parameters
    ----------
    name : `str`
        The identifier of the fitter, e.g. `balanced_frontal_face_aam`

    Returns
    -------
    path :  `Path`
        A path to this fitter on disk
    """
    return menpofit_data_dir_path() / filename_for_fitter(name)


def copy_and_yield(fsrc, fdst, length=1024*1024):
    """copy data from file-like object fsrc to file-like object fdst"""
    while 1:
        buf = fsrc.read(length)
        if not buf:
            break
        fdst.write(buf)
        yield


def download_file(url, destination, verbose=False):
    r"""
    Download a file from a URL to a path, optionally reporting the progress

    Parameters
    ----------
    url : `str`
        The URL of a remote resource that should be downloaded
    destination : `Path`
        The path on disk that the file will be downloaded to
    verbose : `bool`, optional
        If ``True``, report the progress of the download dynamically.
    """
    from menpo.visualize.textutils import print_progress, bytes_str
    req = urlopen(url)
    chunk_size_bytes = 512 * 1024

    with open(str(destination), 'wb') as fp:

        # Retrieve a generator that we can keep yielding from to download the
        # file in chunks.
        copy_progress = copy_and_yield(req, fp, length=chunk_size_bytes)

        if verbose:
            # wrap the download object with print progress to log the status
            n_bytes = int(req.headers['content-length'])
            n_items = int(ceil((1.0 * n_bytes) / chunk_size_bytes))
            prefix = 'Downloading {}'.format(bytes_str(n_bytes))
            copy_progress = print_progress(copy_progress, n_items=n_items,
                                           show_count=False, prefix=prefix)

        for _ in copy_progress:
            pass

    req.close()


def load_fitter(name):
    r"""
    Load a fitter with identifier ``name``, pulling it from a remote URL if
    needed.

    Parameters
    ----------
    name : `str`
        The identifier of the fitter, e.g. `balanced_frontal_face_aam`

    Returns
    -------
    fitter : `Fitter`
        A pretrained menpofit `Fitter` that is ready to use.
    """
    path = path_of_fitter(name)
    if not path.exists():
        print('Downloading {} fitter'.format(name))
        url = url_of_fitter(name)
        print(url, path)
        download_file(url, path, verbose=True)
    # Load the pickle and immediately invoke it.
    try:
        return import_pickle(path)()
    except Exception as e:
        # Hmm something went wrong, and we couldn't load this fitter.
        # Purge it so next time we will redownload.
        print('Error loading fitter - purging damaged file: {}'.format(path))
        print('Please try running again')
        path.unlink()
        raise e
