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


class PickleWrappedFitter(object):
    r"""
    Wrapper around a menpofit fitter so that we can a) efficiently pickle it
    and b) parametrize over both the fitter construction and the fit
    methods (e.g. ``.fit_from_bb()`` and ``.fit_from_shape()``)

    Pickling menpofit fitters is a little tricky for a two reasons. Firstly,
    on construction of a fitter from a deformable model some amount of
    pre-processing takes place which allocates potentially large arrays. To
    ship a compact model we would therefore rather delay the construction of
    the fitter until load time on the client.

    If this was the only issue, we could achieve this by simply saving a
    partial over the fitter constructor with all the ``args`` and ``kwargs``
    the fitter constructor takes - after loading the pickle, invoking the
    partial with no args (it's parameters being fully specified) would return
    the fitter and all would be well.

    However, we also may want to choose **fit-time** parameters for the fitter
    for optimal usage, (for instance, a choice over the ``max_iters``
    kwarg that we know to be efficient). This leaves us with a problem,
    as now we need to have some entity that can store state which we can pass
    to both the fitter and to the resulting fitters methods on the client at
    unpickle time.

    This class is the solution to this problem. To use, you should **pickle
    down a partial over this class** specifying all arguments and kwargs
    needed for the fitter constructor and for the fit methods.

    At load time, menpofit will invoke the partial, returning this object
    instantiated. This offers the same API as a menpofit fitter, and so can
    be used transparently to fit. If you wish to access the original fitter
    (without fit parameter customization) this can be accessed as the
    `wrapped_fitter` property.

    Parameters
    ----------
    fitter_cls : `Fitter`
        A menpofit fitter class that will be constructed at unpickle time,
        e.g. :map:`LucasKanadeAAMFitter`
    fitter_args : `tuple`
        A tuple of all args that need to be passed to ``fitter_cls`` at
        construction time e.g. ``(aam,)``
    fitter_kwargs : `dict`
        A dictionary of kwargs that will to be passed to ``fitter_cls`` at
        construction time e.g.
        ``{ 'lk_algorithm_cls': WibergInverseCompositional }``
    fit_from_bb_kwargs : `dict`, e.g. ``{ max_iters: [25, 5] }``
        A dictionary of kwargs that will to be passed to the
        wrapped fitter's ``fit_from_bb`` method at fit time. These in effect
        change the defaults that the original fitter offered, but can still
        be overridden at call time (e.g.
        ``self.fit_from_bb(image, bbox, max_iters=[50, 50])`` would take
        precedence over the max_iters in the above example)
    fit_from_shape_kwargs : `dict`, e.g. ``{ max_iters: [25, 5] }``
        A dictionary of kwargs that will to be passed to the
        wrapped fitter's ``fit_from_shape`` method at fit time. These in
        effect change the defaults that the original fitter offered,
        but can still be overridden at call time (e.g.
        ``self.fit_from_shape(image, shape, max_iters=[50, 50])`` would take
        precedence over the max_iters in the above example)
    """
    def __init__(self, fitter_cls, fitter_args, fitter_kwargs,
                 fit_from_bb_kwargs, fit_from_shape_kwargs):
        self.wrapped_fitter = fitter_cls(*fitter_args, **fitter_kwargs)
        self._fit_from_bb_kwargs = fit_from_bb_kwargs
        self._fit_from_shape_kwargs = fit_from_shape_kwargs

    def fit_from_bb(self, image, bounding_box, **kwargs):
        r"""
        Fits the fitter to an image given an initial bounding box, using the
        optimal parameters that we chosen for this pickled fitter.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image to be fitted.
        bounding_box : `menpo.shape.PointDirectedGraph`
            The initial bounding box from which the fitting procedure will
            start. Note that the bounding box is used in order to align the
            model's reference shape.
        kwargs : `dict`, optional
            Other kwargs to override the optimal defaults. See the
            documentation for ``.fit_from_bb()`` on the type of
            `self.wrapped_fitter` to see what can be provided here.

        Returns
        -------
        fitting_result : ``FittingResult`` or subclass
            The fitting result containing the result of the fitting
            procedure.
        """
        # start with the optimal kwargs
        final_kwargs = self._fit_from_bb_kwargs.copy()
        # If the user provided kwargs at runtime, they take precedence.
        final_kwargs.update(kwargs)
        # call the wrapped fitter with the updated kwargs
        return self.wrapped_fitter.fit_from_bb(image, bounding_box,
                                               **final_kwargs)

    def fit_from_shape(self, image, initial_shape, **kwargs):
        r"""
        Fits the fitter to an image given an initial shape, using the
        optimal parameters that we chosen for this pickled fitter.

        Parameters
        ----------
        image : `menpo.image.Image` or subclass
            The image to be fitted.
        initial_shape : `menpo.shape.PointCloud`
            The initial shape estimate from which the fitting procedure
            will start.
        kwargs : dict
            Other kwargs to override the optimal defaults. See the
            documentation for ``.fit_from_shape()`` on the type of
            `self.wrapped_fitter` to see what can be provided here.

        Returns
        -------
        fitting_result : ``FittingResult`` or subclass
            The fitting result containing the result of the fitting
            procedure.
        """
        # start with the optimal kwargs
        final_kwargs = self._fit_from_shape_kwargs.copy()
        # If the user provided kwargs at runtime, they take precedence.
        final_kwargs.update(kwargs)
        # call the wrapped fitter with the updated kwargs
        return self.wrapped_fitter.fit_from_shape(image, initial_shape,
                                                  **final_kwargs)


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
