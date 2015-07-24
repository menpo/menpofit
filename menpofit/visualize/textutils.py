from menpo.visualize import print_progress as menpo_print_progress


def print_progress(iterable, prefix='', n_items=None, offset=0,
                   show_bar=True, show_count=True, show_eta=True,
                   end_with_newline=True, verbose=True):
    r"""
    Please see the menpo ``print_progress`` documentation.

    This method is identical to the print progress method, but adds a verbose
    flag which allows the printing to be skipped if necessary.
    """
    if verbose:
        # Yield the images from the menpo print_progress (yield from would
        # be perfect here :( )
        for i in menpo_print_progress(iterable, prefix=prefix, n_items=n_items,
                                      offset=offset, show_bar=show_bar,
                                      show_count=show_count, show_eta=show_eta,
                                      end_with_newline=end_with_newline):
            yield i
    else:
        # Skip the verbosity!
        for i in iterable:
            yield i
