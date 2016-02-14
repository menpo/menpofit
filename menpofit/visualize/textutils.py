from menpo.visualize import print_progress as menpo_print_progress


def print_progress(iterable, prefix='', n_items=None, offset=0,
                   show_bar=True, show_count=True, show_eta=True,
                   end_with_newline=True, verbose=True):
    r"""
    Print the remaining time needed to compute over an iterable.

    To use, wrap an existing iterable with this function before processing in
    a for loop (see example).

    The estimate of the remaining time is based on a moving average of the last
    100 items completed in the loop.

    This method is identical to `menpo.visualize.print_progress`, but adds a
    `verbose` flag which allows the printing to be skipped if necessary.

    Parameters
    ----------
    iterable : `iterable`
        An iterable that will be processed. The iterable is passed through by
        this function, with the time taken for each complete iteration logged.
    prefix : `str`, optional
        If provided a string that will be prepended to the progress report at
        each level.
    n_items : `int`, optional
        Allows for ``iterator`` to be a generator whose length will be assumed
        to be `n_items`. If not provided, then ``iterator`` needs to be
        `Sizable`.
    offset : `int`, optional
        Useful in combination with ``n_items`` - report back the progress as
        if `offset` items have already been handled. ``n_items``  will be left
        unchanged.
    show_bar : `bool`, optional
        If False, The progress bar (e.g. [=========      ]) will be hidden.
    show_count : `bool`, optional
        If False, The item count (e.g. (4/25)) will be hidden.
    show_eta : `bool`, optional
        If False, The estimated time to finish (e.g. - 00:00:03 remaining)
        will be hidden.
    end_with_newline : `bool`, optional
        If False, there will be no new line added at the end of the dynamic
        printing. This means the next print statement will overwrite the
        dynamic report presented here. Useful if you want to follow up a
        print_progress with a second print_progress, where the second
        overwrites the first on the same line.
    verbose : `bool`, optional
        Printing is performed only if set to ``True``.

    Raises
    ------
    ValueError
        ``offset`` provided without ``n_items``

    Examples
    --------
    This for loop: ::

        from time import sleep
        for i in print_progress(range(100)):
            sleep(1)

    prints a progress report of the form: ::

        [=============       ] 70% (7/10) - 00:00:03 remaining
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
