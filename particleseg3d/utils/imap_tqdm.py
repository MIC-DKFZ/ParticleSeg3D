from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


def imap_tqdm(function, iterable, processes, output=None, chunksize=1, desc=None, disable=False, **kwargs):
    """
    Run a function in parallel with a tqdm progress bar and an arbitrary number of arguments.
    Results are always ordered and the performance should be the same as of Pool.map.
    TODO: Still needs more performance testing
    :param function: The function that should be parallelized.
    :param iterable: The iterable passed to the function.
    :param processes: The number of processes used for the parallelization.
    :param chunksize: The iterable is based on the chunk size chopped into chunks and submitted to the process pool as separate tasks.
    :param desc: The description displayed by tqdm in the progress bar.
    :param disable: Disables the tqdm progress bar.
    :param kwargs: Any additional arguments that should be passed to the function.
    """
    if kwargs:
        function_wrapper = partial(wrapper, function=function, **kwargs)
    else:
        function_wrapper = partial(wrapper, function=function)

    if output is None:
        output = [None] * len(iterable)
    with Pool(processes=processes) as p:
        with tqdm(desc=desc, total=len(iterable), disable=disable) as pbar:
            for i, result in p.imap_unordered(function_wrapper, enumerate(iterable), chunksize=chunksize):
                output[i] = result
                pbar.update()
    return output


def wrapper(enum_iterable, function, **kwargs):
    i = enum_iterable[0]
    result = function(enum_iterable[1], **kwargs)
    return i, result