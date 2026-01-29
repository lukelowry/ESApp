"""
Function decorators for debugging and profiling.
"""

from functools import wraps
from time import time
from typing import Callable, TypeVar

__all__ = ['timing']

F = TypeVar('F', bound=Callable)


def timing(func: F) -> F:
    """
    Decorator that prints the execution time of a function.

    Parameters
    ----------
    func : callable
        The function to wrap.

    Returns
    -------
    callable
        Wrapped function that prints timing information.

    Examples
    --------
    >>> @timing
    ... def slow_function():
    ...     time.sleep(1)
    ...
    >>> slow_function()
    'slow_function' took: 1.0012 sec
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        elapsed = time() - start
        print(f'{func.__name__!r} took: {elapsed:.4f} sec')
        return result
    return wrapper
