import functools
from timeit import default_timer as timer
import numpy as np

__all__ = ["timer", "einsum"]

einsum = functools.partial(np.einsum, optimize=True)
