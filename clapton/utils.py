import numpy as np


def n_to_dits(n, dims):
    """
    n = ... + x_3 a_2 a_1 a_0 + x_2 a_1 a_0 + x_1 a_0 + x_0
    dims = [..., a_3, a_2, a_1, a_0]
    x = [..., x_3, x_2, x_1, x_0], 0 <= x_i < a_i
    return x
    """
    assert n <= np.prod(dims) - 1, "n cannot be represented in this basis"
    x = np.zeros(len(dims), dtype=int)
    i = 0
    while n > 0:
        x[-1-i] = n % dims[-1-i]
        n = n // dims[-1-i]
        i += 1
    return x

def dits_to_n(x, dims):
    """
    n = ... + x_3 a_2 a_1 a_0 + x_2 a_1 a_0 + x_1 a_0 + x_0
    dims = [..., a_3, a_2, a_1, a_0]
    x = [..., x_3, x_2, x_1, x_0], 0 <= x_i < a_i
    return n
    """
    assert len(x) <= len(dims), "x and dims do not have compatible dimensions"
    if len(x) < len(dims):
        x = np.concatenate(np.zeros(len(dims) - len(x)), x)
    return np.cumprod(dims[-1:0:-1]) @ x[-2::-1] + x[-1]