"""Matrix expansion (low-resolution bins -> 1-nt grid).

`--cython` loads the OpenMP-parallel Cython kernel `river._expand`; if it is
not compiled, RIVER falls back to the vectorised NumPy implementation, which
gives identical results.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("river")


def expand_matrix_numpy(matrix, old_res, new_res, length):
    t = int(old_res / new_res)
    n_r = min(matrix.shape[0], -(-length // t))
    n_c = min(matrix.shape[1], -(-length // t))
    rr = np.arange(min(length, n_r * t)) // t
    cc = np.arange(min(length, n_c * t)) // t
    block = matrix[:n_r, :n_c]
    if len(rr) == length and len(cc) == length:
        return np.ascontiguousarray(block[np.ix_(rr, cc)], dtype=np.float64)
    out = np.zeros((length, length))
    out[:len(rr), :len(cc)] = block[np.ix_(rr, cc)]
    return out


def expand_matrix_legacy(matrix, old_res, new_res, length):
    """Original pure-Python implementation, kept for testing."""
    tmp = np.zeros((length, length))
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            time = int(old_res / new_res)
            l, r = row * time, col * time
            l_end = l + time if l + time <= length - 1 else length
            r_end = r + time if r + time <= length - 1 else length
            tmp[l:l_end, r:r_end] = matrix[row, col]
    return tmp


def get_expander(use_cython=False):
    if use_cython:
        try:
            from river._expand import expand_matrix_parallel
        except ImportError:
            log.warning("Cython extension river._expand is not compiled; "
                        "falling back to NumPy (same results).")
        else:
            def _cy(matrix, old_res, new_res, length):
                m = np.ascontiguousarray(matrix, dtype=np.float64)
                return expand_matrix_parallel(m, int(old_res), int(new_res), int(length))
            log.info("Using Cython/OpenMP matrix expansion")
            return _cy
    return expand_matrix_numpy
