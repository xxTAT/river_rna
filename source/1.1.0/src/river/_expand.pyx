# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""OpenMP-parallel matrix expansion (source of the former expand.c)."""
import numpy as np
cimport numpy as cnp
from cython.parallel import prange

cnp.import_array()


def expand_matrix_parallel(double[:, ::1] matrix, int old_res, int new_res, int length):
    cdef:
        int time = old_res // new_res
        Py_ssize_t n_rows = matrix.shape[0]
        Py_ssize_t n_cols = matrix.shape[1]
        cnp.ndarray[cnp.float64_t, ndim=2] out = np.zeros((length, length), dtype=np.float64)
        double[:, ::1] ov = out
        Py_ssize_t i, j, x, y, li, lj, ri, rj
        double val
    for i in prange(n_rows, nogil=True, schedule='dynamic'):
        li = i * time
        if li >= length:
            continue
        ri = li + time
        if ri > length:
            ri = length
        for j in range(n_cols):
            lj = j * time
            if lj >= length:
                break
            rj = lj + time
            if rj > length:
                rj = length
            val = matrix[i, j]
            for x in range(li, ri):
                for y in range(lj, rj):
                    ov[x, y] = val
    return out
