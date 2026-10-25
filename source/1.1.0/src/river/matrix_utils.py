"""
Matrix helpers shared by step2/step3.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.signal import convolve2d

BEDPE_HEADER = "chrom1\tstart1\tend1\tchrom2\tstart2\tend2\n"


# ----------------------------------------------------------------- FASTA
def load_fasta(path, seq_id=None):
    seqs, cur = {}, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                cur = line[1:].split()[0]
                seqs[cur] = []
            elif cur is not None:
                seqs[cur].append(line)
    if not seqs:
        raise ValueError(f"No sequence found in {path}")
    if seq_id is None:
        seq_id = next(iter(seqs))
    if seq_id not in seqs:
        raise KeyError(f"Sequence '{seq_id}' not in {path}; available: {list(seqs)[:5]}")
    return seq_id, "".join(seqs[seq_id])


def write_fasta(path, name, seq):
    with open(path, "w") as f:
        f.write(f">{name}\n{seq}\n")


# ------------------------------------------------------- juicer dump files
def read_dump(file, res):
    data = np.loadtxt(file, usecols=(0, 1, 2), ndmin=2)
    if data.size == 0:
        raise ValueError(f"Empty contact dump: {file}")
    s = (data[:, 0] // res).astype(np.int64)
    e = (data[:, 1] // res).astype(np.int64)
    v = data[:, 2].copy()
    v[np.isnan(v)] = 0
    v[v < 0] = 0
    return s, e, v


def dump_to_dense(s, e, v):
    """Symmetric dense matrix (same as original get_matrix_npz)."""
    s1 = sparse.coo_matrix((v, (s, e))).toarray()
    s2 = sparse.coo_matrix((v, (e, s))).toarray()
    n = max(s1.shape)
    m = np.zeros((n, n))
    nz = np.nonzero(s1)
    m[nz] = s1[nz]
    nz = np.nonzero(s2)
    m[nz] = s2[nz]
    return m


def load_dump_matrix(file, res):
    return dump_to_dense(*read_dump(file, res))


def zero_diagonal_cross(m):
    """Original step2 treatment of the low-resolution matrix."""
    w = m.shape[0]
    for i in range(1, w - 1):
        m[i, i] = m[i - 1, i] = m[i, i - 1] = m[i, i + 1] = m[i + 1, i] = 0
    return m


def get_oe(matrix):
    """Observed / expected per diagonal"""
    matrix = np.array(matrix).copy()
    lens = matrix.shape[0]
    mask = np.ones(lens)
    cut_off = lens * 0.0001
    matrix_num = matrix.copy()
    matrix_num[matrix_num > 0] = 1
    col_sum = matrix_num.sum(axis=0)
    sites = np.arange(lens)
    mask[col_sum <= cut_off] = 0
    mask = mask == 1
    x_o, y_o = sites.copy(), sites.copy()
    dis = np.zeros(lens)
    mask_o = mask.copy()
    exp_matrix = np.ones((lens, lens))
    x_e, y_e = sites.copy(), sites.copy()
    for i in range(lens):
        diag_n = matrix[(x_o, y_o)]
        diag_line = diag_n[mask_o]
        if diag_line.shape[0] == 0:
            dis[i] = 0
        else:
            dis[i] = diag_line.mean()
            x_o = np.delete(x_o, -1)
            y_o = np.delete(y_o, 0)
            mask_o = np.delete(mask_o, -1)
        if dis[i] == 0:
            x_e = np.delete(x_e, -1)
            y_e = np.delete(y_e, 0)
            continue
        exp_matrix[(x_e, y_e)] = dis[i]
        exp_matrix[(y_e, x_e)] = dis[i]
        x_e = np.delete(x_e, -1)
        y_e = np.delete(y_e, 0)
    return np.asarray(matrix / exp_matrix)


def mask_by_oe(mat, oe):
    """Set cells with O/E <= 1 to zero"""
    h = min(mat.shape[0], oe.shape[0])
    w = min(mat.shape[1], oe.shape[1])
    region = mat[:h, :w]
    region[oe[:h, :w] <= 1] = 0
    return mat


def pad_or_crop(m, n):
    if m.shape == (n, n):
        return m
    out = np.zeros((n, n), dtype=m.dtype)
    h, w = min(m.shape[0], n), min(m.shape[1], n)
    out[:h, :w] = m[:h, :w]
    return out


# --------------------------------------------------------- canny pieces
def gaussian_kernel(sigma, ksize):
    ksize = int(ksize)
    k = np.zeros((ksize, ksize))
    for i in range(ksize):
        for j in range(ksize):
            k[i, j] = np.round(math.exp((-1 / (2 * sigma ** 2)) * ((i - 1) ** 2 + (j - 1) ** 2))
                               / (2 * math.pi * sigma ** 2), 4)
    k /= np.sum(k)
    return np.round(k, 4)


def gaussian_blur(m, kernel):
    return convolve2d(m, kernel, mode="same", boundary="fill", fillvalue=0)


def nms(m):
    out = np.zeros_like(m)
    if m.shape[0] < 3 or m.shape[1] < 3:
        return out
    c = m[1:-1, 1:-1]
    keep = (c != 0) & (c >= m[:-2, :-2]) & (c >= m[2:, 2:])
    out[1:-1, 1:-1] = np.where(keep, c, 0)
    return out


def double_threshold(n, tlf=0.01, thf=0.03):
    W, H = n.shape
    mx = np.max(n)
    tl, th = tlf * mx, thf * mx
    dt = np.zeros((W, H), dtype=bool)
    if W < 3 or H < 3:
        return dt.astype(int), tl, th
    strong = np.zeros((W, H), bool)
    weak = np.zeros((W, H), bool)
    strong[1:-1, 1:-1] = n[1:-1, 1:-1] > th
    weak[1:-1, 1:-1] = n[1:-1, 1:-1] > tl
    cand = weak & ~strong
    idx = np.arange(H)
    # ---- pass 1 (Gauss-Seidel from empty) ----
    for i in range(1, W - 1):
        prev = dt[i - 1]
        up = np.zeros(H, bool)
        up[1:] = prev[:-1] | prev[1:]
        base = strong[i] | (cand[i] & up)
        allowed = strong[i] | cand[i]
        last_base = np.maximum.accumulate(np.where(base, idx, -1))
        last_break = np.maximum.accumulate(np.where(~allowed, idx, -1))
        dt[i] = allowed & (last_base > last_break)
    n_strong = int(strong.sum())
    if int(dt.sum()) == n_strong:
        return dt.astype(int), tl, th
    # ---- remaining passes: least fixed point ----
    while True:
        nb = np.zeros((W, H), bool)
        nb[1:-1, 1:-1] = (dt[:-2, :-2] | dt[:-2, 1:-1] | dt[2:, :-2] | dt[2:, 1:-1]
                          | dt[1:-1, :-2] | dt[1:-1, 2:])
        new = dt | (cand & nb)
        if np.array_equal(new, dt):
            break
        dt = new
    return dt.astype(int), tl, th


# ----------------------------------------------------------- bedpe I/O
def write_pixel_bedpe(path, rows, cols, res, chrom, header=True):
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    with open(path, "w") as f:
        if header:
            f.write(BEDPE_HEADER)
        for r, c in zip(rows.tolist(), cols.tolist()):
            f.write(f"{chrom}\t{r * res}\t{r * res + res}\t{chrom}\t{c * res}\t{c * res + res}\n")


def read_bedpe_starts(path):
    """Return integer arrays (start1, end1, start2, end2), header skipped."""
    s1, e1, s2, e2 = [], [], [], []
    with open(path) as f:
        for line in f:
            if "chrom1" in line or not line.strip():
                continue
            sub = line.split()
            s1.append(int(sub[1])); e1.append(int(sub[2]))
            s2.append(int(sub[4])); e2.append(int(sub[5]))
    return (np.array(s1, dtype=np.int64), np.array(e1, dtype=np.int64),
            np.array(s2, dtype=np.int64), np.array(e2, dtype=np.int64))


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
