"""Data structures shared by step4 and step5.

A *cluster* is one structure feature of step 2: the bounding box
(start1, end1, start2, end2) of a group of connected Canny pixels.
Both new steps need.

* the clusters themselves and their geometry (diagonal, span, overlap),
* the interaction support of sub-matrices (`SupportMatrices`),
* the candidate base-pair pixels = Canny feature  AND  partition probability

all of which are kept sparse/at their own resolution: keeps the memory of step4/5
below ~1 GB even for a 30 kb genome.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import sparse

from .matrix_utils import (BEDPE_HEADER, get_oe, load_dump_matrix, mask_by_oe, read_bedpe_starts, read_dump)

log = logging.getLogger("river")


# ------------------------------------------------------------------ clusters
def load_clusters(bedpe):
    """(start1, end1, start2, end2) of every structure feature, deduplicated."""
    s1, e1, s2, e2 = read_bedpe_starts(bedpe)
    out = {(int(a), int(b), int(c), int(d)) for a, b, c, d in zip(s1, e1, s2, e2)
           if b > a and d > c}
    return sorted(out)


def cluster_name(c):
    return "_".join(str(int(x)) for x in c)


def span(c):
    """Genomic span of a cluster"""
    return int(c[3]) - int(c[0])


def is_diagonal(c, gap):
    """True when the two arms touch"""
    return int(c[1]) + gap > int(c[2]) - gap


def region_of(c):
    """Region that has to be folded for a diagonal feature"""
    return min(int(c[0]), int(c[2])), max(int(c[1]), int(c[3]))


def write_clusters_bedpe(path, clusters, chrom, names=None):
    with open(path, "w") as f:
        f.write(BEDPE_HEADER)
        for i, c in enumerate(clusters):
            extra = f"\t{names[i]}" if names else ""
            f.write(f"{chrom}\t{int(c[0])}\t{int(c[1])}\t{chrom}\t"
                    f"{int(c[2])}\t{int(c[3])}{extra}\n")


def classify_clusters(clusters, params):
    """Split into long-range, short-range non-diagonal and short-range diagonal"""
    long_range, nondiag, diag = [], [], []
    for c in clusters:
        if span(c) > params.cluster_max_distance:
            long_range.append(c)
        elif is_diagonal(c, params.cluster_diagonal_gap):
            diag.append(c)
        else:
            nondiag.append(c)
    return long_range, nondiag, diag


def overlaps(a0, a1, b0, b1, fraction=0.5):
    """containment, or overlap > `fraction` of either side."""
    if (a0 <= b0 and a1 >= b1) or (a0 >= b0 and a1 <= b1):
        return True
    if a0 <= b0 <= a1:
        ov = a1 - b0
    elif b0 <= a0 <= b1:
        ov = b1 - a0
    else:
        return False
    return ov > (a1 - a0) * fraction or ov > (b1 - b0) * fraction


# -------------------------------------------------------------- interaction data
def load_dump_sparse(file, res, length=None):
    """Symmetric CSR contact matrix"""
    s, e, v = read_dump(file, res)
    n = int(max(s.max(), e.max())) + 1
    if length:
        n = max(n, -(-int(length) // int(res)) + 1)
    off = s != e
    rows = np.concatenate([s, e[off]])
    cols = np.concatenate([e, s[off]])
    vals = np.concatenate([v, v[off]])
    key = rows.astype(np.int64) * n + cols
    order = np.lexsort((np.arange(key.size), key))
    key_sorted = key[order]
    keep = np.ones(key.size, bool)
    keep[:-1] = key_sorted[:-1] != key_sorted[1:]
    sel = order[keep]
    return sparse.csr_matrix((vals[sel], (rows[sel], cols[sel])), shape=(n, n))


def expand_block(m, res, r0, r1, c0, c1):
    """Dense 1-nt block of a low-resolution matrix (== expand + slice)."""
    r0, r1, c0, c1 = int(r0), int(r1), int(c0), int(c1)
    R0, C0 = r0 // res, c0 // res
    R1, C1 = (r1 - 1) // res + 1, (c1 - 1) // res + 1
    sub = np.zeros((R1 - R0, C1 - C0))
    hr, hc = min(R1, m.shape[0]), min(C1, m.shape[1])
    if hr > R0 and hc > C0:
        sub[:hr - R0, :hc - C0] = m[R0:hr, C0:hc].toarray()
    rows = np.arange(r0, r1) // res - R0
    cols = np.arange(c0, c1) // res - C0
    return sub[np.ix_(rows, cols)]


class SupportMatrices:
    """interaction support merged over `score_res` (10/5/1 bp default)"""

    def __init__(self, matrix_files, score_res, length):
        self.length = int(length)
        self.res = tuple(sorted({int(r) for r in score_res}, reverse=True))
        self.mats = {}
        for r in self.res:
            if r not in matrix_files:
                raise KeyError(f"no {r} bp contact matrix from step 2")
            self.mats[r] = load_dump_sparse(matrix_files[r], r, length)
        self.coarse = self.res[0]

    def block(self, r0, r1, c0, c1):
        out = np.zeros((int(r1) - int(r0), int(c1) - int(c0)))
        for r, m in self.mats.items():
            out += expand_block(m, r, r0, r1, c0, c1)
        return out

    def block_at(self, res, r0, r1, c0, c1):
        return expand_block(self.mats[int(res)], int(res), r0, r1, c0, c1)

    def block_sum(self, res, r0, r1, c0, c1):
        """Sum of the expanded block without materialising it."""
        res = int(res)
        m = self.mats[res]
        r0, r1, c0, c1 = int(r0), int(r1), int(c0), int(c1)
        R = np.arange(r0, r1) // res
        C = np.arange(c0, c1) // res
        Rb, rw = np.unique(R, return_counts=True)
        Cb, cw = np.unique(C, return_counts=True)
        ok_r, ok_c = Rb < m.shape[0], Cb < m.shape[1]
        Rb, rw, Cb, cw = Rb[ok_r], rw[ok_r], Cb[ok_c], cw[ok_c]
        if not Rb.size or not Cb.size:
            return 0.0
        sub = m[Rb.min():Rb.max() + 1, Cb.min():Cb.max() + 1].toarray()
        sub = sub[np.ix_(Rb - Rb.min(), Cb - Cb.min())]
        return float(rw @ sub @ cw)

    def dense(self, res):
        return self.mats[int(res)].toarray()

    def value(self, a, b, res=None):
        """Single interaction value, used for the per-nucleotide annotations."""
        r = int(res or self.coarse)
        m = self.mats[r]
        i, j = int(a) // r, int(b) // r
        if i >= m.shape[0] or j >= m.shape[1]:
            return 0.0
        return float(m[i, j])

    def pair_value(self, a, b):
        """interaction support of one base pair, merged over `score_res`."""
        return sum(self.value(a, b, r) for r in self.res)

    def annotator(self, res=None):
        r = int(res or self.coarse)
        m = self.mats[r].tocsr()

        def _f(a, b):
            i, j = int(a) // r, int(b) // r
            if i >= m.shape[0] or j >= m.shape[1]:
                return 0.0
            return float(m[i, j])
        return _f


# --------------------------------------------------- partition / feature map
def build_partition_sparse(list_file, txt_dir, length):
    """-log10 P matrix as CSR; overlapping windows averaged"""
    from pathlib import Path

    cur = None
    for name in Path(list_file).read_text().split():
        parts = name.replace(".txt", "").split("_")
        ls, rs = int(parts[0]), int(parts[2])
        d = np.loadtxt(Path(txt_dir) / name, skiprows=2, ndmin=2)
        if d.size == 0:
            continue
        keep = d[:, 2] > 0
        r = d[keep, 0].astype(np.int64) - 1 + ls
        c = d[keep, 1].astype(np.int64) - 1 + rs
        ok = (r < length) & (c < length) & (r >= 0) & (c >= 0)
        m = sparse.csr_matrix((d[keep, 2][ok], (r[ok], c[ok])), shape=(length, length))
        if cur is None:
            cur = m
            continue
        both = cur.astype(bool).multiply(m.astype(bool)).tocsr()
        avg = (cur + m).multiply(both).tocsr() * 0.5
        cur = (cur - cur.multiply(both) + m - m.multiply(both) + avg).tocsr()
        cur.eliminate_zeros()
    if cur is None:
        raise RuntimeError("no partition probabilities found; run step 1 first")
    return cur.tocoo()


class CannyGrid:
    """Canny feature pixels of step 2 on the mid-resolution grid, O/E masked."""

    def __init__(self, feature_bedpe, mid_res, length, oe_matrix_file=None):
        self.res = int(mid_res)
        n = -(-int(length) // self.res) + 2
        s1, _, s2, _ = read_bedpe_starts(feature_bedpe)
        r, c = s1 // self.res, s2 // self.res
        if len(r):
            n = max(n, int(r.max()) + 1, int(c.max()) + 1)
        self.grid = np.zeros((n, n), dtype=bool)
        self.grid[r, c] = True
        if oe_matrix_file is not None:
            oe = get_oe(load_dump_matrix(oe_matrix_file, self.res))
            m = self.grid.astype(float)
            mask_by_oe(m, oe)
            self.grid = m > 0

    def at(self, rows, cols):
        r = np.asarray(rows) // self.res
        c = np.asarray(cols) // self.res
        ok = (r < self.grid.shape[0]) & (c < self.grid.shape[1])
        out = np.zeros(r.shape, dtype=bool)
        out[ok] = self.grid[r[ok], c[ok]]
        return out

    def has_support(self, rect):
        s1, e1, s2, e2 = (int(x) for x in rect)
        r0, r1 = s1 // self.res, (e1 - 1) // self.res + 1
        c0, c1 = s2 // self.res, (e2 - 1) // self.res + 1
        r1 = min(r1, self.grid.shape[0])
        c1 = min(c1, self.grid.shape[1])
        if r1 <= r0 or c1 <= c0:
            return False
        return bool(self.grid[r0:r1, c0:c1].any())


class FeaturePixels:
    """Candidate base pairs: feature pixel AND partition probability > 0"""

    def __init__(self, rows, cols, vals, length):
        order = np.lexsort((cols, rows))
        self.rows = rows[order]
        self.cols = cols[order]
        self.vals = vals[order]
        self.length = int(length)
        self.value = dict(zip(zip(self.rows.tolist(), self.cols.tolist()), self.vals.tolist()))

    @classmethod
    def build(cls, params, step1_list, feature_bedpe, oe_matrix_file, length,
              diag_zero_width=None, max_pair_distance=None):
        from pathlib import Path

        part = build_partition_sparse(step1_list, Path(step1_list).parent / "tmp_pfs_txt", length)
        r, c, v = part.row, part.col, part.data
        w = params.diag_zero_width if diag_zero_width is None else diag_zero_width
        keep = np.abs(r - c) >= int(w)
        if max_pair_distance:
            keep &= np.abs(r - c) < int(max_pair_distance)
        r, c, v = r[keep], c[keep], v[keep]
        canny = CannyGrid(feature_bedpe, params.mid_res, length, oe_matrix_file)
        keep = canny.at(r, c)
        log.info("  candidate pixels: %d", int(keep.sum()))
        return cls(r[keep], c[keep], v[keep], length), canny

    def in_rect(self, r0, r1, c0, c1):
        """Row-major ordered pixels inside [r0, r1) x [c0, c1)."""
        lo = np.searchsorted(self.rows, int(r0), "left")
        hi = np.searchsorted(self.rows, int(r1), "left")
        r = self.rows[lo:hi]
        c = self.cols[lo:hi]
        sel = (c >= int(c0)) & (c < int(c1))
        return list(zip(r[sel].tolist(), c[sel].tolist()))

    def in_rects(self, rects):
        out = set()
        for rect in rects:
            out.update(self.in_rect(*rect))
        return sorted(out)

    def write_bedpe(self, path, chrom):
        with open(path, "w") as f:
            f.write(BEDPE_HEADER)
            for a, b in zip(self.rows.tolist(), self.cols.tolist()):
                f.write(f"{chrom}\t{a}\t{a + 1}\t{chrom}\t{b}\t{b + 1}\n")


# ------------------------------------------------------------------ scoring
def score_units(unit_loc, support, prob, params):
    """sum over the stem members of  m + m / (-log10 P).
    `support` is the merged interaction block clipped at `score_clip_percentile`;
    `prob(i, j)` returns the partition / bipartition probability of a pair.
    Only the units above `stem_score_percentile` are kept.
    """
    if not unit_loc:
        return []
    scores = []
    for uid, locs in unit_loc.items():
        count = 0.0
        for rc in locs:
            m = float(support(*rc))
            p = float(prob(*rc))
            count += m + (m / p if p else 0.0)
        scores.append((uid, count))
    cut = np.percentile([s for _, s in scores], params.unit_score_percentile)
    return [(u, s) for u, s in scores if s >= cut]


def clipped_block(block, percentile):
    """Clip the interaction support at a percentile."""
    out = np.array(block, dtype=float, copy=True)
    if out.size:
        np.minimum(out, np.percentile(out, percentile), out=out)
    return out


# ------------------------------------------------------------------ summary
SUMMARY_COLUMNS = ("cluster", "space", "type", "category", "conflict", "part",
                   "branch", "structure", "units", "energy", "representative",
                   "pairs", "support_sum", "support_mean",
                   "supported_pairs", "supported_fraction", "ambiguity_mean",
                   "ambiguity_max", "ambiguity_span",
                   "pair_distance_mean", "pair_distance_median",
                   "stem_length", "well_supported", "ct")


def write_summary(outdir, results, name="all_structures"):
    """
    Flat TSV table plus one merged BEDPE of every structure of a step.
    The merged file carries `category|conflict|cluster|structureN` in its name column.
    """
    from pathlib import Path

    outdir = Path(outdir)
    cols = [c for c in SUMMARY_COLUMNS if any(c in r for r in results)]
    with open(outdir / f"{name}.tsv", "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in results:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    merged = outdir / f"{name}.bedpe"
    with open(merged, "w") as f:
        f.write(BEDPE_HEADER.rstrip("\n") + "\tname\n")
        for r in results:
            bed = Path(r["ct"]).with_suffix(".bedpe")
            if not bed.exists():
                continue
            label = "|".join(str(r[k]) for k in ("category", "conflict", "cluster") if r.get(k))
            label = f"{label}|structure{r['structure']}"
            with open(bed) as b:
                for line in b:
                    if "chrom1" in line or not line.strip():
                        continue
                    f.write(line.rstrip("\n") + "\t" + label + "\n")
    return merged
