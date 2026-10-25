"""Step 7: global well-folded structure screening.
Identifies well-folded modules within the genome-wide representative structure produced by step 3.

1. Partition the representative structure into *structural units*.
    A unit is the interval enclosed by an outermost base pair

2. For each unit compute three attributes:
   * ``ambiguity_mean`` – mean pairing ambiguity PA(i,j) = H_i × H_j over the base
                          pairs in the unit, where H_i is the per-nucleotide Shannon
                          entropy of the raw 10 bp interaction profile restricted to
                          partners within 3000 nt
   * ``support_mean``   – mean interaction support RRI[i//10, j//10] over the base
                          pairs in the unit, where RRI is the O/E-normalised 10 bp
                          matrix masked to the 3000 nt diagonal band
   * ``length``         – number of nucleotides spanned by the unit.

3. Z-score-standardise all three attributes together .

4. Ward hierarchical clustering with Euclidean distance on the standardised
   features; the number of clusters is supplied by the caller.

5. A cluster is well-folded when **both** hold:
   * its median ``ambiguity_mean`` < background median ambiguity;
   * its median ``support_mean``   > background median support.

Outputs
* ``structural_units.tsv``          – one row per unit, attributes + cluster label
* ``cluster_stats.tsv``             – per-cluster medians and well-folded flag
* ``well_folded_units.bedpe``       – BEDPE of base pairs in well-folded units
* ``well_folded_units.tsv``         – rows of structural_units.tsv for wf units
* ``summary.json``                  – machine-readable version of the above
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from .clusters import build_partition_sparse
from .matrix_utils import BEDPE_HEADER, ensure_dir, get_oe, load_dump_matrix
from .metrics import shannon_track

log = logging.getLogger("river")

_SHORT_RANGE_MASK = 3000
_RRI_RES = 10


# ------------------------------------------------------------------ units
def _read_pairs_from_bedpe(bedpe_path):
    """0-based (a, b) pairs, a ≤ b."""
    pairs = []
    with open(bedpe_path) as f:
        for line in f:
            if "chrom1" in line or not line.strip():
                continue
            p = line.split()
            a, b = int(p[1]), int(p[4])
            if a <= b:
                pairs.append((a, b))
    return sorted(set(pairs))


def split_structure_units(pairs, length):
    """Partition a secondary structure into outermost-pair-bounded units.

    A unit is bounded by an outermost base pair: one that is not enclosed by
    any other pair in the structure.  Concretely, pair (a, b) is outermost when
    no other pair (c, d) satisfies c ≤ a and d ≥ b and (c, d) ≠ (a, b).  The
    unit spans [a, b] inclusive.

    Returns a list of (start, end) tuples (0-based, inclusive), sorted by start.
    """
    if not pairs:
        return []
    pair_set = set(pairs)

    def _enclosed(a, b):
        return any(c <= a and d >= b and (c, d) != (a, b) for c, d in pair_set)

    outermost = sorted((a, b) for a, b in pairs if not _enclosed(a, b))
    # merge overlapping outermost pairs into units
    units = []
    for a, b in outermost:
        if units and a <= units[-1][1]:
            units[-1] = (units[-1][0], max(units[-1][1], b))
        else:
            units.append([a, b])
    return [(s, e) for s, e in units]


# ------------------------------------------------------------------ attributes
def _unit_attributes(units, pairs, rri_matrix, shannon_array, rri_res=10):
    """
    Compute (ambiguity_mean, support_mean, length) for each unit.
    ambiguity_mean : mean of H_i × H_j over base pairs whose both ends are in the unit
    support_mean   : mean of rri_matrix[i//res, j//res] (O/E, 3000 nt band) over those pairs
    length         : e − s + 1
    """
    pair_set = set(pairs)
    rows = []
    for s, e in units:
        unit_pairs = [(a, b) for a, b in pair_set if s <= a <= e and s <= b <= e]
        if not unit_pairs:
            rows.append((0.0, 0.0, int(e - s + 1)))
            continue
        amb = float(np.mean([shannon_array[a] * shannon_array[b] for a, b in unit_pairs]))
        sup = float(np.mean([rri_matrix[a // rri_res, b // rri_res] for a, b in unit_pairs]))
        rows.append((amb, sup, int(e - s + 1)))
    return rows


# ----------------------------------------------------------------- background
def _band_mask(matrix, n):
    """Keep only diagonals 1 … n−1."""
    out = np.zeros_like(matrix)
    size = matrix.shape[0]
    for k in range(1, min(n, size)):
        i = np.arange(size - k)
        out[i, i + k] = matrix[i, i + k]
        out[i + k, i] = matrix[i + k, i]
    return out


def _background_medians(step1_list, step2_info, params, L):
    """
    Entropy track, support matrix and background medians.
    * Entropy  : raw 10 bp matrix, partners within 3000 nt.
    * RRI      : O/E of the raw 10 bp matrix, masked to the 3000 nt band.
    * Background pairs: (i, j) with 0 < |i − j| < 3000, partition P > 0 and
      RRI[i//10, j//10] > 0.
    * amb_bg = median of H_i × H_j, sup_bg = median of RRI over those pairs.
    """
    res = _RRI_RES
    mats = {int(k): v for k, v in step2_info["matrices"].items()}
    if res not in mats:
        raise RuntimeError(
            f"step 7 needs the {res} bp matrix from step 2 (found {sorted(mats)})")
    raw = load_dump_matrix(mats[res], res)         
    band = _SHORT_RANGE_MASK // res               

    shan_arr = shannon_track(raw, res, L, span_nt=_SHORT_RANGE_MASK)
    rri_m = _band_mask(get_oe(raw), band)

    # pad so that every nucleotide index i < L maps into the matrix
    nbin = -(-int(L) // res)
    if rri_m.shape[0] < nbin:
        pad = nbin - rri_m.shape[0]
        rri_m = np.pad(rri_m, ((0, pad), (0, pad)))

    # partition matrix, binary, short-range only
    part = build_partition_sparse(step1_list, Path(step1_list).parent / "tmp_pfs_txt", L)
    r = part.row.astype(np.int64)
    c = part.col.astype(np.int64)
    dist = np.abs(r - c)
    keep = (dist > 0) & (dist < _SHORT_RANGE_MASK)
    r, c = r[keep], c[keep]
    # require RRI signal at the same position
    sup_vals = rri_m[r // res, c // res]
    keep = sup_vals > 0
    r, c, sup_vals = r[keep], c[keep], sup_vals[keep]
    if len(r) == 0:
        log.warning("Step7: no background pairs with partition P > 0 and RRI > 0")
        return 0.0, 0.0, shan_arr, rri_m, res

    amb_bg = float(np.median(shan_arr[r] * shan_arr[c]))
    sup_bg = float(np.median(sup_vals))
    return amb_bg, sup_bg, shan_arr, rri_m, res


# ------------------------------------------------------------------ driver
def run_step7(params, step2_info, step3_outdir, outdir, n_clusters,
              step1_list=None, chrom="chr"):
    """Run global well-folded screening.

    Parameters
    ----------
    params        : RiverParams
    step2_info    : dict from step2_manifest.json
    step3_outdir  : directory of step 3 outputs (summary.json + structureN/merge_structure/)
    outdir        : output directory for step 7
    n_clusters    : number of Ward clusters (user-specified)
    step1_list    : path to partition_file_id.txt (step 1 output); if None,
                    inferred from step3_outdir/../step1_partition/partition_file_id.txt
    chrom         : chromosome name written to the BEDPE output (``-c/--chrom``)
    """
    outdir = ensure_dir(outdir)
    step3_outdir = Path(step3_outdir)

    # locate the representative whole-structure BEDPE from step3
    # (lowest total free energy, flagged in step3 summary.json; structure1 if absent)
    rep_n = 1
    summ = step3_outdir / "summary.json"
    if summ.exists():
        rep_n = next((r["structure"] for r in json.loads(summ.read_text())
                      if r.get("representative")), 1)
    rep_bedpe = step3_outdir / f"structure{rep_n}" / "merge_structure" / "whole_structure.bedpe"
    log.info("Step7: using step3 representative structure%d", rep_n)
    if not rep_bedpe.exists():
        raise FileNotFoundError(
            f"representative structure not found at {rep_bedpe}; run step 3 first")

    if step1_list is None:
        step1_list = step3_outdir.parent / "step1_partition" / "partition_file_id.txt"
    step1_list = Path(step1_list)
    if not step1_list.exists():
        raise FileNotFoundError(f"{step1_list} missing; run step 1 first")

    pairs = _read_pairs_from_bedpe(rep_bedpe)
    if not pairs:
        log.warning("Step7: no base pairs in representative structure — nothing to screen")
        return []
    L = max(b for _, b in pairs) + 1

    log.info("Step7: %d base pairs, partitioning into units …", len(pairs))
    units = split_structure_units(pairs, L)
    log.info("  %d structural units", len(units))

    # background medians + entropy + RRI matrix
    amb_bg, sup_bg, shan_arr, rri_m, rri_res = _background_medians(str(step1_list), step2_info, params, L)
    log.info("  background medians: ambiguity=%.4f, support=%.4f", amb_bg, sup_bg)

    attrs = _unit_attributes(units, pairs, rri_m, shan_arr, rri_res)
    if len(attrs) < 2:
        log.warning("Step7: fewer than 2 units — skipping clustering")
        return []

    data = np.array(attrs, dtype=float)   # (N, 3): ambiguity_mean, support_mean, length
    # z-score, same as sklearn StandardScaler (ddof=0; constant columns -> 0)
    std = data.std(axis=0)
    std[std == 0] = 1.0
    data_scaled = (data - data.mean(axis=0)) / std

    # Ward clustering
    n_clusters = max(1, min(n_clusters, len(units)))
    Z = linkage(data_scaled, method="ward", metric="euclidean")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")  # 1-based

    # cluster statistics
    unit_rows = []
    for i, ((s, e), (amb, sup, ln), lbl) in enumerate(zip(units, attrs, labels)):
        unit_rows.append({
            "unit": i + 1,
            "start": s, "end": e,
            "ambiguity_mean": round(amb, 6),
            "support_mean": round(sup, 6),
            "length": ln,
            "cluster": int(lbl),
        })

    cluster_ids = sorted(set(labels))
    cluster_stats = []
    for cid in cluster_ids:
        rows = [r for r in unit_rows if r["cluster"] == cid]
        med_amb = float(np.median([r["ambiguity_mean"] for r in rows]))
        med_sup = float(np.median([r["support_mean"] for r in rows]))
        wf = bool(med_amb < amb_bg and med_sup > sup_bg)
        cluster_stats.append({
            "cluster": int(cid),
            "n_units": len(rows),
            "ambiguity_median": round(med_amb, 6),
            "support_median": round(med_sup, 6),
            "well_folded": wf,
        })
        for r in rows:
            r["well_folded"] = wf
    n_wf = sum(1 for r in unit_rows if r.get("well_folded"))
    log.info("  %d/%d units in well-folded clusters", n_wf, len(unit_rows))

    # write outputs
    _write_tsv(outdir / "structural_units.tsv", unit_rows,
               ["unit", "start", "end", "ambiguity_mean", "support_mean", "length",
                "cluster", "well_folded"])
    _write_tsv(outdir / "cluster_stats.tsv", cluster_stats,
               ["cluster", "n_units", "ambiguity_median", "support_median", "well_folded"])

    wf_units = [r for r in unit_rows if r.get("well_folded")]
    _write_tsv(outdir / "well_folded_units.tsv", wf_units,
               ["unit", "start", "end", "ambiguity_mean", "support_mean", "length",
                "cluster", "well_folded"])

    # BEDPE of base pairs in well-folded units
    with open(outdir / "well_folded_units.bedpe", "w") as f:
        f.write(BEDPE_HEADER.rstrip("\n") + "\tunit\n")
        for r in unit_rows:
            if not r.get("well_folded"):
                continue
            s, e = r["start"], r["end"]
            for a, b in pairs:
                if s <= a <= e and s <= b <= e:
                    f.write(f"{chrom}\t{a}\t{a + 1}\t{chrom}\t{b}\t{b + 1}\t{r['unit']}\n")

    summary = {
        "n_units": len(unit_rows),
        "n_clusters": n_clusters,
        "n_well_folded_units": n_wf,
        "background_ambiguity_median": round(amb_bg, 6),
        "background_support_median": round(sup_bg, 6),
        "clusters": cluster_stats,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Step7: done -> %s", outdir / "well_folded_units.bedpe")
    return unit_rows


def _write_tsv(path, rows, cols):
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
