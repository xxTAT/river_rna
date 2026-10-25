"""Structure metrics: RIC-seq interaction support and pairing ambiguity.

Two numbers describe how trustworthy a predicted structure is:

``support``   how much RIC-seq signal its base pairs carry.  A helix that the
              data really sees has a high support; one that only the
              thermodynamic model likes has almost none.
``ambiguity`` how ambiguous the pairing of its nucleotides is.  The Shannon
              entropy of the interaction profile of a nucleotide is small when
              it interacts with one partner and large when its reads are spread
              over many partners; the ambiguity of a base pair is the product of
              the entropies of its two nucleotides.

Both are computed for every structure of step4 and step5, so the downstream
analysis no longer has to recompute them from the BEDPE files.
"""
from __future__ import annotations

import numpy as np

from .matrix_utils import BEDPE_HEADER

METRIC_COLUMNS = ("pairs", "support_sum", "support_mean", "supported_pairs",
                  "supported_fraction", "ambiguity_mean", "ambiguity_max",
                  "ambiguity_span", "pair_distance_mean", "pair_distance_median",
                  "stem_length")


def shannon_track(matrix, res, length, span_nt=0):
    """
    Per-nucleotide Shannon entropy of the interaction profile.
    `span_nt <= 0` uses the whole matrix; a positive value keeps only the partners closer than `span_nt` nucleotides.  
    The entropy of a bin is repeated `res` times to give a per-nucleotide track.
    """
    matrix = np.asarray(matrix, dtype=float)
    size = matrix.shape[0]
    if span_nt and span_nt > 0:
        n = max(int(span_nt // res), 2)
        band = np.zeros_like(matrix)
        for k in range(1, n):
            if k >= size:
                break
            i = np.arange(size - k)
            band[i, i + k] = matrix[i, i + k]
            band[i + k, i] = matrix[i + k, i]
    else:
        band = matrix
    total = band.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = band / total
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    safe = np.where(p > 0, p, 1.0)
    ent = -(np.where(p > 0, p * np.log2(safe), 0.0)).sum(axis=0)
    track = np.repeat(ent, res)
    if track.size < length:
        track = np.concatenate([track, np.zeros(length - track.size)])
    return track[:length]


def _amb(shannon, a, b):
    if shannon is None:
        return 0.0
    n = shannon.size
    if a >= n or b >= n:
        return 0.0
    return float(shannon[a] * shannon[b])


def structure_metrics(pairs, support, shannon, span=0):
    """Metrics of predicted structure.

    pairs   : absolute 0-based (a, b) base pairs
    support : callable (a, b) -> interaction value of that pair
    shannon : per-nucleotide entropy track (or None)
    span    : the entropy window that produced `shannon` (0 = whole molecule)
    """
    pairs = sorted((min(a, b), max(a, b)) for a, b in pairs)
    n = len(pairs)
    if not n:
        out = {c: 0 if c in ("pairs", "supported_pairs", "stem_length") else 0.0 for c in METRIC_COLUMNS}
        out["ambiguity_span"] = int(span)
        return out
    sup = np.array([float(support(a, b)) for a, b in pairs])
    amb = np.array([_amb(shannon, a, b) for a, b in pairs])
    dist = np.array([b - a + 1 for a, b in pairs], dtype=float)
    left = [a for a, _ in pairs]
    right = [b for _, b in pairs]
    supported = int((sup > 0).sum())
    return {
        "pairs": n,
        "support_sum": round(float(sup.sum()), 4),
        "support_mean": round(float(sup.mean()), 4),
        "supported_pairs": supported,
        "supported_fraction": round(supported / n, 4),
        "ambiguity_mean": round(float(amb.mean()), 4),
        "ambiguity_max": round(float(amb.max()), 4),
        "ambiguity_span": int(span),
        "pair_distance_mean": round(float(dist.mean()), 2),
        "pair_distance_median": round(float(np.median(dist)), 2),
        "stem_length": (max(left) - min(left) + 1) + (max(right) - min(right) + 1),
    }


def write_pair_metrics(path, pairs, support, shannon, chrom):
    """Per-base-pair table: support, entropies and ambiguity of every pair."""
    pairs = sorted((min(a, b), max(a, b)) for a, b in pairs)
    with open(path, "w") as f:
        f.write(BEDPE_HEADER.rstrip("\n")
                + "\tdistance\tsupport\tshannon1\tshannon2\tambiguity\n")
        for a, b in pairs:
            s1 = float(shannon[a]) if shannon is not None and a < shannon.size else 0.0
            s2 = float(shannon[b]) if shannon is not None and b < shannon.size else 0.0
            f.write(f"{chrom}\t{a}\t{a + 1}\t{chrom}\t{b}\t{b + 1}\t{b - a + 1}\t"
                    f"{float(support(a, b)):.4f}\t{s1:.4f}\t{s2:.4f}\t{s1 * s2:.4f}\n")


def flag_well_supported(results, support_stat="sum",
                        ambiguity_key="ambiguity_mean", group_key="space"):
    """Well-folded screening: support above and ambiguity below the mean of the
    structure's own structural space (short- or long-range).

    Only the representative structure of each feature takes part (the paper
    evaluates one structure per feature); `support_stat` chooses between the
    summed support of the structure (v1 notebooks) and its mean support per base
    pair (the wording of the manuscript).  Adds `well_supported` to every result
    and returns the thresholds of each space.
    """
    support_key = {"sum": "support_sum", "mean": "support_mean"}[support_stat]
    groups = {}
    for r in results:
        if not r.get("pairs") or r.get("representative") is False:
            continue
        groups.setdefault(r.get(group_key, r.get("ambiguity_span", 0)), []).append(r)
    cuts = {}
    for key, rows in groups.items():
        sup_cut = float(np.mean([r.get(support_key, 0.0) for r in rows]))
        amb_cut = float(np.mean([r.get(ambiguity_key, 0.0) for r in rows]))
        cuts[str(key)] = {"structures": len(rows), "support_stat": support_key,
                          "support_mean_cut": round(sup_cut, 4),
                          "ambiguity_mean_cut": round(amb_cut, 4)}
        for r in rows:
            r["well_supported"] = bool(r.get(support_key, 0.0) > sup_cut
                                       and r.get(ambiguity_key, 0.0) < amb_cut)
    for r in results:
        r.setdefault("well_supported", False)
    return cuts
