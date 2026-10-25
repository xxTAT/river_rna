"""
Structure prediction inside one structure feature (cluster)
    ``partition``  ->short-range cluster
    ``bipartition``->long-range cluster

    ``cofold``     ->short-range non-diagonal cluster
                     long-range cluster
    ``fold``       ->diagonal cluster
                     optionally forbidding a conflicting rectangle

Every feature produces one sub-directory containing:
    ``structureN.ct/.bedpe/.dbn``
    ``structureN_annotation.txt`` (support per nucleotide)
      the constraint files that were used
"""
from __future__ import annotations

import logging

import numpy as np

from .clusters import clipped_block, cluster_name, region_of, score_units
from .folding import (best_structure, cofold_constraint, cofold_energy, forbidden_pairs,
                      pairs_from_dots, parse_cofold, parse_ct, read_cofold_fasta,
                      relax_constraint, write_cofold_fasta, write_constraint,
                      write_structure_outputs)
from .matrix_utils import BEDPE_HEADER, ensure_dir, write_fasta
from .metrics import structure_metrics, write_pair_metrics
from .structure_tree import build_structure_paths, chain_units, path_pairs

log = logging.getLogger("river")


# ------------------------------------------------------------------ helpers
def mark_representative(results):
    """
    Flag the representative structure of a feature: the lowest free energy.
    `energy` is recorded for every candidate during folding.
    Candidates without an energy rank last; ties keep the earlier candidate.
    """
    if not results:
        return results
    best = min(range(len(results)),
               key=lambda i: (results[i].get("energy") is None,
                              results[i].get("energy") or 0.0, i))
    for i, r in enumerate(results):
        r["representative"] = i == best
    return results

def _units_local(pixels, to_local):
    """Chain pixels into stems and translate them into local coordinates."""
    unit_abs = chain_units(pixels)
    unit_loc = {u: [to_local(a, b) for a, b in locs] for u, locs in unit_abs.items()}
    return unit_abs, unit_loc


def _write_units(path, unit_abs, chrom):
    with open(path, "w") as f:
        f.write(BEDPE_HEADER)
        for locs in unit_abs.values():
            for a, b in locs:
                f.write(f"{chrom}\t{a}\t{a + 1}\t{chrom}\t{b}\t{b + 1}\n")


def _candidate_structures(unit_abs, unit_loc, support_block, prob, params,
                          origin, n_local):
    """Score the stems of a cluster and enumerate the candidate structures."""
    if not unit_loc:
        return [], []
    block = clipped_block(support_block, params.score_clip_percentile)
    r0, c0 = origin

    def _support(a, b):
        i, j = a - r0, b - c0
        if 0 <= i < block.shape[0] and 0 <= j < block.shape[1]:
            return block[i, j]
        return 0.0

    scores_abs = score_units(unit_abs, _support, prob, params)
    if not scores_abs:
        return [], []
    paths = build_structure_paths(scores_abs, unit_loc, n_local)
    return paths, scores_abs


# ------------------------------------------------- branch 1: RNAcofold pairs
def fold_cluster_cofold(tool, params, seq, chrom, cluster, unit_abs, unit_loc,
                        support_block, prob, support, outdir, resume=False,
                        n_local=None, shannon=None, shannon_span=0):
    """Fold the two arms of a non-diagonal cluster against each other."""
    s1, e1, s2, e2 = (int(x) for x in cluster)
    left_len, right_len = e1 - s1, e2 - s2
    n_local = n_local or (left_len + right_len)
    outdir = ensure_dir(outdir)
    _write_units(outdir / "units.bedpe", unit_abs, chrom)
    paths, scores = _candidate_structures(unit_abs, unit_loc, support_block, prob,
                                          params, (s1, s2), n_local)
    if not paths:
        log.info("    %s: no structural unit", cluster_name(cluster))
        return []
    lpad = min(params.cofold_pad, len(seq) - e1)
    rpad = min(params.cofold_pad, s2)
    left_seq, right_seq = seq[s1:e1 + lpad], seq[s2 - rpad:e2]
    annotate = support.annotator()
    results = []
    for n, path in enumerate(paths, 1):
        wdir = ensure_dir(outdir / f"structure{n}")
        prefix = wdir / f"structure{n}"
        pairs = [(a, b - left_len) for a, b in path_pairs(path, unit_loc)]
        fasta = wdir / f"structure{n}.fasta"
        cons = cofold_constraint(pairs, left_len, right_len, lpad, rpad)
        write_cofold_fasta(fasta, cluster_name(cluster), left_seq, right_seq, cons)
        out = wdir / f"structure{n}.cofold"
        if not (resume and out.exists() and out.stat().st_size > 0):
            _run_cofold(tool, params, fasta, out, wdir)
        dots = parse_cofold(out.read_text())
        if dots is None:
            log.warning("    %s structure%d: RNAcofold gave no structure",
                        cluster_name(cluster), n)
            continue
        ld, rd = dots
        abs_pairs = [(s1 + a, s2 + b) for a, b in pairs_from_dots(ld, rd, lpad, rpad)]
        kept = write_structure_outputs(str(prefix), seq, [(s1, e1), (s2, e2)], abs_pairs, chrom, annotate)
        (wdir / "units.txt").write_text("\n".join(path) + "\n")
        write_pair_metrics(f"{prefix}_pairs.tsv", kept, annotate, shannon, chrom)
        res = {"cluster": cluster_name(cluster), "branch": "cofold",
               "structure": n, "units": len(path) - 1,
               "energy": cofold_energy(out.read_text()), "ct": f"{prefix}.ct"}
        res.update(structure_metrics(kept, annotate, shannon, shannon_span))
        results.append(res)
    return mark_representative(results)


def _run_cofold(tool, params, fasta, out, cwd):
    """`RNAcofold --enforceConstraint`, relaxing the constraint on failure."""
    tool.cofold(fasta, out, enforce=True, cwd=cwd)
    tries = 0
    while out.stat().st_size == 0 and tries < params.cofold_max_retry:
        tries += 1
        name, left, right, cons = read_cofold_fasta(fasta)
        ld, rd = cons.split("&")
        ld, rd = relax_constraint(ld, rd)
        write_cofold_fasta(fasta, name, left, right, ld + "&" + rd)
        log.info("    relaxed constraint (%d) and re-ran RNAcofold", tries)
        tool.cofold(fasta, out, enforce=True, cwd=cwd)
    if out.stat().st_size == 0:
        log.warning("    constraint still unsatisfiable; folding without enforcement")
        tool.cofold(fasta, out, enforce=False, cwd=cwd)


# --------------------------------------------- branch 2: RNAstructure Fold
def fold_cluster_region(tool, params, seq, chrom, cluster, region, unit_abs,
                        unit_loc, support_block, prob, support, outdir,
                        resume=False, forbid_rect=None, origin=None, shannon=None,
                        shannon_span=0):
    """Fold a whole (diagonal) region"""
    start, end = int(region[0]), int(region[1])
    outdir = ensure_dir(outdir)
    _write_units(outdir / "units.bedpe", unit_abs, chrom)
    paths, scores = _candidate_structures(unit_abs, unit_loc, support_block, prob,
                                          params, origin or (start, start), end - start)
    annotate = support.annotator()
    forbids = forbidden_pairs(start, forbid_rect) if forbid_rect else ()
    if not paths:
        log.info("    %s: no structural unit, folding without pair constraints",
                 cluster_name(cluster))
        paths = [("s0",)]
    results = []
    for n, path in enumerate(paths, 1):
        wdir = ensure_dir(outdir / f"structure{n}")
        prefix = wdir / f"structure{n}"
        fasta = wdir / f"structure{n}.fasta"
        const = wdir / f"structure{n}.const"
        ct = wdir / f"structure{n}_fold.ct"
        write_fasta(fasta, f"{start}_{end}", seq[start:end])
        pairs = [(a + 1, b + 1) for a, b in path_pairs(path, unit_loc)]
        write_constraint(const, pairs, forbids)
        if not (resume and ct.exists() and ct.stat().st_size > 0):
            tool.fold(fasta, ct, const if (pairs or forbids) else None)
        structs, energy = parse_ct(ct, start)
        abs_pairs = [(a - 1, b - 1) for a, b in best_structure(structs, energy)]
        mfe = min(energy.values()) if energy else None
        if mfe is not None and mfe >= 10000:        # no ENERGY line in the CT
            mfe = None
        kept = write_structure_outputs(str(prefix), seq, [(start, end)], abs_pairs, chrom, annotate, restrict=cluster)
        (wdir / "units.txt").write_text("\n".join(path) + "\n")
        write_pair_metrics(f"{prefix}_pairs.tsv", kept, annotate, shannon, chrom)
        res = {"cluster": cluster_name(cluster), "branch": "fold",
               "structure": n, "units": len(path) - 1, "energy": mfe,
               "ct": f"{prefix}.ct"}
        res.update(structure_metrics(kept, annotate, shannon, shannon_span))
        results.append(res)
    return mark_representative(results)


# ------------------------------------------------------------ cluster driver
def predict_nondiagonal_cluster(tool, params, seq, chrom, cluster, feature,
                                support, outdir, resume=False, shannon=None,
                                shannon_span=0):
    """Short-range non-diagonal cluster: candidate pixels -> units -> RNAcofold."""
    s1, e1, s2, e2 = (int(x) for x in cluster)
    left_len = e1 - s1
    pixels = feature.in_rect(s1, e1, s2, e2)
    unit_abs, unit_loc = _units_local(pixels, lambda a, b: (a - s1, left_len + b - s2))
    block = support.block(s1, e1, s2, e2)
    return fold_cluster_cofold(tool, params, seq, chrom, cluster, unit_abs, unit_loc,
                               block, lambda a, b: feature.value.get((a, b), 0.0),
                               support, outdir, resume, shannon=shannon,
                               shannon_span=shannon_span)


def predict_diagonal_cluster(tool, params, seq, chrom, cluster, feature, support,
                             outdir, resume=False, pixels=None, region=None,
                             forbid_rect=None, shannon=None, shannon_span=0):
    """Diagonal cluster (or a group of them): units -> `Fold -c` on the region."""
    s1, e1, s2, e2 = (int(x) for x in cluster)
    start, end = region or region_of(cluster)
    if pixels is None:
        pixels = feature.in_rect(s1, e1, s2, e2)
    unit_abs, unit_loc = _units_local(pixels, lambda a, b: (a - start, b - start))
    block = support.block(start, end, start, end)
    return fold_cluster_region(tool, params, seq, chrom, cluster, (start, end),
                               unit_abs, unit_loc, block,
                               lambda a, b: feature.value.get((a, b), 0.0), support,
                               outdir, resume, forbid_rect, origin=(start, start),
                               shannon=shannon, shannon_span=shannon_span)


def predict_long_range_cluster(tool, params, seq, chrom, cluster, support, outdir,
                               resume=False, shannon=None, shannon_span=0):
    """Long-range cluster: `bipartition` probabilities -> units -> RNAcofold."""
    s1, e1, s2, e2 = (int(x) for x in cluster)
    left_len, right_len = e1 - s1, e2 - s2
    outdir = ensure_dir(outdir)
    bdir = ensure_dir(outdir / "bipartition")
    name = cluster_name(cluster)
    fa1, fa2 = bdir / f"{s1}_{e1}.fasta", bdir / f"{s2}_{e2}.fasta"
    pfs, txt = bdir / f"{name}.pfs", bdir / f"{name}.txt"
    if not (resume and txt.exists() and txt.stat().st_size > 0):
        write_fasta(fa1, f"{s1}_{e1}", seq[s1:e1])
        write_fasta(fa2, f"{s2}_{e2}", seq[s2:e2])
        log.info("    bipartition %s (%d x %d nt)", name, left_len, right_len)
        tool.bipartition(fa1, fa2, pfs)
        tool.probability_plot(pfs, txt)
    prob_matrix = read_bipartition(txt, left_len, right_len)
    rows, cols = np.nonzero(prob_matrix > 0)
    pixels = [(int(a) + s1, int(b) + s2) for a, b in zip(rows, cols)]
    unit_abs, unit_loc = _units_local(pixels, lambda a, b: (a - s1, left_len + b - s2))

    def _prob(a, b):
        return prob_matrix[a - s1, b - s2]

    block = support.block(s1, e1, s2, e2)
    return fold_cluster_cofold(tool, params, seq, chrom, cluster, unit_abs, unit_loc,
                               block, _prob, support, outdir, resume, shannon=shannon,
                               shannon_span=shannon_span)


def read_bipartition(txt, left_len, right_len):
    """`ProbabilityPlot -t` output of bipartition -> (left_len, right_len) matrix.

    bipartition concatenates the two fragments with three linker nucleotides,
    so the second fragment starts at index left_len + 4 (1-based).
    """
    m = np.zeros((left_len, right_len))
    lines = [l for i, l in enumerate(open(txt)) if i >= 2 and l.strip()]
    if not lines:
        return m
    d = np.loadtxt(lines, ndmin=2)
    if d.size == 0:
        return m
    r = d[:, 0].astype(np.int64) - 1
    c = d[:, 1].astype(np.int64) - 4 - left_len
    ok = (r >= 0) & (r < left_len) & (c >= 0) & (c < right_len) & (d[:, 2] > 0)
    m[r[ok], c[ok]] = d[ok, 2]
    return m
