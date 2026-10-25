"""Step 5: alternative structures.

Regions whose interactions are spread over many partners (high Shannon entropy)
can fold in more than one way. Structural features that compete for the same
nucleotides are collected, classified and each of them is folded separately

|---------------Type------------------|----------Lable-----------|--------------------------Description------------------------------|
|Type I   (crossing mutual exclusion) | cross                    |  the shared region lies on opposite sides of the two features     |
|Type II  (same-side competition)     | vertical_horizontal      |  the shared region lies on the same side of both features         |
|Type III (nested inclusion)          | diagonal                 |  one feature lies entirely within the formation region of another |
|Type IV  (local-distal competition)  | diagonal_inside_and_out  |  a set of local features competes with a distal feature           |
"""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np

from .cluster_fold import predict_diagonal_cluster, predict_nondiagonal_cluster
from .clusters import (FeaturePixels, SupportMatrices, cluster_name, is_diagonal,
                       load_clusters, overlaps, span, write_clusters_bedpe,
                       write_summary)
from .matrix_utils import ensure_dir
from .metrics import flag_well_supported, shannon_track

log = logging.getLogger("river")

CATEGORIES = ("diagonal", "cross", "vertical_horizontal", "diagonal_inside_and_out")

TOPOLOGY = {"cross": "I", "vertical_horizontal": "II", "diagonal": "III", "diagonal_inside_and_out": "IV"}


# ------------------------------------------------------------------ entropy
def local_shannon(matrix, span_nt, res, length):
    """Shannon entropy of the interaction profile of every bin"""
    return shannon_track(matrix, res, length, span_nt)


def select_high_shannon(clusters, shannon, max_distance):
    """Clusters whose right arm sits in an above-average entropy region."""
    average = float(shannon.mean())
    out = []
    for c in clusters:
        if span(c) > max_distance:
            continue
        s2, e2 = int(c[2]), min(int(c[3]) + 1, shannon.size)
        if s2 < shannon.size and (shannon[s2:e2] >= average).any():
            out.append(c)
    return out, average


def score_clusters(clusters, support, res, percentile):
    """Rank off-diagonal clusters by their total interaction signal (top 50 % default)."""
    values = [support.block_sum(res, c[0], c[1], c[2], c[3]) for c in clusters]
    if not values:
        return [], []
    cut = np.percentile(values, percentile)
    keep = [c for c, v in zip(clusters, values) if v >= cut]
    drop = [c for c, v in zip(clusters, values) if v < cut]
    return keep, drop


# ------------------------------------------------------------- conflict sets
def _arm_overlap(c, slop):
    s1, e1, s2, e2 = (int(x) for x in c)
    return (s1 - slop, e1 + 2 * slop), (s2 - slop, e2 + 2 * slop)


def classify_conflicts(high_clusters, distant_ok, params):
    """Split the candidates and collect all conflicting pairs per category."""
    slop = params.alt_conflict_slop
    frac = params.alt_conflict_overlap_fraction
    diagonal_cand, distant_cand = [], []
    for c in high_clusters:
        left, right = _arm_overlap(c, slop)
        if max(left[0], right[0]) >= min(left[1], right[1]):     # arms disjoint
            if tuple(c) in distant_ok:
                distant_cand.append(tuple(c))
        else:
            diagonal_cand.append(tuple(c))
    conflicts = {k: {} for k in CATEGORIES}

    def _add(cat, a, b):
        conflicts[cat].setdefault(a, set()).add(b)
        conflicts[cat].setdefault(b, set()).add(a)

    for a, b in itertools.combinations(sorted(set(diagonal_cand)), 2):
        if (overlaps(a[0], a[1], b[0], b[1], frac) or overlaps(a[2], a[3], b[2], b[3], frac)):
            _add("diagonal", a, b)
    for a, b in itertools.combinations(sorted(set(distant_cand)), 2):
        if (overlaps(a[0], a[1], b[0], b[1], frac) or overlaps(a[2], a[3], b[2], b[3], frac)):
            _add("vertical_horizontal", a, b)
        if (overlaps(a[2], a[3], b[0], b[1], frac) or overlaps(a[0], a[1], b[2], b[3], frac)):
            _add("cross", a, b)
    conflicts["diagonal_inside_and_out"] = {c: set() for c in sorted(set(distant_cand))}
    return conflicts


def filter_by_feature(conflict, canny):
    """Drop clusters without any Canny feature pixe"""
    alive = {c: set(p) for c, p in conflict.items() if canny.has_support(c)}
    return {c: sorted(p & set(alive)) for c, p in alive.items()}


def conflict_pairs(conflict):
    """Unordered, deterministic list of conflicting cluster pairs."""
    seen, out = set(), []
    for c in sorted(conflict):
        for p in sorted(conflict[c]):
            key = tuple(sorted((c, tuple(p))))
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
    return out


def diagonal_partners(cluster, diagonal_clusters, slop):
    """Diagonal clusters that overlap the region of a off-diagonal cluster."""
    r0 = min(int(cluster[0]), int(cluster[2]))
    r1 = max(int(cluster[1]), int(cluster[3]))
    out = []
    for d in diagonal_clusters:
        d0 = min(int(d[0]), int(d[2]))
        d1 = max(int(d[1]), int(d[3]))
        if d0 < r1 + slop and r0 < d1 + slop:
            out.append(tuple(d))
            r0, r1 = min(r0, d0), max(r1, d1)
    return out, (r0, r1)


# --------------------------------------------------------------- comparison
SCORE_COLUMNS = ("conflict", "type", "category", "cluster", "part", "structure", "units",
                 "pairs", "support_sum", "support_mean", "supported_pairs",
                 "supported_fraction", "ambiguity_mean", "ambiguity_span", "energy",
                 "representative", "support_share",
                 "rank", "best", "ct")


def read_pairs_bedpe(path):
    pairs = []
    with open(path) as f:
        for line in f:
            if "chrom1" in line or not line.strip():
                continue
            sub = line.split()
            pairs.append((int(sub[1]), int(sub[4])))
    return pairs


def score_alternative_structures(results, support):
    """interaction support of every candidate, compared inside its own conflict.

    For each predicted structure the interaction support of its base pairs is
    summed over the `score_res` matrices.  Candidates of one conflict are then
    ranked by the mean support per base pair, which does not depend on how many
    pairs a conformation has - a off-diagonal duplex of 12 pairs and the local
    hairpins that occupy the same nucleotides stay comparable.
    """
    for r in results:
        if "support_sum" not in r:      # metrics not computed by the folder
            bed = Path(r["ct"]).with_suffix(".bedpe")
            pairs = read_pairs_bedpe(bed) if bed.exists() else []
            vals = [support.pair_value(a, b) for a, b in pairs]
            n = len(vals)
            supported = sum(1 for v in vals if v > 0)
            r["pairs"] = n
            r["support_sum"] = round(float(sum(vals)), 4)
            r["support_mean"] = round(float(sum(vals) / n), 4) if n else 0.0
            r["supported_pairs"] = supported
            r["supported_fraction"] = round(supported / n, 4) if n else 0.0
    groups = {}
    for r in results:
        groups.setdefault(r.get("conflict", "none"), []).append(r)
    for rows in groups.values():
        total = sum(x["support_sum"] for x in rows)
        for x in rows:
            x["support_share"] = round(x["support_sum"] / total, 4) if total else 0.0
        order = sorted(rows, key=lambda x: (-x["support_mean"], -x["pairs"], x["cluster"], x["structure"]))
        for i, x in enumerate(order, 1):
            x["rank"] = i
            x["best"] = i == 1
    return results


def write_score_table(outdir, results, name="conflict_scores"):
    """One global table plus a copy inside every conflict directory."""
    outdir = Path(outdir)
    rows = sorted(results, key=lambda r: (r.get("category", ""), r.get("conflict", ""), r.get("rank", 0)))

    def _dump(path, subset):
        with open(path, "w") as f:
            f.write("\t".join(SCORE_COLUMNS) + "\n")
            for r in subset:
                f.write("\t".join(str(r.get(c, "")) for c in SCORE_COLUMNS) + "\n")

    _dump(outdir / f"{name}.tsv", rows)
    per_conflict = {}
    for r in rows:
        per_conflict.setdefault((r.get("category"), r.get("conflict")), []).append(r)
    for (cat, conf), subset in per_conflict.items():
        if not cat or not conf:
            continue
        d = outdir / cat / f"conflict_{conf.rsplit('_', 1)[-1]}"
        if d.is_dir():
            _dump(d / f"{name}.tsv", subset)
    return outdir / f"{name}.tsv"


# ------------------------------------------------------------------- driver
def run_step5(params, tool, seq, chrom, step1_list, step2_info, outdir, resume=False):
    outdir = ensure_dir(outdir)
    L = len(seq)
    if not step2_info.get("clusters"):
        raise RuntimeError("step 5 needs the structure-feature clusters of step 2")
    clusters = load_clusters(step2_info["clusters"])
    mats = {int(k): v for k, v in step2_info["matrices"].items()}
    support = SupportMatrices(mats, params.score_res, L)
    coarse = support.coarse

    dense_coarse = support.dense(coarse)
    shannon = local_shannon(dense_coarse, params.alt_shannon_span, coarse, L)
    # step5 only ever folds short-range features
    metric_span = (int(params.cluster_max_distance)
                   if params.metric_shannon_span is None
                   or params.metric_shannon_span < 0
                   else int(params.metric_shannon_span))
    metric_shannon = (shannon if metric_span == params.alt_shannon_span else shannon_track(dense_coarse, coarse, L, metric_span))
    np.savetxt(outdir / "shannon.txt", np.column_stack([np.arange(L), shannon]), fmt="%d\t%.6f")
    high, average = select_high_shannon(clusters, shannon, params.cluster_max_distance)
    log.info("Step5: mean Shannon entropy %.3f, %d of %d cluster(s) in high-entropy "
             "regions", average, len(high), len(clusters))
    write_clusters_bedpe(outdir / "clusters_high_shannon.bedpe", high, chrom)

    short = [c for c in clusters if span(c) <= params.cluster_max_distance]
    diag = [c for c in short if is_diagonal(c, params.cluster_diagonal_gap)]
    nondiag = [c for c in short if not is_diagonal(c, params.cluster_diagonal_gap)]
    keep, drop = score_clusters(nondiag, support, coarse, params.alt_cluster_score_percentile)
    write_clusters_bedpe(outdir / "clusters_nondiagonal_kept.bedpe", keep, chrom)
    write_clusters_bedpe(outdir / "clusters_nondiagonal_dropped.bedpe", drop, chrom)

    conflicts = classify_conflicts(high, {tuple(c) for c in keep}, params)

    feature, canny = FeaturePixels.build(params, step1_list, step2_info["feature_bedpe"], mats[coarse], L, max_pair_distance=None)
    feature.write_bedpe(outdir / "structure_feature.bedpe", chrom)
    conflicts = {k: filter_by_feature(v, canny) for k, v in conflicts.items()}
    for cat in CATEGORIES:
        write_clusters_bedpe(outdir / f"conflict_{cat}.bedpe", sorted(conflicts[cat]), chrom)
        log.info("  %s: %d cluster(s)", cat, len(conflicts[cat]))

    results = []
    for cat in ("diagonal", "cross", "vertical_horizontal"):
        pairs = conflict_pairs(conflicts[cat])
        cdir = ensure_dir(outdir / cat)
        for k, pair in enumerate(pairs, 1):
            log.info("  %s conflict %d/%d: %s vs %s", cat, k, len(pairs), cluster_name(pair[0]), cluster_name(pair[1]))
            pdir = ensure_dir(cdir / f"conflict_{k}")
            for c in pair:
                wdir = pdir / f"cluster_{cluster_name(c)}"
                if cat == "diagonal":
                    res = predict_diagonal_cluster(tool, params, seq, chrom, c,
                                                   feature, support, wdir, resume,
                                                   shannon=metric_shannon,
                                                   shannon_span=metric_span)
                else:
                    res = predict_nondiagonal_cluster(tool, params, seq, chrom, c,
                                                      feature, support, wdir, resume,
                                                      shannon=metric_shannon,
                                                   shannon_span=metric_span)
                for r in res:
                    r["conflict"] = f"{cat}_{k}"
                    r["category"] = cat
                    r["type"] = TOPOLOGY[cat]
                    r["space"] = "short"
                results += res

    cat = "diagonal_inside_and_out"
    cdir = ensure_dir(outdir / cat)
    for k, c in enumerate(sorted(conflicts[cat]), 1):
        log.info("  %s conflict %d/%d: %s", cat, k, len(conflicts[cat]), cluster_name(c))
        pdir = ensure_dir(cdir / f"conflict_{k}")
        res = predict_nondiagonal_cluster(tool, params, seq, chrom, c, feature, support,
                                          pdir / f"nondiagonal_{cluster_name(c)}", resume,
                                          shannon=metric_shannon,
                                          shannon_span=metric_span)
        for r in res:
            r.update(conflict=f"{cat}_{k}", category=cat, type=TOPOLOGY[cat], space="short", part="off-diagonal")
        results += res
        partners, region = diagonal_partners(c, diag, params.alt_conflict_slop)
        pixels = feature.in_rects(partners) if partners else []
        region_cluster = (region[0], region[1], region[0], region[1])
        res = predict_diagonal_cluster(tool, params, seq, chrom, region_cluster,
                                       feature, support,
                                       pdir / f"diagonal_{region[0]}_{region[1]}",
                                       resume, pixels=pixels, region=region,
                                       forbid_rect=c, shannon=metric_shannon,
                                       shannon_span=metric_span)
        for r in res:
            r.update(conflict=f"{cat}_{k}", category=cat, type=TOPOLOGY[cat], space="short", part="diagonal")
        results += res

    score_alternative_structures(results, support)
    table = write_score_table(outdir, results)
    for conf in sorted({r["conflict"] for r in results}):
        rows = [r for r in results if r["conflict"] == conf]
        best = min(rows, key=lambda r: r["rank"])
        log.info("  %s: %d candidate(s), best %s structure%d "
                 "(%.3f support/pair, %.0f%% of the conflict)", conf, len(rows),
                 best["cluster"], best["structure"], best["support_mean"],
                 100 * best["support_share"])
    (outdir / "summary.json").write_text(json.dumps(results, indent=2, default=str))
    write_summary(outdir, results, "all_alternative_structures")
    log.info("Step5: comparison table -> %s", table)
    log.info("Step5: %d alternative structure(s) -> %s", len(results), outdir / "all_alternative_structures.bedpe")
    return results
