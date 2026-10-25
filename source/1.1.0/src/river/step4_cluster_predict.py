"""Step 4: predict structure for each structural feature.
Every feature of step 2 is folded on its own:

* long range, span > `cluster_max_distance`-> `bipartition` + `RNAcofold`
* short range, off-diagonal                -> `partition`   + `RNAcofold`
* short range, diagonal                    -> `partition`   + `Fold -c`
"""
from __future__ import annotations

import json
import logging

import numpy as np

from .cluster_fold import (predict_diagonal_cluster, predict_long_range_cluster, predict_nondiagonal_cluster)
from .clusters import (FeaturePixels, SupportMatrices, classify_clusters, cluster_name, load_clusters, write_clusters_bedpe, write_summary)
from .matrix_utils import BEDPE_HEADER, ensure_dir
from .metrics import shannon_track

log = logging.getLogger("river")


def run_step4(params, tool, seq, chrom, step1_list, step2_info, outdir, resume=False):
    outdir = ensure_dir(outdir)
    L = len(seq)
    if not step2_info.get("clusters"):
        raise RuntimeError("step 4 needs the structure-feature clusters of step 2 "
                           "(bedtools is required; re-run step 2)")
    clusters = load_clusters(step2_info["clusters"])
    long_range, nondiag, diag = classify_clusters(clusters, params)
    log.info("Step4: %d cluster(s): %d long-range, %d short-range off-diagonal, "
             "%d short-range diagonal", len(clusters), len(long_range),
             len(nondiag), len(diag))
    write_clusters_bedpe(outdir / "clusters_long_range.bedpe", long_range, chrom)
    write_clusters_bedpe(outdir / "clusters_short_range_nondiagonal.bedpe", nondiag, chrom)
    write_clusters_bedpe(outdir / "clusters_short_range_diagonal.bedpe", diag, chrom)

    mats = {int(k): v for k, v in step2_info["matrices"].items()}
    support = SupportMatrices(mats, params.score_res, L)

    # pairing ambiguity: Shannon entropy product for each pair:
        # long-range features are always scored against the whole rna
        # short-range ones against the partners within metric_shannon_span
        # (-1 = cluster_max_distance, 0 = whole rna)
    spans = _metric_spans(params)
    dense_coarse = support.dense(support.coarse)
    tracks = {s: shannon_track(dense_coarse, support.coarse, L, s) for s in sorted(set(spans.values()))}
    _write_shannon(outdir / "shannon.txt", tracks, L)
    log.info("pairing ambiguity: entropy window %s nt (long range), %s nt "
             "(short range); 0 = whole rna", spans["long"], spans["short"])
    results = []

    if long_range:
        ldir = ensure_dir(outdir / "long_range")
        for k, c in enumerate(long_range, 1):
            log.info("  [%d/%d] long-range cluster %s", k, len(long_range), cluster_name(c))
            results += _space(predict_long_range_cluster(
                tool, params, seq, chrom, c, support, ldir / f"cluster_{cluster_name(c)}", resume,
                shannon=tracks[spans["long"]], shannon_span=spans["long"]), "long")

    if nondiag or diag:
        feature, _ = FeaturePixels.build(params, step1_list,
                                         step2_info["feature_bedpe"],
                                         mats[params.mid_res], L,
                                         max_pair_distance=None)
        feature.write_bedpe(outdir / "structure_feature.bedpe", chrom)
        sdir = ensure_dir(outdir / "short_range")
        for k, c in enumerate(nondiag, 1):
            log.info("  [%d/%d] off-diagonal cluster %s", k, len(nondiag), cluster_name(c))
            results += _space(predict_nondiagonal_cluster(
                tool, params, seq, chrom, c, feature, support,
                sdir / f"cluster_{cluster_name(c)}", resume,
                shannon=tracks[spans["short"]], shannon_span=spans["short"]), "short")
        for k, c in enumerate(diag, 1):
            log.info("  [%d/%d] diagonal cluster %s", k, len(diag), cluster_name(c))
            results += _space(predict_diagonal_cluster(
                tool, params, seq, chrom, c, feature, support,
                sdir / f"cluster_{cluster_name(c)}", resume,
                shannon=tracks[spans["short"]], shannon_span=spans["short"]), "short")

    (outdir / "summary.json").write_text(json.dumps(results, indent=2))
    write_summary(outdir, results, "all_cluster_structures")
    log.info("Step4: %d structure(s) for %d cluster(s) -> %s",
             len(results), len({r["cluster"] for r in results}),
             outdir / "all_cluster_structures.bedpe")
    return results


def _space(results, space):
    for r in results:
        r["space"] = space
    return results


def _metric_spans(params):
    """Entropy window used for the long-range and the short-range branch.

    The long-range window is always the whole rna (0); `metric_shannon_span`
    only sets the short-range window (< 0 or None = cluster_max_distance).
    """
    span = params.metric_shannon_span
    if span is None or span < 0:
        short = int(params.cluster_max_distance)
    else:
        short = int(span)
    return {"long": 0, "short": short}


def _write_shannon(path, tracks, length):
    cols = sorted(tracks)
    data = np.column_stack([np.arange(length)] + [tracks[s] for s in cols])
    header = "pos\t" + "\t".join(f"shannon_span{s}" for s in cols)
    np.savetxt(path, data, fmt=["%d"] + ["%.6f"] * len(cols), delimiter="\t", header=header, comments="")

