"""Step 6: local well-folded structure screening.

Evaluates the representative structure of every structural feature from step 4.
A structure is classified as well-folded when its interaction support (sum of 
signal across its base pairs) exceeds the mean of its structural space,
and its mean pairing ambiguity is below that mean.

Short-range and long-range structures are screened separately.
Both results are written in a single run.

Shannon entropy
---------------
* Long-range features  : whole-molecule entropy (span_nt = 0).
* Short-range features : entropy within ``metric_shannon_span`` nt
  (-1 = ``cluster_max_distance``, default 3000 nt), the same windows as step 4.

All interaction-derived values use the 10 bp matrix, as in the v1 notebooks
(``extract_matrix(..._10_KR.matrix, 10)``).

Metric source
-------------
``source="recompute"`` (default) re-reads the base pairs of every
representative structure and recomputes ``support_sum`` / ``ambiguity_mean``
from the 10 bp matrix of step 2.
``source="step4"`` reuses the values already stored in the step 4
``summary.json`` without touching any matrix; rows whose ``ambiguity_span``
differs from the expected window are reported.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .clusters import load_dump_sparse
from .matrix_utils import BEDPE_HEADER, ensure_dir
from .metrics import shannon_track

log = logging.getLogger("river")

# Column name used for the well-folded support threshold (C1: sum, not mean).
_SUPPORT_KEY = "support_sum"
_AMBIGUITY_KEY = "ambiguity_mean"
_METRIC_RES = 10          # bp; the v1 notebooks use the 10 bp matrix throughout
SOURCES = ("recompute", "step4")


# ------------------------------------------------------------------ helpers
def _load_representatives(summary_path):
    """Return only the representative structure of each feature."""
    results = json.loads(Path(summary_path).read_text())
    return [r for r in results if r.get("representative") is True]


def _expected_spans(params):
    """Entropy window of each space; identical to step 4 ``_metric_spans``."""
    span = params.metric_shannon_span
    short = (int(params.cluster_max_distance) if span is None or span < 0
             else int(span))
    return {"long": 0, "short": short}


def _recompute_metrics(results, matrix_file, spans, L):
    """Recompute support_sum and ambiguity_mean from the 10 bp matrix.

    support   : sum over base pairs of M10[a//10, b//10]  (v1 ``calculate_pair_score``)
    ambiguity : mean over base pairs of H_a × H_b          (v1 ``calculate_pair_shannon``)
    """
    res = _METRIC_RES
    m = load_dump_sparse(matrix_file, res, L)
    dense = m.toarray()
    tracks = {s: shannon_track(dense, res, L, s) for s in set(spans.values())}
    n = dense.shape[0]

    updated = []
    for r in results:
        space = r.get("space", "short")
        span = spans.get(space, spans["short"])
        track = tracks[span]
        pairs = _read_pairs(Path(r["ct"]).with_suffix(".bedpe"))
        r = dict(r)
        if not pairs:
            updated.append(r)
            continue
        amb = np.array([float(track[a] * track[b]) for a, b in pairs])
        sup = [float(dense[a // res, b // res]) if a // res < n and b // res < n else 0.0
               for a, b in pairs]
        r["pairs"] = len(pairs)
        r["ambiguity_mean"] = round(float(amb.mean()), 4)
        r["ambiguity_span"] = span
        r["support_sum"] = round(float(sum(sup)), 4)
        updated.append(r)
    return updated


def _reuse_step4_metrics(results, spans):
    """Take support_sum / ambiguity_mean straight from the step 4 summary."""
    missing = [r.get("cluster") for r in results if r.get("pairs", 0) > 0
               and (_SUPPORT_KEY not in r or _AMBIGUITY_KEY not in r)]
    if missing:
        raise RuntimeError(
            f"step 4 summary lacks {_SUPPORT_KEY}/{_AMBIGUITY_KEY} for "
            f"{len(missing)} structure(s) (e.g. {missing[0]}); "
            "re-run step 4 or use source='recompute'")
    mismatch = {}
    for r in results:
        space = r.get("space", "short")
        want = spans.get(space, spans["short"])
        got = r.get("ambiguity_span")
        if got is not None and int(got) != want:
            mismatch.setdefault(space, set()).add(int(got))
    for space, got in sorted(mismatch.items()):
        log.warning("  %s-range: step 4 ambiguity window %s nt differs from the "
                    "expected %d nt; values are used as stored",
                    space, sorted(got), spans[space])
    return [dict(r) for r in results]


def _read_pairs(bedpe_path):
    """0-based (a, b) pairs from a BEDPE file."""
    if not Path(bedpe_path).exists():
        return []
    pairs = []
    with open(bedpe_path) as f:
        for line in f:
            if "chrom1" in line or not line.strip():
                continue
            p = line.split()
            a, b = int(p[1]), int(p[4])
            if a <= b:
                pairs.append((a, b))
    return pairs


def _flag_and_write(results, outdir, space):
    """Threshold within one structural space, write outputs, return thresholds."""
    reps = [r for r in results if r.get("space") == space and r.get("pairs", 0) > 0]
    if not reps:
        return {}
    sup_vals = np.array([r.get(_SUPPORT_KEY, 0.0) for r in reps])
    amb_vals = np.array([r.get(_AMBIGUITY_KEY, 0.0) for r in reps])
    sup_cut = float(sup_vals.mean())
    amb_cut = float(amb_vals.mean())
    n_well = 0
    for r in reps:
        r["well_supported"] = bool(r.get(_SUPPORT_KEY, 0.0) > sup_cut and r.get(_AMBIGUITY_KEY, 0.0) < amb_cut)
        if r["well_supported"]:
            n_well += 1
    # write BEDPE of well-folded structures
    out_bed = outdir / f"well_folded_{space}_range.bedpe"
    with open(out_bed, "w") as f:
        f.write(BEDPE_HEADER.rstrip("\n") + "\tname\n")
        for r in reps:
            if not r.get("well_supported"):
                continue
            bed = Path(r["ct"]).with_suffix(".bedpe")
            label = f"{r['cluster']}|structure{r['structure']}"
            if bed.exists():
                with open(bed) as b:
                    for line in b:
                        if "chrom1" in line or not line.strip():
                            continue
                        f.write(line.rstrip("\n") + "\t" + label + "\n")
    log.info("  %s-range: %d/%d well-folded -> %s", space, n_well, len(reps), out_bed.name)
    return {
        "space": space,
        "total": len(reps),
        "well_folded": n_well,
        "support_cut": round(sup_cut, 4),
        "ambiguity_cut": round(amb_cut, 4),
    }


# ------------------------------------------------------------------ driver
def run_step6(params, step2_info, step4_outdir, outdir, source="recompute",
              length=None):
    """Screen step 4 representative structures for local well-folded elements.

    Parameters
    ----------
    params       : RiverParams
    step2_info   : dict from step2_manifest.json (only used by "recompute")
    step4_outdir : directory of step 4 outputs (contains summary.json)
    outdir       : output directory for step 6
    source       : "recompute" (default) recalculates the metrics from the
                   10 bp matrix; "step4" reuses the values of step 4
    length       : RNA length (nt); inferred from the 10 bp matrix if None
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    outdir = ensure_dir(outdir)
    step4_outdir = Path(step4_outdir)
    summary_path = step4_outdir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"{summary_path} missing; run step 4 first")

    results = _load_representatives(summary_path)
    if not results:
        log.warning("Step6: no representative structures found in step 4 summary")
        return []

    spans = _expected_spans(params)
    log.info("Step6: %d representative structure(s), metrics from %s "
             "(entropy window: long %d nt, short %d nt; 0 = whole rna)",
             len(results), source, spans["long"], spans["short"])

    if source == "step4":
        results = _reuse_step4_metrics(results, spans)
    else:
        mats = {int(k): v for k, v in step2_info["matrices"].items()}
        if _METRIC_RES not in mats:
            raise RuntimeError(f"step 6 needs the {_METRIC_RES} bp matrix from step 2 "
                               f"(found {sorted(mats)})")
        if length is None:
            length = load_dump_sparse(mats[_METRIC_RES], _METRIC_RES).shape[0] * _METRIC_RES
        results = _recompute_metrics(results, mats[_METRIC_RES], spans, int(length))

    for space in ("short", "long"):
        _write_scatter(outdir / f"scatter_{space}_range.txt",
                       [r for r in results if r.get("space") == space])

    cuts = {}
    for space in ("short", "long"):
        c = _flag_and_write(results, outdir, space)
        if c:
            cuts[space] = c

    # merged output across both spaces
    all_well = [r for r in results if r.get("well_supported")]
    merged_bed = outdir / "well_folded_all.bedpe"
    with open(merged_bed, "w") as f:
        f.write(BEDPE_HEADER.rstrip("\n") + "\tname\n")
        for r in all_well:
            bed = Path(r["ct"]).with_suffix(".bedpe")
            label = f"{r['space']}|{r['cluster']}|structure{r['structure']}"
            if bed.exists():
                with open(bed) as b:
                    for line in b:
                        if "chrom1" in line or not line.strip():
                            continue
                        f.write(line.rstrip("\n") + "\t" + label + "\n")

    for r in results:
        r["metric_source"] = source
    (outdir / "summary.json").write_text(json.dumps(results, indent=2))
    (outdir / "thresholds.json").write_text(json.dumps(
        {"metric_source": source, "spans": spans, **cuts}, indent=2))
    log.info("Step6: %d well-folded structure(s) total -> %s", len(all_well), merged_bed.name)
    return results


def _write_scatter(path, results):
    """TSV scatter table: cluster, space, support_sum, ambiguity_mean, well_supported."""
    with open(path, "w") as f:
        f.write("cluster\tspace\tsupport_sum\tambiguity_mean\twell_supported\n")
        for r in results:
            f.write(f"{r.get('cluster','')}\t{r.get('space','')}\t"
                    f"{r.get(_SUPPORT_KEY, '')}\t{r.get(_AMBIGUITY_KEY, '')}\t"
                    f"{r.get('well_supported', '')}\n")
