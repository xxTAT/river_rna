"""Step 3: rna-wide structure prediction.

1. partition probability matrix (-log10 P) with near-diagonal / long-range removal
2. intersection with Canny feature pixels -> candidate pixels
3. candidate stems + fusion scores
4. greedy conflict/pseudoknot-free structure tree -> candidate structures
5. RNAstructure Fold with pair constraints
   short : whole molecule
   medium/long : structural domains
"""
from __future__ import annotations

import gc
import json
import logging
import warnings
from pathlib import Path

import numpy as np

from .accel import get_expander
from .domains import connect_domain_and_fill_blank, fallback_domains, split_domain
from .folding import parse_ct, to_dot_bracket, write_constraint
from .matrix_utils import (BEDPE_HEADER, ensure_dir, get_oe, load_dump_matrix,
                           mask_by_oe, pad_or_crop, read_bedpe_starts, write_fasta)
from .structure_tree import build_structure_paths, chain_units
from .structure_tree import compatible, is_pseudoknot  # noqa: F401  (re-export)

log = logging.getLogger("river")


# --------------------------------------------------------------------- inputs
def build_partition_matrix(list_file, txt_dir, length):
    """-log10 P matrix; overlapping windows are averaged sequentially"""
    m = np.zeros((length, length))
    for name in Path(list_file).read_text().split():
        parts = name.replace(".txt", "").split("_")
        ls, rs = int(parts[0]), int(parts[2])
        d = np.loadtxt(Path(txt_dir) / name, skiprows=2, ndmin=2)
        if d.size == 0:
            continue
        keep = d[:, 2] > 0
        r = d[keep, 0].astype(np.int64) - 1 + ls
        c = d[keep, 1].astype(np.int64) - 1 + rs
        v = d[keep, 2]
        cur = m[r, c]
        m[r, c] = np.where(cur == 0, v, (cur + v) / 2)
    return m


def restrict_distance(m, zero_width, max_dist):
    """Remove |i-j| < zero_width and (optionally) |i-j| >= max_dist, in place."""
    L = m.shape[0]
    for i in range(L):
        m[i, max(0, i - zero_width + 1):min(L, i + zero_width)] = 0
        if max_dist:
            m[i, :max(0, i - max_dist + 1)] = 0
            m[i, i + max_dist:] = 0
    return m


def load_canny_matrix(bedpe, res, length):
    s1, _, s2, _ = read_bedpe_starts(bedpe)
    r, c = s1 // res, s2 // res
    n = -(-length // res) + 2
    if len(r):
        n = max(n, int(r.max()) + 1, int(c.max()) + 1)
    m = np.zeros((n, n))
    m[r, c] = 1
    return m


# ---------------------------------------------------------------- units
def number_structural_units(rows, cols, length, chrom, out_bedpe):
    """Chain candidate pixels along anti-diagonals (i, j) -> (i+1, j-1)."""
    unit_loc = chain_units(zip(rows.tolist(), cols.tolist()))
    with open(out_bedpe, "w") as f:
        f.write(BEDPE_HEADER)
        for locs in unit_loc.values():
            for a, b in locs:
                f.write(f"{chrom}\t{a}\t{a + 1}\t{chrom}\t{b}\t{b + 1}\n")
    return unit_loc


def score_units(unit_loc, feature_val, merge, params):
    p_clip = np.percentile(merge, params.score_clip_percentile)
    np.minimum(merge, p_clip, out=merge)
    p_mask = None
    thr = -np.log10(params.score_mask_pvalue)
    if params.score_mask:
        pos = merge[merge > 0]
        if pos.size:
            p_mask = np.percentile(pos, params.score_mask_percentile)
    scores = []
    for uid, locs in unit_loc.items():
        count = 0
        for rc in locs:
            m = merge[rc]
            pv = feature_val[rc]
            if p_mask is not None and pv <= thr and m <= p_mask:
                m = 0.0
            count += m + (m / pv)
        scores.append((uid, count))
    if not scores:
        return []
    cut = np.percentile([s for _, s in scores], params.unit_score_percentile)
    return [(u, s) for u, s in scores if s >= cut]


# --------------------------------------------------------- folding & output
def domain_constraints(domains, pairs, require_stack):
    ps = set(pairs)
    if require_stack:
        keep = {p for p in pairs if (p[0] - 1, p[1] + 1) in ps or (p[0] + 1, p[1] - 1) in ps}
    else:
        keep = ps
    log.info("    constraint pairs: %d of %d kept", len(keep), len(ps))
    out = {}
    for s, e in domains:
        out[(s, e)] = sorted((a - s + 1, b - s + 1) for a, b in keep
                             if s <= a < e and s <= b < e and a <= b)
    return out



def fold_and_merge(tool, seq, domains, constraints, workdir, chrom, resume=False):
    L = len(seq)
    cdir = ensure_dir(Path(workdir) / "sequence_constrain")
    sdir = ensure_dir(Path(workdir) / "domain_structure")
    mdir = ensure_dir(Path(workdir) / "merge_structure")
    merged = {}
    total_energy = 0.0
    (cdir / "domain.txt").write_text("".join(f"{s}_{e}\n" for s, e in domains))
    for k, (s, e) in enumerate(domains, 1):
        prefix = f"{s}_{e}"
        fa, cs, ct = cdir / f"{prefix}.fasta", cdir / f"{prefix}.const", sdir / f"{prefix}.ct"
        write_fasta(fa, prefix, seq[s:e])
        write_constraint(cs, constraints[(s, e)])
        if not (resume and ct.exists() and ct.stat().st_size > 0):
            log.info("    [%d/%d] Fold %s (%d nt)%s", k, len(domains), prefix, e - s,
                     " [GPU]" if tool.use_gpu else "")
            tool.fold(fa, ct, cs if constraints[(s, e)] else None)
        structs, energy = parse_ct(ct, s)
        # Take the first structure in the CT file (structure1); this one corresponds to the lowest free energy in practice.
        # If it has no base pair (e.g. constraints too strict), fall back to the next one.
        first = next((sid for sid, prs in structs.items() if prs), None)
        if first is None:
            msg = (f"current constraints could not produce a structure in region "
                   f"{prefix} ({chrom}:{s + 1}-{e}); region left unpaired")
            log.warning("    %s", msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        elif first != next(iter(structs)):
            log.info("    %s: first structure has no pair; using %s instead", prefix, first)
        if first is not None and energy.get(first, 10000) < 10000:
            total_energy += energy[first]
        for sid, prs in structs.items():
            with open(sdir / f"{prefix}_{sid}.bedpe", "w") as bf:
                bf.write(BEDPE_HEADER)
                for a, b in prs:
                    bf.write(f"{chrom}\t{a - 1}\t{a}\t{chrom}\t{b - 1}\t{b}\n")
        if first:
            for a, b in structs[first]:
                merged[a] = b
                merged[b] = a
    pairs = sorted({(min(a, b), max(a, b)) for a, b in merged.items()})
    with open(mdir / "whole_structure.ct", "w") as wc:
        wc.write(f"{L}\n")
        for i in range(1, L + 1):
            wc.write(f"{i} {seq[i - 1]} {i - 1} {i + 1 if i < L else 0} {merged.get(i, 0)} {i}\n")
    with open(mdir / "whole_structure.bedpe", "w") as wb:
        wb.write(BEDPE_HEADER)
        for a, b in pairs:
            wb.write(f"{chrom}\t{a - 1}\t{a}\t{chrom}\t{b - 1}\t{b}\n")
    (mdir / "whole_structure.dbn").write_text(f">whole_structure\n{seq}\n{to_dot_bracket(pairs, L)}\n")
    return mdir / "whole_structure.ct", len(pairs), round(total_energy, 2)


# ------------------------------------------------------------------- driver
def run_step3(params, tool, seq, chrom, step1_list, step2_info, outdir,
              use_cython=False, resume=False):
    outdir = ensure_dir(outdir)
    L = len(seq)
    expand = get_expander(use_cython)
    mats = {int(k): v for k, v in step2_info["matrices"].items()}
    mid = params.mid_res

    log.info("Step3: loading data (L=%d)", L)
    part = build_partition_matrix(step1_list, Path(step1_list).parent / "tmp_pfs_txt", L)
    restrict_distance(part, params.diag_zero_width, params.max_pair_distance)

    canny = load_canny_matrix(step2_info["feature_bedpe"], mid, L)
    mask_by_oe(canny, get_oe(load_dump_matrix(mats[mid], mid)))
    canny = expand(canny, mid, 1, L)
    sel = (canny == 1) & (part > 0)
    del canny
    rows, cols = np.nonzero(sel)
    del sel
    vals = part[rows, cols]
    del part
    gc.collect()
    with open(outdir / "structural_feature.bedpe", "w") as f:
        for r, c in zip(rows.tolist(), cols.tolist()):
            f.write(f"{chrom}\t{r}\t{r + 1}\t{chrom}\t{c}\t{c + 1}\n")
    feature_val = dict(zip(zip(rows.tolist(), cols.tolist()), vals.tolist()))
    log.info("  candidate pixels: %d", len(feature_val))

    unit_loc = number_structural_units(rows, cols, L, chrom, outdir / "units_check.bedpe")
    log.info("  structural units: %d", len(unit_loc))
    if not unit_loc:
        raise RuntimeError("No structural unit found; check the input data/parameters.")

    merge = np.zeros((L, L))
    for res in params.score_res:
        m = load_dump_matrix(mats[res], res)
        merge += pad_or_crop(expand(m, res, 1, L), L)
        del m
    unit_scores = score_units(unit_loc, feature_val, merge, params)
    del merge
    gc.collect()
    log.info("  units passing score percentile: %d", len(unit_scores))

    paths = build_structure_paths(unit_scores, unit_loc, L)
    log.info("  candidate structures: %d", len(paths))
    results = []
    for n, path in enumerate(paths, 1):
        wdir = ensure_dir(outdir / f"structure{n}")
        bed = wdir / f"structure{n}.bedpe"
        pairs = []
        with open(bed, "w") as f:
            for uid in path:
                if uid == "s0":
                    continue
                for a, b in unit_loc[uid]:
                    f.write(f"{chrom}\t{a}\t{a + 1}\t{chrom}\t{b}\t{b + 1}\n")
                    pairs.append((a, b))
        (wdir / "units.txt").write_text("\n".join(path) + "\n")
        if params.domain_split:
            try:
                dl = split_domain(bed, L, chrom)
                domains = connect_domain_and_fill_blank(
                    dl, bed, L, chrom, params.domain_max_len,
                    params.domain_edge_len, params.domain_gap_len)
            except (IndexError, ValueError) as e:
                log.warning("  domain construction failed (%s); using fixed windows", e)
                domains = []
            domains = [d for d in dict.fromkeys(domains) if d[1] > d[0]]
            if not domains:
                domains = fallback_domains(L, params.domain_max_len)
        else:
            domains = [(0, L)]
        log.info("  structure%d: %d unit(s), %d domain(s)", n, len(path) - 1, len(domains))
        cons = domain_constraints(domains, pairs, params.constraint_require_stack)
        ct, npairs, energy = fold_and_merge(tool, seq, domains, cons, wdir, chrom, resume)
        results.append({"structure": n, "units": len(path) - 1, "domains": len(domains),
                        "pairs": npairs, "energy": energy, "ct": str(ct)})
    # representative: candidate structure with the lowest total free energy
    # (structures without any base pair are not eligible)
    pool = [r for r in results if r["pairs"] > 0] or results
    best = min(pool, key=lambda r: r["energy"])["structure"] if pool else None
    for r in results:
        r["representative"] = r["structure"] == best
    if best is not None:
        log.info("  representative: structure%d (energy %.2f)", best,
                 next(r["energy"] for r in results if r["structure"] == best))
    (outdir / "summary.json").write_text(json.dumps(results, indent=2))
    return results
