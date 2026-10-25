"""Step 1: base-pair probabilities with RNAstructure partition/ProbabilityPlot.

whole  : the full sequence is computed at once (short/medium RNAs)
window : overlapping windows (long RNAs, e.g. 5000 nt with 3000 nt overlap)
"""
from __future__ import annotations

import logging
from pathlib import Path

from .matrix_utils import ensure_dir, write_fasta

log = logging.getLogger("river")


def split_windows(length, window, overlap):
    if length <= window:
        return [(0, length - 1)]
    frags, i = [], 0
    step = window - overlap
    if step <= 0:
        raise ValueError("window must be larger than overlap")
    while True:
        if i + window > length:
            frags.append((length - window, length - 1))
            break
        frags.append((i, i + window - 1))
        i += step
    seen, out = set(), []
    for f in frags:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def run_step1(params, tool, seq, outdir, resume=False):
    outdir = ensure_dir(outdir)
    fa_dir = ensure_dir(outdir / "tmp_fasta")
    txt_dir = ensure_dir(outdir / "tmp_pfs_txt")
    L = len(seq)
    if params.partition_mode == "window":
        frags = split_windows(L, params.window, params.overlap)
    else:
        frags = [(0, L - 1)]
    log.info("Step1: %d partition region(s)%s", len(frags), " [GPU]" if tool.use_gpu else "")
    names = []
    for k, (s, e) in enumerate(frags, 1):
        prefix = f"{s}_{e}_{s}_{e}"
        fa = fa_dir / f"{s}_{e}.fasta"
        pfs = txt_dir / f"{prefix}.pfs"
        txt = txt_dir / f"{prefix}.txt"
        if resume and txt.exists() and txt.stat().st_size > 0:
            log.info("  [%d/%d] %s exists, skipped", k, len(frags), txt.name)
        else:
            write_fasta(fa, f"{s}_{e}", seq[s:e + 1])
            log.info("  [%d/%d] partition %s", k, len(frags), prefix)
            tool.partition(fa, pfs)
            tool.probability_plot(pfs, txt)
        names.append(txt.name)
    list_file = outdir / "partition_file_id.txt"
    list_file.write_text("".join(n + "\n" for n in names))
    return list_file
