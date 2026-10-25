"""Command-line interface: `river predict`, `river batch`, `river presets`."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .external import Toolchain
from .matrix_utils import ensure_dir, load_fasta, write_fasta
from .presets import PRESETS, get_params

log = logging.getLogger("river")

OVERRIDE_ARGS = {
    # cli option: (dest, type, help)
    "--window": ("window", int, "step1: partition window (long RNA)"),
    "--overlap": ("overlap", int, "step1: partition window overlap (long RNA)"),
    "--canny-tlf": ("canny_tlf", float, "step2: low threshold factor"),
    "--canny-thf": ("canny_thf", float, "step2: high threshold factor"),
    "--rri-ctf": ("rri_ctf", float, "step2: RRI gaussian cutoff factor"),
    "--low-res": ("low_res", int, "step2/3/4/5: low resolution (nt)"),
    "--mid-res": ("mid_res", int, "step2/3/4/5: mid resolution (nt)"),
    "--high-res": ("high_res", int, "step2/3/4/5: high resolution (nt)"),
    "--max-pair-distance": ("max_pair_distance", int, "step3: maximum base-pair span; 0 disables"),
    "--unit-score-percentile": ("unit_score_percentile", float, "step3/4/5: percentile cut of stem scores"),
    "--diag-zero-width": ("diag_zero_width", int, "step3/4/5: remove pairs with |i-j| < width"),
    "--cluster-max-distance": ("cluster_max_distance", int, "step4/5/6: span above which a feature is long-range"),
    "--cluster-diagonal-gap": ("cluster_diagonal_gap", int, "step4/5/6: anchor gap below which a feature is diagonal"),
    "--metric-shannon-span": ("metric_shannon_span", int, "step4/5: entropy window (nt) of the pairing ambiguity "
                              "of short-range features; -1 = cluster_max_distance (default), 0 = whole molecule."
                              "Long-range features always use the whole molecule"),
#    "--metric-support-stat": ("metric_support_stat", str, "well-folded screening on 'sum' or 'mean' support per base pair"),
    "--alt-shannon-span": ("alt_shannon_span", int, "step5: window of the local Shannon entropy (nt)"),
    "--alt-cluster-score-percentile": ("alt_cluster_score_percentile", float, "step5: percentile cut of the feature scores"),
}


def setup_logging(outdir=None, verbose=False):
    root = logging.getLogger("river")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(fmt)
    root.addHandler(h)
    if outdir:
        fh = logging.FileHandler(Path(outdir) / "river.log")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def load_config(path):
    if not path:
        return {}
    text = Path(path).read_text()
    if str(path).endswith(".json"):
        return json.loads(text)
    import yaml
    return yaml.safe_load(text) or {}


def predict(fasta, hic, chrom, outdir, rna_type="auto", seq_id=None, length=None,
            use_gpu=False, use_cython=False, threads=1, steps=(1, 2), resume=False,
            config=None, overrides=None, n_clusters=None, step6_source="recompute"):
    """Python API: run the RIVER pipeline."""
    t0 = time.time()
    outdir = ensure_dir(outdir)
    name, seq = load_fasta(fasta, seq_id) #seq_id str(seq)
    seq = seq.upper()
    if length and length != len(seq):
        log.warning("--length %d differs from FASTA length %d; FASTA length is used",
                    length, len(seq))
    ov = dict(config or {})
    ov.pop("rna_type", None)
    ov.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if ov.get("max_pair_distance") == 0:
        ov["max_pair_distance"] = None
    params = get_params(rna_type, len(seq), ov)
    log.info("RIVER %s | %s (%d nt) | type=%s | gpu=%s | cython=%s", __version__, name, len(seq), params.rna_type, use_gpu, use_cython)

    prep = ensure_dir(outdir / "step0_prepare")
    write_fasta(prep / "input.fasta", name, seq)
    run_info = {"version": __version__, "sequence": name, "length": len(seq),
                "fasta": str(Path(fasta).resolve()), "hic": str(Path(hic).resolve()),
                "chrom": chrom, "gpu": use_gpu, "cython": use_cython,
                "params": params.to_dict()}
    (prep / "run_config.json").write_text(json.dumps(run_info, indent=2, default=list))

    tool = Toolchain(use_gpu=use_gpu, threads=threads)
    tool.check(steps)

    s1_dir = outdir / "step1_partition"
    s2_dir = outdir / "step2_structure_feature_locate"
    s3_dir = outdir / "step3_optimal_structure_predict"
    s4_dir = outdir / "step4_feature_structure_predict"
    s5_dir = outdir / "step5_alternative_structure_predict"
    s6_dir = outdir / "step6_local_wellfolded_screen"
    s7_dir = outdir / "step7_global_wellfolded_screen"
    list_file = s1_dir / "partition_file_id.txt"
    s2_info = None

    def _need_inputs():
        nonlocal s2_info
        if s2_info is None:
            manifest = s2_dir / "step2_manifest.json"
            if not manifest.exists():
                raise FileNotFoundError(f"{manifest} missing; run step 2 first")
            s2_info = json.loads(manifest.read_text())
        if not list_file.exists():
            raise FileNotFoundError(f"{list_file} missing; run step 1 first")
        return s2_info

    if 1 in steps:
        from .step1_partition import run_step1
        list_file = run_step1(params, tool, seq, s1_dir, resume)
    if 2 in steps:
        from .step2_features import run_step2
        s2_info = run_step2(params, tool, Path(hic).resolve(), chrom, len(seq), s2_dir, resume)
    if 3 in steps:
        from .step3_predict import run_step3
        res = run_step3(params, tool, seq, chrom, list_file, _need_inputs(), s3_dir, use_cython=use_cython, resume=resume)
        for r in res:
            log.info("  structure%d: %d pairs -> %s", r["structure"], r["pairs"], r["ct"])
    if 4 in steps:
        from .step4_cluster_predict import run_step4
        run_step4(params, tool, seq, chrom, list_file, _need_inputs(), s4_dir, resume=resume)
    if 5 in steps:
        from .step5_alternative import run_step5
        run_step5(params, tool, seq, chrom, list_file, _need_inputs(), s5_dir, resume=resume)
    if 6 in steps:
        from .step6_local_wellfolded import run_step6
        if not s4_dir.exists():
            raise FileNotFoundError(f"{s4_dir} missing; run step 4 first")
        if step6_source == "step4":
            run_step6(params, None, s4_dir, s6_dir, source="step4", length=len(seq))
        else:
            run_step6(params, _need_inputs(), s4_dir, s6_dir, source="recompute", length=len(seq))
    if 7 in steps:
        from .step7_global_wellfolded import run_step7
        if not n_clusters:
            raise ValueError("--n-clusters is required for step 7")
        run_step7(params, _need_inputs(), s3_dir, s7_dir,
                  n_clusters=n_clusters, step1_list=list_file, chrom=chrom)
    log.info("Finished in %.1f min", (time.time() - t0) / 60)
    return outdir


# ------------------------------------------------------------------ parsers
def _add_common(p):
    g = p.add_argument_group("runtime")
    g.add_argument("--gpu", action="store_true", help="use partition-cuda / Fold-cuda")
    g.add_argument("--cython", action="store_true", help="use the Cython/OpenMP expansion kernel")
    g.add_argument("-t", "--threads", type=int, default=1, help="OMP threads for RNAstructure")
    g.add_argument("--steps", default="1,2",
                   help="steps to run: 1 partition (default), 2 features(default), 3 rna-wide structure, "
                        "4 structure of each feature, 5 alternative structures, 6 local well-folded,"
                        "6 local well-folded, 7 global well-folded (e.g. 1,2,3,4,5,6,7 or 4,6)")
    g.add_argument("--n-clusters", dest="n_clusters", type=int,
                   default=None, help="number of Ward clusters for step 7 global well-folded analysis")
    g.add_argument("--step6-source", dest="step6_source", default="recompute", choices=["recompute", "step4"],
                   help="step 6 metrics: 'recompute' from the 10 bp matrix (default) "
                        "or reuse the step 4 summary.json values ('step4')")
    g.add_argument("--resume", action="store_true", help="reuse existing intermediate files")
    g.add_argument("--config", help="YAML/JSON file overriding preset parameters")
    g.add_argument("-v", "--verbose", action="store_true")
    o = p.add_argument_group("parameter overrides")
    for opt, (dest, typ, hlp) in OVERRIDE_ARGS.items():
        o.add_argument(opt, dest=dest, type=typ, default=None, help=hlp)


def _overrides(args):
    return {dest: getattr(args, dest) for _, (dest, _, _) in OVERRIDE_ARGS.items()}


def _steps(s):
    return tuple(sorted({int(x) for x in s.split(",") if x.strip()}))


def build_parser():
    ap = argparse.ArgumentParser(prog="river", description="RIVER")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="predict the structure using interaction data")
    p.add_argument("-f", "--fasta", required=True)
    p.add_argument("-s", "--seq-id", help="record in FASTA (default: first)")
    p.add_argument("--hic", required=True, help=".hic contact map")
    p.add_argument("-c", "--chrom", required=True, help="chromosome name inside the .hic")
    p.add_argument("-r", "--rna-type", default="auto", choices=["auto", "short", "medium", "long"])
    p.add_argument("-l", "--length", type=int, help="(optional) expected length check")
    p.add_argument("-o", "--outdir", required=True)
    _add_common(p)

    pr = sub.add_parser("presets", help="print preset parameters")
    pr.add_argument("rna_type", nargs="?", choices=list(PRESETS))
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "presets":
        sel = [args.rna_type] if args.rna_type else list(PRESETS)
        print(json.dumps({k: PRESETS[k].to_dict() for k in sel}, indent=2, default=list))
        return 0
    cfg = load_config(args.config)
    common = dict(use_gpu=args.gpu, use_cython=args.cython, threads=args.threads,
                  steps=_steps(args.steps),resume=args.resume,
                  config=cfg, overrides=_overrides(args),
                  n_clusters=getattr(args, "n_clusters", None))
    if args.cmd == "predict":
        out = ensure_dir(args.outdir)
        setup_logging(out, args.verbose)
        predict(args.fasta, args.hic, args.chrom, out, args.rna_type, args.seq_id,
                args.length, **common)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
