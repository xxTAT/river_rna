"""Wrappers around external programs (RNAstructure, juicer_tools, bedtools).

GPU mode replaces the SMP executables of RNAstructure by the CUDA builds:
    partition-smp -> partition-cuda
    Fold-smp      -> Fold-cuda
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger("river")

GPU_MAP = {"partition-smp": "partition-cuda", "Fold-smp": "Fold-cuda"}


class ToolError(RuntimeError):
    pass


class Toolchain:
    def __init__(self, use_gpu=False, threads=1, bedtools="bedtools", juicertools="juicer_tools", rnacofold="RNAcofold"):
        self.use_gpu = use_gpu
        self.threads = int(threads)
        self.bedtools = bedtools
        self.juicertools = juicertools
        self.rnacofold = rnacofold

    # ------------------------------------------------------------------ utils
    def exe(self, name: str) -> str:
        return GPU_MAP.get(name, name) if self.use_gpu else name

    def run(self, args, cwd=None, check=True, stdin=None, capture=False):
        args = [str(a) for a in args]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.threads)
        log.debug("$ %s", " ".join(args))
        res = subprocess.run(args, cwd=cwd, env=env, input=stdin,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE if capture else subprocess.STDOUT,
                             text=True)
        if check and res.returncode != 0:
            err = res.stderr if capture else res.stdout
            raise ToolError(f"Command failed ({res.returncode}): {' '.join(args)}\n{err}")
        return res.stdout

    def check(self, steps):
        """Fail early if a required program is missing."""
        need = []
        if 1 in steps:
            need += [self.exe("partition-smp"), "ProbabilityPlot"]
        if 2 in steps:
            need += [self.juicertools]
        if 3 in steps or 4 in steps or 5 in steps:
            need += [self.exe("Fold-smp")]
        if 4 in steps or 5 in steps:
            need += [self.rnacofold]
        if 4 in steps:
            need += [self.bipartition_exe]
        missing = [n for n in need if shutil.which(n) is None]
        if missing:
            raise ToolError("Missing executables in PATH: " + ", ".join(missing))
        if (set(steps) & {1, 3, 4, 5}) and "DATAPATH" not in os.environ:
            log.warning("DATAPATH is not set; RNAstructure needs it to locate its thermodynamic parameter tables.")

    # --------------------------------------------------------- RNAstructure
    def partition(self, fasta, pfs):
        self.run([self.exe("partition-smp"), fasta, pfs])

    def probability_plot(self, pfs, txt):
        self.run(["ProbabilityPlot", "-t", pfs, txt])

    def fold(self, fasta, ct, constraint=None):
        args = [self.exe("Fold-smp"), fasta, ct]
        if constraint:
            args += ["-c", constraint]
        self.run(args)

    @property
    def bipartition_exe(self):
        """`bipartition-smp` if available, else the serial `bipartition`."""
        if shutil.which("bipartition-smp") is None and shutil.which("bipartition"):
            return "bipartition"
        return "bipartition-smp"

    def bipartition(self, fasta1, fasta2, pfs):
        """Base-pair probabilities between two sequences (long-range clusters)."""
        self.run([self.bipartition_exe, fasta1, fasta2, pfs])

    # --------------------------------------------------------------- Vienna
    def cofold(self, fasta, out, enforce=True, cwd=None):
        """RNAcofold of a `seq1&seq2` FASTA with its constraint line.

        The input is piped, which works with every ViennaRNA 2.x release; an
        empty output means the enforced constraint is unsatisfiable.
        """
        args = [self.rnacofold, "--constraint", "--noPS"]
        if enforce:
            args.append("--enforceConstraint")
        else:
            args.remove("--constraint")
        text = Path(fasta).read_text()
        stdout = self.run(args, cwd=cwd, check=False, stdin=text, capture=True)
        Path(out).write_text(stdout if stdout.strip() else "")
        return Path(out).stat().st_size > 0

    # -------------------------------------------------------------- hic dump
    def dump_observed(self, hic, chrom, res, norm, out):
        out = Path(out)
        cmd = [
            self.juicertools,
            "dump",
            "observed",
            norm,
            hic,
            chrom,
            chrom,
            "BP",
            res,
            out,
        ]
        self.run(cmd)
        return out.exists() and out.stat().st_size > 0

    # ------------------------------------------------------------- bedtools
    def pairtopair(self, a, b, slop, out):
        if shutil.which(self.bedtools) is None:
            raise ToolError("bedtools not found")
        txt = self.run([self.bedtools, "pairtopair", "-a", a, "-b", b,
                        "-is", "-rdn", "-slop", slop])
        Path(out).write_text(txt)
