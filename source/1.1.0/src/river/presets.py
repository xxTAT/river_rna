"""Parameter presets of RIVER.
See Supplementary Material 3 (RIVER modules and parameter settings) for the rationale of every value.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Optional, Tuple

AUTO_SHORT_MAX = 1000     # nt, used by --rna-type auto
AUTO_MEDIUM_MAX = 10000   # nt; sequences above this use windowed partition


@dataclass(frozen=True)
class RiverParams:
    rna_type: str
    # ---- resolutions (bp) ----
    low_res: int = 10            # canny without NMS, diagonal cross removed
    mid_res: int = 5             # canny with NMS + O/E mask; step3 feature grid
    high_res: int = 1            # dumped for completeness
    score_res: Tuple[int, ...] = (10, 5, 1)   # matrices merged for stem scoring
    norm_order: Tuple[str, ...] = ("KR", "NONE", "VC_SQRT", "VC")
    # ---- step1 partition ----
    partition_mode: str = "whole"     # whole | window
    window: int = 5000
    overlap: int = 3000
    # ---- step2 canny ----
    gaussian_sigma: float = 1.5
    gaussian_ksize: int = 3
    canny_tlf: float = 0.01
    canny_thf: float = 0.03
    rri_ctf: float = 0.01
    run_clusters: bool = True         # bedtools pixel clustering
    # ---- step3 RNA-wide structure prediction ----
    diag_zero_width: int = 10         # |i-j| < width is removed
    max_pair_distance: Optional[int] = None
    score_clip_percentile: float = 99.0
    score_mask: bool = False
    score_mask_pvalue: float = 0.75
    score_mask_percentile: float = 98.0
    unit_score_percentile: float = 96.0
    domain_split: bool = True
    domain_max_len: int = 600
    domain_edge_len: int = 100
    domain_gap_len: int = 200
    constraint_require_stack: bool = True
    # ---- step4: strucure prediction within each feature ----
    cluster_max_distance: int = 3000      # span end2-start1 -> long range above
    cluster_diagonal_gap: int = 10        # anchors closer than this -> diagonal
    cofold_pad: int = 2                   # nt added to each RNAcofold fragment
    cofold_max_retry: int = 5             # constraint relaxation attempts
    # ---- step4/5: structure metrics ----
    metric_shannon_span: int = -1         # nt window of the pairing-ambiguity
                                          # entropy of short-range features;
                                          # -1 = cluster_max_distance,
                                          # 0 = whole molecule. Long-range
                                          # features always use the whole
                                          # molecule
    # ---- step5: alternative structures ----
    alt_shannon_span: int = 3000          # nt window of the local entropy
    alt_cluster_score_percentile: float = 50.0
    alt_conflict_slop: int = 10           # nt added before the anchor tests
    alt_conflict_overlap_fraction: float = 0.5
    # ---- step6: local well-folded structures ----
    metric_support_stat: str = "sum"      # well-folded screening on the summed or the mean support per pair
    # ---- step7: global well-folded structures ----
    n_clusters: int = 5                  # retained cluster number 

    @property
    def dump_resolutions(self):
        return tuple(sorted({self.low_res, self.mid_res, self.high_res,
                             *self.score_res}, reverse=True))

    def to_dict(self):
        return asdict(self)

    def update(self, **kw):
        valid = {f.name for f in fields(self)}
        bad = set(kw) - valid
        if bad:
            raise ValueError(f"Unknown parameter(s): {sorted(bad)}")
        clean = {}
        for k, v in kw.items():
            if v is None and k != "max_pair_distance":
                continue
            if k in ("score_res", "norm_order") and not isinstance(v, tuple):
                v = tuple(v)
            clean[k] = v
        return replace(self, **clean)


PRESETS = {
    "short": RiverParams(
        rna_type="short", low_res=10, mid_res=5, high_res=1,
        partition_mode="whole", diag_zero_width=5, max_pair_distance=None,
        score_mask=False, domain_split=False, constraint_require_stack=False,
    ),
    "medium": RiverParams(
        rna_type="medium", low_res=10, mid_res=5, high_res=1,
        partition_mode="whole", diag_zero_width=10, max_pair_distance=1600,
        score_mask=True, domain_split=True, constraint_require_stack=True,
    ),
    "long": RiverParams(
        rna_type="long", low_res=15, mid_res=10, high_res=5,
        partition_mode="window", window=5000, overlap=3000,
        diag_zero_width=10, max_pair_distance=2500,
        score_mask=False, domain_split=True, constraint_require_stack=True,
    ),
}


def resolve_rna_type(rna_type: str, length: int) -> str:
    if rna_type != "auto":
        return rna_type
    if length < AUTO_SHORT_MAX:
        return "short"
    if length < AUTO_MEDIUM_MAX:
        return "medium"
    return "long"


def get_params(rna_type: str, length: int, overrides: Optional[dict] = None) -> RiverParams:
    t = resolve_rna_type(rna_type, length)
    if t not in PRESETS:
        raise ValueError(f"rna_type must be short/medium/long/auto, got {rna_type}")
    p = PRESETS[t]
    if overrides:
        p = p.update(**overrides)
    return p
