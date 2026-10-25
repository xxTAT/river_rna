# RIVER steps

Detailed description of the seven steps of `river predict`.
Parameter names refer to [`parameters.md`](parameters.md).

| step | directory | needs | external programs |
|---|---|---|---|
| 1 | `step1_partition` | FASTA | `partition-smp` (`partition-cuda`), `ProbabilityPlot` |
| 2 | `step2_structure_feature_detect` | `.hic` | `juicer_tools`; `bedtools` |
| 3 | `step3_optimal_structure_predict` | 1, 2 | `Fold-smp` (`Fold-cuda`) |
| 4 | `step4_cluster_structure_predict` | 1, 2 | `Fold-smp`, `RNAcofold`, `bipartition-smp` |
| 5 | `step5_alternative_structure_predict` | 1, 2 | `Fold-smp`, `RNAcofold` |
| 6 | `step6_local_wellfolded` | 4 (+ 2 with `--step6-source recompute`) | – |
| 7 | `step7_global_wellfolded` | 3 | – |

All coordinates in BEDPE files are 0-based; CT files are 1-based.
With `--resume`, files that already exist and are not empty are reused instead of being recomputed.

---

## Step 1 – base-pair probabilities

*Fusion scoring of candidate base pairs*

The partition function of the sequence is computed with RNAstructure `partition`, and the base-pair probabilities are exported with `ProbabilityPlot -t` as −log10 P.

* `partition_mode = whole` (presets `short`, `medium`): the whole sequence at once.
* `partition_mode = window` (preset `long`, i.e. RNAs ≥ 10,000 nt with `-r auto`): overlapping windows of `window` nt with `overlap` nt overlap. Where windows overlap, values are averaged.

**Output**

```
step1_partition/
├── tmp_fasta/<s>_<e>.fasta
├── tmp_pfs_txt/<s>_<e>_<s>_<e>.{pfs,txt}   # −log10 P per window
└── partition_file_id.txt                   # list of the .txt files
```

---

## Step 2 – structural feature localization

*Datasets and preprocessing; Structural feature localization*

1. The contact matrices are dumped from the `.hic` at every resolution needed by the preset (`low_res`, `mid_res`, `high_res`, `score_res`). For each resolution the normalisations in `norm_order` are tried in order (KR first); the first non-empty one is kept.
2. A modified Canny edge detector is run on two resolutions:
   * low resolution (`low_res`): the diagonal cross is removed, Gaussian filter, double threshold;
   * mid resolution (`mid_res`): O/E mask (pixels with O/E ≤ 1 removed), Gaussian filter, non-maximum suppression, double threshold.

   The thresholds are `canny_tlf` and `canny_thf` (1 % and 3 % of the maximum signal).
3. The low-resolution pixels are mapped onto the mid-resolution grid, gaps on the diagonal are filled, and the two results are merged.
4. Pixels are kept only where the Gaussian-smoothed interaction signal exceeds `rri_ctf` × its maximum.
5. Adjacent pixels are grouped into **structural features** (clusters) with `bedtools pairtopair` (within one `mid_res` unit). Each feature is the bounding box `(start1, end1, start2, end2)` of its pixels.

**Output**

```
step2_structure_feature_detect/
├── <hic>_<norm>_<res>.matrix                         # dumped matrices
├── <...>_<low>.bedpe, <...>_<mid>_OEmask.bedpe       # Canny pixels
├── <...>_remake<mid>_..._fill_diagonal_filter_gaussian.bedpe   # final feature pixels
├── <...>_filter_gaussian_clusters.bedpe              # structural features
└── step2_manifest.json                               # paths used by the later steps
```

---

## Step 3 – representative structure of the whole RNA

*Fusion scoring of candidate base pairs; Representative structure assembly*

1. **Candidate base pairs**: partition probabilities (step 1) inside the Canny feature pixels (step 2, O/E masked). Pairs with |i − j| < `diag_zero_width` and, if set, |i − j| ≥ `max_pair_distance` are removed.
2. **Fusion score** of a pair: FS = RRI × (1 + 1 / (−log10 P)), where RRI is the sum of the interaction signals at the `score_res` resolutions, clipped at the `score_clip_percentile` percentile. With `score_mask` (preset `medium`), weak pairs (P below `score_mask_pvalue` and signal below the `score_mask_percentile` percentile) get no interaction support.
3. **Stems**: candidate pairs are chained along anti-diagonals into stems; a stem's score is the sum of the fusion scores of its pairs. Stems below the `unit_score_percentile` percentile (96th) are dropped.
4. **Candidate structures**: stems are added greedily in descending order of score, skipping stems that conflict with or form a pseudoknot with those already chosen. Ties produce several candidate structural constraints, `structure1`, `structure2`, …
5. **Structure completion**: each sequence is folded with RNAstructure `Fold` using  constraints.
   
   * `domain_split = false` (preset `short`): the whole RNA is folded once.
   * `domain_split = true` (`medium`, `long`): the RNA is cut into domains of at most `domain_max_len` nt at the boundaries of the constraining stems (`domain_edge_len`, `domain_gap_len` control the joining of small pieces); each domain is folded separately and the domains are merged. With `constraint_require_stack`, only constraint pairs that stack on a neighbouring pair are kept.
   
   In each domain the first structure of the CT file that contains base pairs is used. If none does (the constraints cannot be satisfied), a warning names the region and the region is left unpaired.
6. **Representative**: among the candidates that have base pairs, the one with the lowest total free energy (sum over its domains) is marked `representative` in `summary.json`.

**Output**

```
step3_optimal_structure_predict/
├── structural_feature.bedpe          # candidate base pairs
├── units_check.bedpe                 # all stems
├── summary.json                      # one row per candidate: units, domains, pairs, energy, representative
└── structureN/
    ├── structureN.bedpe, units.txt   # selected stems
    ├── sequence_constrain/<s>_<e>.{fasta,const}, domain.txt
    ├── domain_structure/<s>_<e>.ct, <s>_<e>_sK.bedpe
    └── merge_structure/whole_structure.{ct,bedpe,dbn}
```

---

## Step 4 – representative structure of every structural feature

*Representative structure assembly (Feature-level structure reconstruction)*

Each feature of step 2 is folded on its own. Its span is `distance between the end of two feature arms`.

| feature | pairing probabilities | folding |
|---|---|---|
| long range: span > `cluster_max_distance` (3,000 nt) | `bipartition` of the two arms | `RNAcofold --constraint --enforceConstraint` |
| short range, arms apart (off-diagonal) | step 1 partition ∩ feature pixels | `RNAcofold --constraint --enforceConstraint` |
| short range, arms touching (diagonal; gap < `cluster_diagonal_gap`) | step 1 partition ∩ feature pixels | `Fold -c` on the whole region |

Inside the feature, stems are built, scored and combined into candidate structures as in step 3; the interaction support is clipped at the 99th percentile of the feature itself.

* **Constraint relaxation (RNAcofold)**: if RNAcofold cannot satisfy the constraint, the innermost constrained pair and all isolated constrained pairs are released and folding is repeated, at most `cofold_max_retry` times; after that the arms are folded without enforcing the constraint.
* **Representative**: when a feature has several candidates, the one with the lowest free energy is marked `representative`.
* **Structure attributes**, computed for every candidate from the 10 bp matrix:
  * `support_sum` / `support_mean` — interaction support of the base pairs (sum / per pair); `supported_pairs`, `supported_fraction`;
  * `ambiguity_mean` / `ambiguity_max` — pairing ambiguity PA(i,j) = H_i × H_j, where H is the Shannon entropy of a nucleotide's interaction profile. Long-range features use the entropy over the whole RNA; short-range features use partners within `metric_shannon_span` (default 3,000 nt). The window used is stored in `ambiguity_span`;
  * `pair_distance_mean`, `pair_distance_median`, `stem_length`;
  * `space` — `short` or `long`.

**Output**

```
step4_cluster_structure_predict/
├── clusters_{long_range,short_range_nondiagonal,short_range_diagonal}.bedpe
├── structure_feature.bedpe           # candidate pixels of short-range features
├── shannon.txt                       # per-nucleotide entropy, one column per window
├── summary.json                      # one row per candidate structure with all attributes
├── all_cluster_structures.{tsv,bedpe}
├── long_range/cluster_<s1>_<e1>_<s2>_<e2>/
│   ├── bipartition/<s>_<e>.fasta, <cluster>.{pfs,txt}
│   ├── units.bedpe
│   └── structureN/structureN.{fasta,cofold,ct,bedpe,dbn}, units.txt,
│                  structureN_annotation.txt, structureN_pairs.tsv
└── short_range/cluster_<s1>_<e1>_<s2>_<e2>/
    ├── units.bedpe
    └── structureN/
        ├── off-diagonal: structureN.{fasta,cofold,ct,bedpe,dbn}
        ├── diagonal:     structureN.{fasta,const}, structureN_fold.ct (raw Fold output),
        │                 structureN.{ct,bedpe,dbn}, structureN_cluster.bedpe (pairs inside the feature)
        └── units.txt, structureN_annotation.txt, structureN_pairs.tsv
```

---

## Step 5 – alternative structures

*Identification and evaluation of alternative structures*

![alternative_structure](source/1.1.0/docs/images/alternative_structure.png)

1. **High-entropy features**: the Shannon entropy of every nucleotide is computed within `alt_shannon_span` nt. Short-range features whose right arm contains a position with entropy ≥ the genome-wide mean are kept.
2. **Signal filter**: off-diagonal features must have a total interaction signal above the `alt_cluster_score_percentile` percentile (top 50 % default).
3. **Conflicts**: two features compete when an arm of one contains, or overlaps by more than `alt_conflict_overlap_fraction` (50 % default), an arm of the other (arms extended by `alt_conflict_slop` nt). Features without any Canny pixel are removed. Conflicts are classified as

   | category | `type` | meaning |
   |---|---|---|
   | `cross` | I | the right arm of one feature overlaps the left arm of the other |
   | `vertical_horizontal` | II | the two features share one arm (one region, two partners) |
   | `diagonal` | III | two local (diagonal) features compete for the same region |
   | `diagonal_inside_and_out` | IV | an off-diagonal feature competes with the local hairpins inside its own region; the local state is folded with the off-diagonal interaction forbidden |

4. Every partner of a conflict is folded separately with the step 4 routines, and the candidates of one conflict are compared by their interaction support (`conflict_scores.tsv`): `support_mean` (ranking criterion), `support_sum`, `support_share`, `rank`.

**Output**

```
step5_alternative_structure_predict/
├── shannon.txt
├── clusters_high_shannon.bedpe, clusters_nondiagonal_{kept,dropped}.bedpe
├── structure_feature.bedpe
├── conflict_{diagonal,cross,vertical_horizontal,diagonal_inside_and_out}.bedpe
├── conflict_scores.tsv               # all conflicts
├── all_alternative_structures.{tsv,bedpe}
├── summary.json
└── <category>/conflict_K/
    ├── conflict_scores.tsv           # this conflict only
    └── cluster_<...>/structureN/…    # (type IV: nondiagonal_<...>/ and diagonal_<s>_<e>/)
```

---

## Step 6 – local well-folded structures

*Identification of well-folded structures (Feature-derived well-folded structures)*

![local_well_folded_structure](source/1.1.0/docs/images/local_well_folded.png)

Only the representative structure of each feature (step 4) is evaluated. Short-range and long-range structures are screened **separately, in the same run**.

* **Interaction support** = sum over the base pairs of the 10 bp signal, M10[i//10, j//10].
* **Pairing ambiguity** = mean over the base pairs of H_i × H_j. Long range: entropy over the whole RNA; short range: entropy within `metric_shannon_span` (default 3,000 nt).
* A structure is **well-folded** when its support is **above** the mean and its ambiguity **below** the mean of its own structural space.

The two attributes can be obtained in two ways (`--step6-source`):

| value | behaviour |
|---|---|
| `recompute` (default) | reads the base pairs of every representative structure and recomputes both attributes from the 10 bp matrix of step 2 |
| `step4` | uses `support_sum` / `ambiguity_mean` stored in the step 4 `summary.json`; no matrix is read. A warning is logged if the entropy window stored by step 4 differs from the current setting |

**Output**

```
step6_local_wellfolded/
├── scatter_{short,long}_range.txt          # cluster, space, support_sum, ambiguity_mean, well_supported
├── well_folded_{short,long}_range.bedpe    # base pairs of the well-folded structures
├── well_folded_all.bedpe                   # both spaces, name = space|cluster|structureN
├── summary.json                            # representative structures with attributes and flag
└── thresholds.json                         # metric source, entropy windows, cut-offs per space
```

---

## Step 7 – global well-folded structures

*Identification of well-folded structures (Unit-based well-folded structures)*

![global_well_folded_structure](source/1.1.0/docs/images/global_well_folded.png)

1. The representative structure of step 3 (`representative` in its `summary.json`) is partitioned into **structural units**: the interval closed by an outermost base pair, i.e. a pair not enclosed by any other pair. Single-stranded gaps between units are excluded.
2. For each unit three attributes are computed from the 10 bp matrix:
   * `Pairing ambiguity` — mean of H_i × H_j over the unit's base pairs, entropy within 3,000 nt;
   * `Interaction support` — mean O/E-normalised interaction signal over the unit's base pairs (O/E matrix restricted to the 3,000 nt band);
   * `length` — unit length.
3. The attributes are Z-score standardised and the units are clustered by Ward's method (Euclidean distance) into `--n-clusters` clusters.
4. **Background medians**: over all positions (i, j) with 0 < |i − j| < 3,000, partition probability > 0 (step 1) and interaction signal > 0.
5. A cluster is **well-folded** when its median `Pairing ambiguity` is below and its median `Interaction support` is above the background medians. All units of a well-folded cluster are well-folded.

The 10 bp resolution and the 3,000 nt window are fixed, as in the analysis of the manuscript. `--n-clusters` has no default; inspect the result (e.g. the dendrogram of `structural_units.tsv`) and adjust it for each dataset.

**Output**

```
step7_global_wellfolded/
├── structural_units.tsv       # unit, start, end, ambiguity, support, length, cluster, well_folded
├── cluster_stats.tsv          # per cluster: n_units, median_ambiguity, median_support, well_folded
├── well_folded_units.tsv      # well-folded units only
├── well_folded_units.bedpe    # base pairs of the well-folded units, name = unit
└── summary.json               # background medians and cluster statistics
```
