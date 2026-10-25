# RIVER options and parameters

* [Commands](#commands)
* [Command-line options](#command-line-options)
* [How parameters are resolved](#how-parameters-are-resolved)
* [Presets](#presets)
* [All parameters](#all-parameters)
* [Fixed values](#fixed-values)

Algorithmic details of each step: [`steps.md`](steps.md).

---

## Commands

| command | purpose |
|---|---|
| `river predict` | run the pipeline for one RNA |
| `river presets [short\|medium\|long]` | print the preset parameters as JSON |

---

## Command-line options

### Input / output (`predict`)

| option | default | description |
|---|---|---|
| `-f, --fasta` | required | RNA sequence |
| `-s, --seq-id` | first record | record to use from the FASTA |
| `--hic` | required | `.hic` interaction map |
| `-c, --chrom` | required | chromosome name inside the `.hic` |
| `-r, --rna-type` | `auto` | preset: `short`, `medium`, `long` or `auto` (by length) |
| `-l, --length` | – | expected length; only checked against the FASTA |
| `-o, --outdir` | required | output directory |

### Runtime (`predict`)

| option | default | description |
|---|---|---|
| `--steps` | `1,2 | comma-separated steps to run, e.g. `1,2,3,4,5,6,7` or `4,6` |
| `--resume` | off | reuse existing, non-empty intermediate files |
| `--config` | – | YAML or JSON file overriding preset parameters |
| `--n-clusters` | – | number of Ward clusters in step 7; **required** when step 7 runs |
| `--step6-source` | `recompute` | step 6 attributes: `recompute` from the 10 bp matrix, or `step4` to reuse the step 4 values |
| `--gpu` | off | use `partition-cuda` / `Fold-cuda` (`bipartition` and `RNAcofold` stay on the CPU) |
| `--cython` | off | use the compiled OpenMP kernel for matrix expansion (falls back to NumPy if not built) |
| `-t, --threads` | 1 | `OMP_NUM_THREADS` for RNAstructure |
| `-v, --verbose` | off | debug logging |

### Parameter overrides

These options override single entries of the parameter table below.

| option | parameter |
|---|---|
| `--low-res`, `--mid-res`, `--high-res` | `low_res`, `mid_res`, `high_res` |
| `--window`, `--overlap` | `window`, `overlap` |
| `--max-pair-distance` | `max_pair_distance` (`0` = no limit) |
| `--diag-zero-width` | `diag_zero_width` |
| `--unit-score-percentile` | `unit_score_percentile` |
| `--canny-tlf`, `--canny-thf` | `canny_tlf`, `canny_thf` |
| `--rri-ctf` | `rri_ctf` |
| `--cluster-max-distance` | `cluster_max_distance` |
| `--cluster-diagonal-gap` | `cluster_diagonal_gap` |
| `--metric-shannon-span` | `metric_shannon_span` |
| `--alt-shannon-span` | `alt_shannon_span` |
| `--alt-cluster-score-percentile` | `alt_cluster_score_percentile` |

Every other parameter can be changed through `--config`.

---

## How parameters are resolved

```
preset  →  --config file  →  command-line overrides
```

Each layer overrides the previous one; keys that are not given keep their value. Unknown keys in a config file are an error.
The final values are written to `step0_prepare/run_config.json`.

`-r auto` chooses the preset from the sequence length:

| length | preset |
|---|---|
| < 1,000 nt | `short` |
| 1,000 – 9,999 nt | `medium` |
| ≥ 10,000 nt | `long` |

---

## Presets

Values that differ between presets; all other parameters use the defaults in the next section.

| parameter | `short` | `medium` | `long` |
|---|---|---|---|
| model system | RPPH1 | 25S / 28S rRNA | coronavirus genomes |
| `low_res` / `mid_res` / `high_res` | 10 / 5 / 1 | 10 / 5 / 1 | 15 / 10 / 5 |
| `partition_mode` | `whole` | `whole` | `window` (5,000 / 3,000) |
| `diag_zero_width` | 5 | 10 | 10 |
| `max_pair_distance` | none | 1000/1600 | 2500 |
| `score_mask` | no | yes | no |
| `domain_split` | no | yes | yes |
| `constraint_require_stack` | no | yes | yes |

---

## All parameters

### Resolutions and matrices (steps 2–5)

| parameter | default | description |
|---|---|---|
| `low_res` | 10 | resolution (bp) of the low-resolution Canny pass (no NMS, diagonal cross removed) |
| `mid_res` | 5 | resolution of the main Canny pass (NMS, O/E mask); grid of the feature pixels |
| `high_res` | 1 | additional resolution dumped from the `.hic` |
| `score_res` | [10, 5, 1] | resolutions summed into the interaction support of the fusion score |
| `norm_order` | [KR, NONE, VC_SQRT, VC] | normalisations tried, in order, when dumping a matrix |

### Step 1 – partition function

| parameter | default | description |
|---|---|---|
| `partition_mode` | `whole` | `whole` = one computation; `window` = overlapping windows |
| `window` | 5000 | window length (nt), `window` mode only |
| `overlap` | 3000 | window overlap (nt) |

### Step 2 – feature localization

| parameter | default | description |
|---|---|---|
| `gaussian_sigma` | 1.5 | σ of the Gaussian filter |
| `gaussian_ksize` | 3 | kernel size of the Gaussian filter |
| `canny_tlf` | 0.01 | low hysteresis threshold, fraction of the maximum signal |
| `canny_thf` | 0.03 | high hysteresis threshold, fraction of the maximum signal |
| `rri_ctf` | 0.01 | feature pixels are kept where the smoothed signal exceeds this fraction of its maximum |
| `run_clusters` | true | group pixels into features with bedtools (needed by steps 4 and 5) |

### Step 3 – whole-RNA structure

| parameter | default | description |
|---|---|---|
| `diag_zero_width` | 10 | candidate pairs with \|i − j\| below this are removed |
| `max_pair_distance` | none | candidate pairs with \|i − j\| at or above this are removed |
| `score_clip_percentile` | 99 | interaction support clipped at this percentile (also used by steps 4/5) |
| `score_mask` | false | drop the support of weak pairs (see the next two parameters) |
| `score_mask_pvalue` | 0.75 | … pairs with P below this … |
| `score_mask_percentile` | 98 | … and support below this percentile of the non-zero support |
| `unit_score_percentile` | 96 | stems scoring below this percentile are dropped (also steps 4/5) |
| `domain_split` | true | fold in domains instead of the whole RNA |
| `domain_max_len` | 600 | maximum domain length (nt) |
| `domain_edge_len` | 100 | a terminal piece shorter than this is merged into its neighbouring domain |
| `domain_gap_len` | 200 | a gap shorter than this between two large domains is merged into the preceding domain instead of becoming its own domain |
| `constraint_require_stack` | true | keep only constraint pairs that stack on a neighbouring pair |

### Step 4 – per-feature structures

| parameter | default | description |
|---|---|---|
| `cluster_max_distance` | 3000 | features spanning more than this (`end2 − start1`, nt) are long-range; also the short-/long-range boundary of steps 5 and 6 |
| `cluster_diagonal_gap` | 10 | a short-range feature whose arms are closer than this is diagonal |
| `cofold_pad` | 2 | nt added to each RNAcofold fragment at the junction |
| `cofold_max_retry` | 5 | constraint relaxations before folding without enforcement |

### Structure attributes (steps 4–6)

| parameter | default | description |
|---|---|---|
| `metric_shannon_span` | -1 | entropy window (nt) of the pairing ambiguity of **short-range** structures: -1 = `cluster_max_distance`, 0 = whole RNA, >0 = that window. Long-range structures always use the whole RNA |

### Step 5 – alternative structures

| parameter | default | description |
|---|---|---|
| `alt_shannon_span` | 3000 | window (nt) of the entropy used to find high-entropy features |
| `alt_cluster_score_percentile` | 50 | off-diagonal features below this percentile of total signal are dropped |
| `alt_conflict_slop` | 10 | nt added to the arms before the overlap test |
| `alt_conflict_overlap_fraction` | 0.5 | arms overlapping by more than this fraction of either arm compete |

---

## Fixed values

| value | where |
|---|---|
| 10 bp matrix for support and entropy | steps 4 , 6 and 7 |
| 3,000 nt window for entropy, O/E band and background | step 7 |
| O/E normalisation of the interaction signal | step 7 |
| short-range / long-range thresholds by the space mean, global by the background median | steps 6 / 7 |
