# RIVER quickstart — preparation and example runs

---

## 1. Preparation

### 1.1 Environment

```bash
conda env create -f environment.yml
conda activate river
pip install .                         # builds the optional Cython/OpenMP kernel
# RIVER_NO_CYTHON=1 pip install .     # skip the kernel (NumPy fallback, identical results)
```

RNAstructure needs its thermodynamic tables:

```bash
export DATAPATH=$PATH_PREFIX/share/rnastructure/data_tables
```

Check the installation and the external programs before the first run:

```bash
river --version
for exe in partition-smp bipartition-smp ProbabilityPlot Fold-smp RNAcofold bedtools; do
    command -v $exe || echo "MISSING: $exe"
done
echo "DATAPATH=$DATAPATH"
```

### 1.2 Input files

The example files are under `data/`.

* **fasta** — one record per RNA; select it with `-s/--seq-id` if the file holds
  several. The sequence length is taken from the FASTA (`-l/--length` is only an
  optional consistency check and issues a warning on mismatch).
* **contact map** — an RNA–RNA contact map built from proximity-ligation data
  (RIC-seq, vRIC-seq, SPLASH, …) with Juicer tools. The `.hic` file must contain
  the resolutions required by the preset: 10/5/1 bp for `short` and `medium`,
  15/10/5/1 bp for `long`. `-c/--chrom` is the chromosome name **inside** the
  `.hic`.

### 1.3 Parameters

Three layers, each overriding the previous one:

```
preset (-r short|medium|long|auto)  →  --config params.yaml/.json  →  command-line options
```

```bash
river presets                 # print every preset
river presets long            # print one preset
```

Every value actually used is written to `step0_prepare/run_config.json`.

---

## 2. Example runs

### 2.0 RNA

| model system | length            | steps         |
| ------------ | ----------------- | ------------- |
| RPPH1        | 341 nt            | 3             |
| 25S/28S rRNA | 3,396 nt/5,070 nt | 3             |
| SARS-CoV-2   | 29,903 nt         | 3,4,5,6,7     |

### 2.1 Short RNA (RPPH1) — whole-molecule structure

```bash
river predict -f data/RPPH1.fasta -s NR_002312.1 \
    --hic data/RPPH1.hic -c NR_002312.1 \
    -r short -o results/RPPH1 -t 8
```

`--steps` defaults to `1,2` (partition and feature localization). Add step 3 to
get the whole-molecule structure:

```bash
river predict -f data/RPPH1.fasta -s NR_002312.1 \
    --hic data/RPPH1.hic -c NR_002312.1 \
    -r short -o results/RPPH1 -t 8 --steps 1,2,3
```

### 2.2 Medium RNA (28S rRNA) — domain folding, Cython kernel

```bash
river predict -f data/28S_rRNA.fasta -s NR_046235.3 \
    --hic data/28S_rRNA_merge_RICseq.hic -c HS1 \
    -r medium --cython -o results/28S -t 16 --steps 1,2,3
```

25S rRNA uses its own pair-distance limit:

```bash
river predict -f data/25S_rRNA.fasta --hic data/yeast_25SrRNA.hic -c HS1 -r medium \
    --config examples/params_25SrRNA.yaml -o results/25S --steps 1,2,3
```

### 2.3 Long RNA (SARS-CoV-2) — GPU, Cython

```bash
# default run: steps 1,2 only
river predict -f data/NC_045512_2_md.fasta -s NC_045512.2 \
    --hic data/SARSCoV2_VERO_RIC_merge.hic -c HS1 -r long \
    --gpu --cython -o results/SARS2 -t 16
```

### 2.4 All seven steps in one go

```bash
river predict -f data/NC_045512_2_md.fasta -s NC_045512.2 \
    --hic data/SARSCoV2_VERO_RIC_merge.hic -c HS1 -r long \
    -o results/SARS2 --steps 1,2,3,4,5,6,7 \
    --cluster-max-distance 3000 -t 16 --n-clusters 5
```

`--n-clusters` must be given explicitly whenever step 7 is requested; there is
no usable default on the command line.

### 2.5 Adding steps 4 and 5 to a finished run

Steps 4 and 5 only need the outputs of steps 1 and 2, so they can be added later
without redoing the expensive partition:

```bash
river predict -f data/NC_045512_2_md.fasta -s NC_045512.2 \
    --hic data/SARSCoV2_VERO_RIC_merge.hic -c HS1 -r long \
    -o results/SARS2 --steps 4,5 --resume -t 16
```

Re-running a single step after changing a parameter works the same way:

```bash
river predict ... -o results/SARS2 --steps 3 --resume --max-pair-distance 2000
river predict ... -o results/SARS2 --steps 4 --resume --cluster-max-distance 2500
river predict ... -o results/SARS2 --steps 5 --resume \
    --alt-shannon-span 2000 --alt-cluster-score-percentile 60
```

`--resume` reuses existing intermediate files; only the requested steps run.

### 2.6 Well-folded structures (steps 6 and 7)

The two screens work on structures that already exist.

| step | screen                  | reads                              | writes                              |
| ---- | ----------------------- | ---------------------------------- | ----------------------------------- |
| 6    | feature-derived (local) | `step4_feature_structure_predict`  | `step6_local_wellfolded_screen`     |
| 7    | unit-based (global)     | `step3_optimal_structure_predict`  | `step7_global_wellfolded_screen`    |

Step 6 stops with an error if the step 4 directory is absent, so request step 4
in the same call when resuming from a default (steps 1,2) run:

```bash
# local well-folded structures (needs step 4)
river predict -f data/NC_045512_2_md.fasta -s NC_045512.2 \
    --hic data/SARSCoV2_VERO_RIC_merge.hic -c HS1 -r long \
    -o results/SARS2 --steps 4,6 --resume -t 16
# The short/long split used by both screens comes from --cluster-max-distance and
# must match the value used by step 4.
```

```bash
# global well-folded structures (needs step 3 and an explicit --n-clusters)
river predict -f data/NC_045512_2_md.fasta -s NC_045512.2 \
    --hic data/SARSCoV2_VERO_RIC_merge.hic -c HS1 -r long \
    -o results/SARS2 --steps 3,7 --resume --n-clusters 5 -t 16
# --n-clusters is the number of Ward clusters of structural units used by step
# 7; every other screening threshold follows the preset and --config. 
```

Both screens in one call:

```bash
river predict -f data/NC_045512_2_md.fasta -s NC_045512.2 \
    --hic data/SARSCoV2_VERO_RIC_merge.hic -c HS1 -r long \
    -o results/SARS2 --steps 3,4,6,7 --resume --n-clusters 5 -t 16
```

### 2.7 Python API

```python
from river.cli import predict

predict("data/NC_045512_2_md.fasta", "data/SARSCoV2_VERO_RIC_merge.hic",
        "HS1", "results/SARS2", rna_type="long", seq_id="NC_045512.2",
        steps=(1, 2, 3, 4, 5, 6, 7), threads=16, use_cython=True,
        n_clusters=5, overrides={"cluster_max_distance": 3000})

# only the feature-level steps of a finished run
predict("data/NC_045512_2_md.fasta", "data/SARSCoV2_VERO_RIC_merge.hic",
        "HS1", "results/SARS2", rna_type="long", seq_id="NC_045512.2",
        steps=(4, 5), resume=True)

# add the two well-folded screens to a finished run
predict("data/NC_045512_2_md.fasta", "data/SARSCoV2_VERO_RIC_merge.hic",
        "HS1", "results/SARS2", rna_type="long", seq_id="NC_045512.2",
        steps=(6, 7), resume=True, n_clusters=5)
```

### 2.8 Output layout

```
results/SARS2/
├── river.log
├── step0_prepare/
├── step1_partition/
├── step2_structure_feature_locate/
├── step3_optimal_structure_predict/
├── step4_feature_structure_predict/
├── step5_alternative_structure_predict/
├── step6_local_wellfolded_screen/
└── step7_global_wellfolded_screen/
```

The complete file list of every step is in [`docs/steps.md`](docs/steps.md).
