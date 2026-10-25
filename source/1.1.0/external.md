### Acceleration

* `--gpu` replaces `partition-smp`→`partition-cuda` and `Fold-smp`→`Fold-cuda`;
  `bipartition-smp` and `RNAcofold` have no CUDA build and stay on the CPU.
* `--cython` uses the OpenMP kernel `river._expand` for matrix expansion; if the
  extension was not compiled RIVER falls back to NumPy with identical results.
* `-t/--threads` sets `OMP_NUM_THREADS` for RNAstructure.