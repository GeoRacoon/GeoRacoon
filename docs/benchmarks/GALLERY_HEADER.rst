Benchmark gallery
=================

This gallery presents performance benchmarks for GeoRacoon's block-wise
parallel raster computations.

The multiple linear regression example explains how the normal equations can
be decomposed into additive contributions from independent spatial blocks. It
also illustrates the two practical tuning dimensions exposed by the
implementation: the spatial block size and the number of worker processes.

The heatmaps use the native, full-raster implementation as their reference.
This reference is shown as ``n_jobs = 1``:

* ``1.0`` means the same runtime or peak memory as the native implementation;
* values below ``1.0`` mean faster execution or lower memory use; and
* values above ``1.0`` mean slower execution or higher memory use.

The benchmark results are machine-dependent. They demonstrate the available
runtime--memory trade-off and should not be interpreted as universal
performance measurements.
