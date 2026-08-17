# Benchmarks

This directory contains the ASV (airspeed velocity) benchmark suite for
`GeoRacoon`. Benchmarks are intended for local or dedicated-machine runs, not
for shared CI runners where load and CPU throttling make timings noisy.

## Running benchmarks

Install the benchmark dependencies first:

```bash
pip install -e .[benchmarks]
```

Register or inspect the ASV machine name:

```bash
asv machine
```

Run a quick smoke check against the current environment:

```bash
asv run --python=same --quick
```

Run the full benchmark suite:

```bash
asv run
```

Publish and preview the HTML report:

```bash
asv publish
asv preview
```

Raw benchmark results are stored in `.asv/results/` and committed to the
repository. Generated environments and HTML output (`.asv/env/`, `.asv/html/`)
are not committed.

## Machine configuration

Benchmark sizing is configured in `machine_configs.json`. The file contains a
required `default` entry and optional machine-specific entries keyed by ASV
machine name:

```json
{
  "default": {
    "block_fraction_denominators": [1, 2, 3, 4],
    "sizes": [1000, 5000, 10000, 20000],
    "ndvi": "data/testing/ndvi/Switzerland_NDVI_binning_2015.tif",
    "landcover": "data/testing/landcover/Switzerland_CLC_2012_reclass8.tif",
    "gaussian_sigma": 10.0,
    "gaussian_truncate": 3.0
  }
}
```

`Machine` in `common.py` reads the ASV machine name from
`~/.asv-machine.json`, merges the matching machine entry over `default`, and
creates `get_<key>()` accessors for every config key. Missing machine-specific
keys fall back to `default`.

The `n_jobs` sweep is derived from the CPU count (`2 .. ncpu - 1`), not from
the JSON config. Block sizes are expressed as a fraction of the raster's total
size: `block_fraction = 1/d` for each configured denominator `d`, and the block
is `(round(f*width), round(f*height))`.

To add a machine:

1. Register or inspect the ASV machine name with `asv machine`.
2. Add a matching entry to `benchmarks/machine_configs.json`.
3. Run `asv run --python=same --quick` to verify the configuration.

## Benchmark structure

### `bench_filter.py`

Benchmarks the parallel border-preserving Gaussian filter
(`convster.parallel.apply_filter` with `convster.filters.bpgaussian`) on the
fixed Swiss NDVI raster (`TimeFilter`/`PeakMemFilter`, sweeping `n_jobs` x
`block_fraction`) and on synthetic rasters of increasing size
(`TimeFilterScaling`/`PeakMemFilterScaling`, full `size` x `n_jobs` x
`block_fraction` grid).

The border-preserving filter adds a halo whose size is fixed by the Gaussian
kernel (`sigma`/`truncate`). Block/fraction combinations where the block is not
larger than that border produce invalid, overlapping views and are skipped in
`setup`.

### `bench_linreg.py`

Benchmarks the full parallel multiple linear regression workflow
(`coonfit.parallel.compute_weights`: selector mask, `X.T @ X`, rank check,
inversion, and optimal weights) on the Swiss NDVI response with the landcover
map as predictor, coregistered onto the NDVI grid
(`TimeComputeWeights`/`PeakMemComputeWeights`) and on synthetic rasters
(`TimeComputeWeightsScaling`/`PeakMemComputeWeightsScaling`).

The landcover raster is coregistered to the NDVI grid with
`riogrande.io.coregister_raster` in `setup` (cached under the system temp dir),
so the reprojection is excluded from the timing and peak-memory measurements.

### Native baselines

Every routine also has a *native* (single-process, no multiprocessing)
counterpart, reported at `n_jobs = 1`:

- `*FilterNative*`: reads the full band, applies `bpgaussian` directly, and
  writes the output — no block division or worker pool.
- `*ComputeWeightsNative*`: builds the selector from the band masks and solves
  the exact normal equations `(X.T @ X)^-1 X.T @ y` on the full data.

The docs figures normalize every heatmap cell by the corresponding native value,
so `< 1` means faster / more memory efficient than the native approach and
`> 1` means slower / less memory efficient.

## Memory measurement

ASV's built-in `peakmem_` samples only the main process, but the parallel work
happens in `multiprocessing.Pool` workers. The `PeakMem*` benchmarks therefore
use `track_` benchmarks that sample the summed RSS of the whole process tree
(parent plus all worker/manager children) via `psutil` during the run
(`common.peak_rss_while`). Reported unit is `bytes`.

## Comparing branches

Use `asv continuous` for feature-branch comparisons:

```bash
asv continuous main HEAD
```

The published benchmark history should generally track the main branch.
