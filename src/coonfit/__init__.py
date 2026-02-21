"""
coonfit — Parallel multiple linear regression for spatial raster data.

This package implements a block-parallel multiple linear regression workflow
where both predictors and the response variable are provided as raster bands
(GeoTIFF files). Regression weights are estimated from valid pixels across the
full spatial extent and can be used to produce spatially continuous model
prediction maps.

Submodules
----------
inference
    Core functions for constructing the predictor matrix ``X``, computing
    ``X.T @ X``, solving the normal equations, and extracting response vectors
    from raster data. Supports optional intercept terms, spatial windowing, and
    boolean pixel selectors.
parallel
    High-level parallelized workflows for computing regression weights
    (``compute_weights``), generating model prediction rasters
    (``compute_model``), and evaluating model quality via RMSE and R².
parallel_helpers
    Internal worker functions used by :mod:`~coonfit.parallel` for
    block-wise computation of partial matrix products, beta coefficients,
    residual sums, and predictor consistency checks.
helper
    Utility functions for detecting rank-deficient predictor matrices and
    counting usable pixels within a boolean selector mask.
exceptions
    Custom exceptions raised during predictor validation and inference
    (e.g. :exc:`~coonfit.exceptions.InvalidPredictorError`,
    :exc:`~coonfit.exceptions.InferenceError`).
"""

_answer_to_everything = 42

