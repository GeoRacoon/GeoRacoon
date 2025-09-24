"""
Add sth here
"""
from __future__ import annotations


def get_blur_params(diameter: float | None = None,
                    sigma: float | None = None,
                    truncate: float = 3) -> dict[str, float]:
    """
    Compute Gaussian blur parameters from either `diameter` or `sigma`.

    Either `diameter` or `sigma` must be provided. Missing values are inferred
    from the others, and `truncate` is used or recomputed to maintain consistency.

    Parameters
    ----------
    diameter
        Kernel diameter. If provided with `sigma`, `truncate` is recomputed.
    sigma
        Standard deviation of the Gaussian kernel. If provided with `diameter`,
        `truncate` is recomputed.
    truncate
        Number of standard deviations at which to truncate the kernel.
        Default is 3. Ignored if both `diameter` and `sigma` are provided
        (recomputed).

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - `diameter`: computed kernel diameter
        - `sigma`: computed standard deviation
        - `truncate`: final truncate value

    Raises
    ------
    TypeError
        If neither `diameter` nor `sigma` is provided.

    Notes
    -----
    The function ensures that `diameter`, `sigma`, and `truncate` are consistent
    according to Gaussian kernel conventions.

    Examples
    --------
    >>> get_blur_params(diameter=15)
    {'diameter': 15, 'sigma': 2.5, 'truncate': 3}
    >>> get_blur_params(sigma=2.0)
    {'diameter': 12.0, 'sigma': 2.0, 'truncate': 3}
    >>> get_blur_params(diameter=15, sigma=3)
    {'diameter': 15, 'sigma': 3, 'truncate': 2.5}
    """
    if diameter is None and sigma is None:
        raise TypeError("Either the `diameter` or the `sigma` parameter "
                        f"must be provided. \nGot: {diameter=}, {sigma=}")
    
    if diameter:
        if sigma:
            truncate = 0.5 * diameter / sigma
        else:
            if truncate:
                sigma = 0.5 * diameter / truncate
    else:
        if sigma:
            diameter = 2 * sigma * truncate

    return dict(diameter=diameter, sigma=sigma, truncate=truncate)