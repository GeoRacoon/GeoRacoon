import numpy as np

from riogrande.io import Source

import linfit.parallel_helpers as lfph

from .conftest import (
    ALL_MAPS,
    get_file
)


@ALL_MAPS
def test_block_ssr_and_sst(datafiles, create_blurred_tif):
    """Test the parallel helpers _block_ssr and _block_sst using a raster band."""
    ndvi_map = Source(path=get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles))
    response_band = ndvi_map.get_bands()[0]

    # Use the same band as model for simplicity
    model_band = response_band

    # Load a small block to keep test fast
    view = (0, 0, 50, 50)
    data = response_band.load_block(view=view)['data']
    selector = ~np.isnan(data)

    # Test SSR
    ssr_parts = []
    lfph._block_ssr({'response': response_band,
                     'model': model_band,
                     'selector': selector,
                     'view': view}, ssr_parts)
    assert len(ssr_parts) == 1
    ssr_value, count = ssr_parts[0]
    assert ssr_value >= 0
    assert count == np.count_nonzero(selector)

    # Test SST
    sst_parts = []
    y_mean = np.nanmean(data)
    lfph._block_sst({'response': response_band,
                     'y_mean': y_mean,
                     'selector': selector,
                     'view': view}, sst_parts)
    assert len(sst_parts) == 1
    sst_value, count = sst_parts[0]
    assert sst_value >= 0
    assert count == np.count_nonzero(selector)


@ALL_MAPS
def test_process_band_count_valid(datafiles):
    """Test _process_band_count_valid using a raster band."""
    ndvi_map = Source(path=get_file(pattern="Switzerland_NDVI_*.tif", datafiles=datafiles))
    band = ndvi_map.get_band(bidx=1)
    data = band.get_data()
    selector = ~np.isnan(data)

    # With a minimum of pixels required for a fit (10) - returns True if so
    valid_counts, (timer,) = lfph._process_band_count_valid(band, selector, no_data=np.nan, limit_count=10)
    assert isinstance(valid_counts, dict)
    assert valid_counts[band] == True
    assert timer.get_duration() >= 0

    # Without limit count (no valid pixesl required
    valid_counts, (timer,) = lfph._process_band_count_valid(band, selector, no_data=np.nan, limit_count=0)
    assert isinstance(valid_counts, dict)
    assert valid_counts[band] == np.count_nonzero(selector)
    assert timer.get_duration() >= 0


@ALL_MAPS
def test_partial_optimal_betas(datafiles, create_blurred_tif):
    """Test _partial_optimal_betas using a simple runner simulation."""
    from multiprocessing import Manager

    output_q = Manager().list()
    response_band = Source(path=create_blurred_tif).get_bands()[0]

    # Minimal params for the test
    params = {'response': response_band}

    # Monkeypatch runner_call to simulate the parallel call
    import riogrande.parallel as rgpara
    orig_runner_call = rgpara.runner_call

    def dummy_runner_call(output_q, func, params, wrapper):
        # simulate a beta dict as output
        output_q.append(wrapper({'beta1': 1.0, 'beta2': 2.0}))

    rgpara.runner_call = dummy_runner_call
    lfph._partial_optimal_betas(params, output_q)
    rgpara.runner_call = orig_runner_call

    assert len(output_q) == 1
    assert 'X' in output_q[0]
    assert isinstance(output_q[0]['X'], list)
    assert output_q[0]['X'] == [1.0, 2.0]
