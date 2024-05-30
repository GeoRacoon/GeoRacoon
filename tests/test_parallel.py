from functools import partial

from time import sleep

import numpy as np
import multiprocessing as mproc
import rasterio as rio

from rasterio.plot import show as rioshow

from landiv_blur import io as lbio
from landiv_blur import processing as lbproc
from landiv_blur import prepare as lbprep
from landiv_blur.filters import gaussian as lbf_gauss

from landiv_blur import parallel as lbpara
from landiv_blur.filters import gaussian as lbf_gauss

from .config import ALL_MAPS

from matplotlib import pyplot as plt

@ALL_MAPS
def test_blur_recombination(datafiles):
    """Assert recombined blur of a layer is identical to processing entire map
    """
    blur_full = str(datafiles / 'blur_full.tif')
    blur_partial = str(datafiles / 'blur_partial.tif')
    diameter = 1000 # this is in meter
    scale = 100  # meter per pixel
    _diameter = diameter/scale
    truncate = 3  # property of the gaussian filter
    view_size = (500, 400)
    blur_params = lbprep.get_blur_params(diameter=_diameter, truncate=truncate)
    min_border = lbf_gauss.compatible_border_size(sigma=blur_params['sigma'],
                                              truncate=truncate)
    border = (100, 100)
    print(f"{border=}, {min_border=}")
    # load the data
    ch_map_tif = list(datafiles.iterdir())[0]
    ch_map = lbio.load_map(ch_map_tif, indexes=1)
    ch_data = ch_map['data']
    profile = ch_map['orig_profile']
    width = profile['width']
    height = profile['height']
    # we will save each layer separately
    profile['count'] = 1
    profile['dtype'] = rio.uint8
    # get the layers
    layers = lbproc.get_lct(ch_data)
    print(f"{layers=}")
    # we partially evaluate the guassian filter to make sure it gets
    # identical parameter everywhere
    # we do not need to pass the diameter to the filter function
    _ = blur_params.pop('diameter')
    img_filter = partial(lbf_gauss.gaussian, **blur_params)
    # we perform the test for each layer
    for layer in layers:
        index = layer + 1
        print(f"\t{layer=}")
        # perform the blur in a single run
        blurred_data = lbproc.get_layer_data(ch_data,
                                             layer=layer,
                                             img_filter=img_filter,
                                             output_dtype=np.uint8
                                             )
        # use multiprocessing and blur block by block
        # first set the parameters for the recombintion task
        blur_output_file = lbprep.output_filename(
            base_name=blur_partial,
            out_type=f"blur_lct_{layer}",
            blur_params=blur_params
        )
        blur_output_params = dict(
            profile=profile,
            as_int=True,
            output_file=blur_output_file,
            count=len(layers)
        )
        # now the parameter for the per block blur tasks
        views, inner_views = lbprep.create_views(view_size=view_size,
                                                 border=border,
                                                 size=(width, height))
        block_params = []
        for view, inner_view in zip(views, inner_views):
            bparams = dict(source=ch_map_tif,
                        layers=[layer],
                        view=view,
                        inner_view=inner_view,
                        img_filter=img_filter,
                        # entropy_as_ubyte=True,
                        blur_as_int=True,)
            block_params.append(bparams)
        manager = mproc.Manager()
        blur_q = manager.Queue()
        # get number of cpu's
        nbr_cpus = mproc.cpu_count() - 1
        print(f"using {nbr_cpus=}")
        pool = mproc.Pool(nbr_cpus)
        # start the blurred layer writer task
        blur_combiner = pool.apply_async(
            lbpara.combine_blurred_land_cover_types,
            (blur_output_params, blur_q, )
        )
        # start the block processing
        all_jobs = []
        for bparams in block_params:
            all_jobs.append(pool.apply_async(
                lbpara.runner_call,
                (blur_q,
                 lbproc.view_blurred,
                 bparams)
            ))
        # now lets wait for all of these jobs to finish
        job_timers = []
        for job in all_jobs:
            # await for the jobs to return (i.e. complete) by calling .get
            # get the duration from the timer object that is returned by .get()
            job_timers.append(job.get())
        # send the final kill job to the queue
        blur_q.put(dict(signal='kill'))
        # wait for the recombination job to terminate
        duration = blur_combiner.get().get_duration()
        # free up the resources
        pool.close()
        pool.join()
        print(f"job took {duration} seconds")

        # now we can read out the tif with the blurred layer anc compare
        blurred_layer_map = lbio.load_map(blur_output_file, indexes=index)
        blurred_layer_data = blurred_layer_map['data']
        print(f"{blurred_layer_data.shape=}")

        print(f"{blurred_data.shape=}")
        print(f"{blurred_data.dtype=}")
        print(f"{blurred_layer_data.shape=}")
        print(f"{blurred_layer_data.dtype=}")
        print(f"{blurred_layer_data.shape=}")
                
        # plt.imshow(blurred_data)
        # plt.savefig(f'{datafiles}/{layer}_single.png')
        # plt.imshow(blurred_layer_data)
        # plt.savefig(f'{datafiles}/{layer}_recombined.png')
        # plt.imshow(blurred_layer_data - blurred_data)
        # plt.savefig(f'{datafiles}/{layer}_diff.png')
        print(f"{datafiles=}")
        np.testing.assert_array_equal(
           blurred_data,
           blurred_layer_data,
           f'For {layer=} the recombined blurred map is different!'
        )


@ALL_MAPS
def test_entropy_recombination(datafiles):
    """Assert recombined entropy map is identical to processing the entire map
    """
    blur_full = str(datafiles / 'blur_full.tif')
    blur_partial = str(datafiles / 'blur_partial.tif')
    diameter = 1000 # this is in meter
    scale = 100  # meter per pixel
    _diameter = diameter/scale
    truncate = 3  # property of the gaussian filter
    view_size = (500, 400)
    blur_params = lbprep.get_blur_params(diameter=_diameter, truncate=truncate)
    min_border = lbf_gauss.compatible_border_size(sigma=blur_params['sigma'],
                                              truncate=truncate)
    border = (50, 50)
    print(f"{border=}, {min_border=}")
    # load the data
    ch_map_tif = list(datafiles.iterdir())[0]
    ch_map = lbio.load_map(ch_map_tif, indexes=1)
    ch_data = ch_map['data']
    profile = ch_map['orig_profile']
    width = profile['width']
    height = profile['height']
    # we will save each layer separately
    profile['count'] = 1
    profile['dtype'] = np.uint8
    # get the layers
    layers = lbproc.get_lct(ch_data)
    print(f"{layers=}")
    # we partially evaluate the guassian filter to make sure it gets
    # identical parameter everywhere
    # we do not need to pass the diameter to the filter function
    _ = blur_params.pop('diameter')
    img_filter = partial(lbf_gauss.gaussian, **blur_params)
    entropy_data = lbproc.get_entropy(ch_data, layers, normed=True,
                                      img_filter=img_filter,
                                      output_dtype=np.uint8)
    # use multiprocessing and blur block by block
    # first set the parameters for the recombintion task
    entropy_output_file = lbprep.output_filename(
        base_name=blur_partial,
        out_type=f"entropy_lct",
        blur_params=blur_params
    )
    entropy_output_params = dict(
        profile=profile,
        as_ubyte=True,
        output_file=entropy_output_file,
        count=len(layers)
    )
    # now the parameter for the per block blur tasks
    views, inner_views = lbprep.create_views(view_size=view_size,
                                                border=border,
                                                size=(width, height))
    block_params = []
    for view, inner_view in zip(views, inner_views):
        bparams = dict(source=ch_map_tif,
                    layers=layers,
                    view=view,
                    inner_view=inner_view,
                    img_filter=img_filter,
                    entropy_as_ubyte=True,
                    blur_as_int=True,)
        block_params.append(bparams)
    manager = mproc.Manager()
    entropy_q = manager.Queue()
    # get number of cpu's
    nbr_cpus = mproc.cpu_count() - 1
    print(f"using {nbr_cpus=}")
    pool = mproc.Pool(nbr_cpus)
    # start the blurred layer writer task
    blur_combiner = pool.apply_async(
        lbpara.combine_entropy_blocks,
        (entropy_output_params, entropy_q, )
    )
    # start the block processing
    all_jobs = []
    for bparams in block_params:
        all_jobs.append(pool.apply_async(
            lbpara.runner_call,
            (entropy_q,
                lbproc.get_entropy_view,
                bparams)
        ))
    # now lets wait for all of these jobs to finish
    job_timers = []
    for job in all_jobs:
        # await for the jobs to return (i.e. complete) by calling .get
        # get the duration from the timer object that is returned by .get()
        job_timers.append(job.get())
    # send the final kill job to the queue
    entropy_q.put(dict(signal='kill'))
    # wait for the recombination job to terminate
    duration = blur_combiner.get().get_duration()
    # free up the resources
    pool.close()
    pool.join()
    print(f"job took {duration} seconds")

    # now we can read out the tif with the blurred layer anc compare
    entropy_recomb_map = lbio.load_map(entropy_output_file, indexes=1)
    entropy_recomb_data = entropy_recomb_map['data']
    print(f"{entropy_data.shape=}")
    print(f"{entropy_recomb_data.shape=}")
    print(f"{entropy_data.dtype=}")
    print(f"{entropy_recomb_data.dtype=}")
            
    # plt.imshow(entropy_data)
    # plt.savefig(f'{datafiles}/entropy_single.png')
    # plt.imshow(entropy_recomb_data)
    # plt.savefig(f'{datafiles}/entropy_recombined.png')
    # plt.imshow(entropy_recomb_data - entropy_data)
    # plt.savefig(f'{datafiles}/entropy_diff.png')
    print(f"{datafiles=}")
    np.testing.assert_array_equal(
        entropy_data,
        entropy_recomb_data,
        f'The recombined entropy map is different!'
    )
