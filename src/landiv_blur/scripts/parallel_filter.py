"""Script to calculate an entropy based land cover type heterogeneity.

Note
----

    - When used as a script, you must either provide a value for the
      `diameter`, or for the `sigma` parameter.
      We truncate the Gaussian kernel after a certain distance from the center.
      The `diameter` is given by twice this truncation distance.
      In practical terms, the truncation distance, `truncate` must be expressed
      in units of the scale parameter, `sigma`, of the Gaussian kernel,
      therefore we have the relation:

          diameter = 2 * truncate * sigma

      As a consequence, two out of the three parameters, (`sigma`, `truncate`,
      `diameter`), need to be provided.
      If all 3 are provided, then `diameter` and `sigma` take precedence,
      meaning that the value provided for `truncate` is overwritten by
      `0.5 * \frac{diameter}{sigma}`
      If `truncate` is not provided, then the default value of `3` is used.

    - The entropy calculation is based on maps with data type float64 that
      resulted from applying a Gaussian smoothing filter. However, all the
      output maps produced by this script (i.e. the smoothened land-cover type
      maps, as well as, the entropy map are converted and then stored as uint8
      maps. The smoothened land-cover type maps contain, by construction only
      values in the range of [0, 1] and the entropy map is normed by the
      maximal possible entropy value (i.e. each land-cover type being equally
      present in a pixel) before rescaled to the uint8-range.

"""
from __future__ import annotations
import warnings
import multiprocessing as mproc
import rasterio as rio
import numpy as np
import os

from copy import copy
from argparse import ArgumentParser, RawDescriptionHelpFormatter


from landiv_blur import helper as lbhelp
from landiv_blur import prepare as lbprep
from landiv_blur.filters import gaussian as lbf_gauss
from landiv_blur.parallel import (
    combine_blurred_categories,
    combine_entropy_blocks,
    block_heterogeneity
)
from landiv_blur.io import compress_tif

# TODO: discuss what we want here

def get_lct_heterogeneity(source: str,
                          output_file: str,
                          scale: float,
                          block_size: tuple[int, int],
                          blur_params: dict,
                          categories: list | None = None,
                          blur_output_dtype: type|str = "uint8",
                          entropy_as_ubyte: bool = True,
                          **params):
    """Compute the entropy-based heterogeneity from a map of land cover types.

    Parameters
    ----------
    source : str
        Path to the land cover type tif file
    output_file : str
        Path to where the heterogeneity tif should be saved
    scale : float
        Size of a single pixel in the same units as `diameter` and `sigma`
    categories: list
        Specify which of the land-cover types to use as categories.
        If not provided then all the land-cover types are used.
    block_size: tuple of int
        Size (width, height) in #pixel of the block that a single job processes
    blur_params : dict
        Parameters for the Gaussian blur. It must contain at least either
        `diameter` or `sigma` in a in meters or any other measure of distance.
    blur_output_dtype:
      Set the data type of the blurred categories before computing the entropy.
      For the available options checkout `np.dtype`.
      Default is `"unit8"`
    entropy_as_ubyte:
        Should the entropy be normalized and returned as ubyte?
    """
    # handle deprecated parameters
    blur_as_int = params.pop('blur_as_int', None)
    if blur_as_int is not None:
        if blur_as_int:
            blur_output_dtype = "uint8"
        else:
            blur_output_dtype = "float64"
        warnings.warn("The parameter `blur_as_int` is deprecated, use "
                      f"`blur_output_dtype` instead!\nUsing {blur_as_int=} leads to "
                      f"{blur_output_dtype=}",
                      category=DeprecationWarning)
    # ###
    # prepare the input for the blocks
    # ###
    # read the metadata from source tif
    with rio.open(source) as dataset:
        dataset = rio.open(source)
        width = dataset.width
        height = dataset.height
        profile = copy(dataset.profile)
    print("The chosen source tif has a dimension of:"
          f"\n\t{width=}\n\t{height=}")

    blur_params=lbprep.get_blur_params(**blur_params)
    # get the diffusion kernel size in pixels
    psigma = blur_params['sigma'] / scale  # get sigma in pixels
    pdiameter = blur_params['diameter'] / scale  # get sigma in pixels
    truncate = blur_params.get('truncate')
    print("Chosen parameters in distance units and corresponting pixels):\n"
          f"\t- sigma: {blur_params['sigma']} => {psigma} pixels\n"
          f"\t- diameter: {blur_params['diameter']} => {pdiameter} pixels\n"
          f"\t- truncate: {blur_params['truncate']} (in sigmas)")
    # the border size of a block should be at least as large as the kernel size
    # TODO: this should be a computed term, rather than simply set
    # set the block size in pixels
    view_size = block_size
    print(f"The block size without border is {view_size=} pixels")
    border = lbf_gauss.compatible_border_size(sigma=psigma, truncate=truncate)
    print(f"The resulting border size is {border=} pixels")
    # TODO:
    # set the filename of the output file
    blur_output_file = lbhelp.output_filename(
        base_name=output_file,
        out_type='blurred',
        blur_params=blur_params
    )

    # now let's prepare the output parameters:
    if categories is None:
        print(f"WARNING:\nYou are using {categories=}")
        print("This will consider all unique values as categories")
        count = None
    else:
        count = len(categories)
    blur_output_params = dict(
        profile=profile,
        count=count,
        dtype=blur_output_dtype,
        output_file=blur_output_file,
    )
    entropy_output_file = lbhelp.output_filename(
        base_name=output_file,
        out_type='entropy',
        blur_params=blur_params
    )

    if entropy_as_ubyte:
        entropy_output_dtype = "uint8"
    else:
        entropy_output_dtype = "float64"
    entropy_output_params = dict(
        blur_params=blur_params,
        profile=profile,
        output_dtype=entropy_output_dtype,
        output_file=entropy_output_file,
    )
    views, inner_views = lbprep.create_views(view_size=view_size,
                                             border=border,
                                             size=(width, height))
    
    block_params = []
    # The parameter for the filter we want to apply:
    filter_params = dict(
        sigma=psigma,
        truncate=blur_params['truncate'],
    )
    filter_output_range = (0.0, 1.0)  # for gaussian filter
    for view, inner_view in zip(views, inner_views):
        bparams = dict(source=source,
                       categories=categories,
                       view=view,
                       inner_view=inner_view,
                       img_filter=lbf_gauss.gaussian,
                       filter_params=filter_params,
                       entropy_as_ubyte=entropy_as_ubyte,
                       blur_output_dtype=blur_output_dtype,
                       filter_output_range=filter_output_range,
                       )
        block_params.append(bparams)

    # ###
    # prepare multiprocessing
    # ###
    manager = mproc.Manager()
    entropy_q = manager.Queue()
    blur_q = manager.Queue()
    # get number of workers
    nbr_workers = lbhelp.get_nbr_workers(params.pop('nbrcpu', None))
    print(f"using {nbr_workers=}")
    pool = mproc.Pool(nbr_workers)


    # start the blurred category writer task
    blur_combiner = pool.apply_async(combine_blurred_categories,
                                     (blur_output_params, blur_q, ))
    # start the entropy writer task
    entropy_combiner = pool.apply_async(combine_entropy_blocks,
                                        (entropy_output_params, entropy_q, ))

    # start the block processing
    all_jobs = []
    for bparams in block_params:
        all_jobs.append(pool.apply_async(block_heterogeneity,
                                         (bparams, entropy_q, blur_q)))
    # collect results
    job_timers = []
    for job in all_jobs:
        # await for the jobs to return (i.e. complete) by calling .get
        # get the duration from the timer object that is returned by .get()
        job_timers.append(job.get().get_duration())

    # once we have all the blocks, add a last element to the queue to stop
    # the combination process
    entropy_q.put(dict(signal='kill'))
    blur_q.put(dict(signal='kill'))
    pool.close()
    # wait for the *_combiner tasks to finish
    pool.join()

    # lzw-compress final output
    compress = params.pop('compress', False)
    if compress:
        compress_tif(blur_output_file)
        compress_tif(entropy_output_file)
        # delete uncompressed files
        os.remove(blur_output_file)
        os.remove(entropy_output_file)
        print("Files compressed successfully")

    total_duration = max(entropy_combiner.get().get_duration(),
                         blur_combiner.get().get_duration())
    print(f"{total_duration=}")
    print(f"maximal duration of single job: {max(job_timers)=}")


def main():
    ap = ArgumentParser(prog='landiv',
                        usage=__doc__,
                        add_help=True,
                        formatter_class=RawDescriptionHelpFormatter)

    ap.add_argument("--source", type=str,
                    help='Path to the land cover type tif file')
    ap.add_argument("--scale", type=float,
                    help='Size of a single pixel in the same units as '
                    '`diameter` and `sigma`')
    ap.add_argument("--diameter", default=None, type=float,
                    help='The diameter of the Gaussian blur')
    ap.add_argument("--sigma", default=None, type=float,
                    help='Scale parameter (squared-root of the '
                    'variance) of the Gaussian diffusion kernel.')
    ap.add_argument("--truncate", default=3.0, type=float,
                    help='Factor to express the truncation limit in '
                    'units of `sigma`')
    ap.add_argument("--output", default='./heterogeneity.tif', type=str,
                    help='Path of the output file (will be overwritten)')
    ap.add_argument("--entropy_ubyte", default=False, type=bool,
                    help='Set if the resulting heterogeneity map should be in '
                         'ubyte (i.e. 0-255 or as float)')
    ap.add_argument("--blur_int", default=False, type=bool,
                    help='Set if the resulting heterogeneity map should be in '
                         'uint8 (i.e. 0-255 or as float)')
    ap.add_argument("--nbrcpu", default=2, type=int,
                    help='Set the number of cpus the script considers')
    ap.add_argument("--bwidth", default=1000, type=int,
                    help='The width of a block in pixels to be '
                    'processed in a single job')
    ap.add_argument("--bheight", default=1000, type=int,
                    help='The height of a block in pixels to be '
                    'processed in a single job')
    ap.add_argument("--nbrlct", default=12, type=int,
                    help='Set the number of land-cover types to consider')
    ap.add_argument("--compress", default=False, type=bool,
                    help='LZW compress output files inplace')

    inargs = vars(ap.parse_args())
    print(inargs)

    source = inargs.pop('source')
    output_file = inargs.pop('output')
    scale = inargs.pop('scale')
    diameter = inargs.pop('diameter')
    sigma = inargs.pop('sigma')
    truncate = inargs.pop('truncate')
    entropy_ubyte = inargs.pop('entropy_ubyte')
    blur_int = inargs.pop('blur_int')
    if blur_int:
        blur_output_dtype = "uint8"
    else:
        blur_output_dtype = "float64"
    nbrcpu = inargs.pop('nbrcpu')
    bwidth = inargs.pop('bwidth')
    bheight = inargs.pop('bheight')
    nbr_lct = inargs.pop('nbrlct')
    compr = inargs.pop('compress')
    # construct the list of land-cover types to use
    categories=list(range(nbr_lct))

    get_lct_heterogeneity(
        source=source,
        scale=scale,
        block_size=(bwidth, bheight),
        categories=categories,
        blur_params=dict(diameter=diameter, sigma=sigma, truncate=truncate),
        output_file=output_file,
        entropy_as_ubyte=entropy_ubyte,
        blur_output_dtype=blur_output_dtype,
        nbrcpu=nbrcpu,
        compress=compr
    )


if __name__ == "__main__":
    main()
