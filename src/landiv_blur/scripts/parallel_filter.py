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
import os

import numpy as np
import multiprocessing as mproc
import rasterio as rio

from copy import copy
from time import perf_counter
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from rasterio.windows import Window


from landiv_blur import io as lbio
from landiv_blur import prepare as lbprep
from landiv_blur import processing as lbproc
from landiv_blur.filters import gaussian as lbf_gauss
from landiv_blur.plotting import plot_entropy


class TimedTask:
    def __init__(self,):
        self.labs = []

    def __enter__(self):
        self.now = perf_counter()
        self.start = self.now
        self.stop = 0.0
        return self

    def get_duration(self, ):
        return self.stop - self.start

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.new_lab()
        self.stop = self.now
        self.dt = self.stop - self.start

    def new_lab(self,):
        self.now = perf_counter()
        try:
            self.labs.append(self.now - self.labs[-1])
        except IndexError:
            self.labs.append(self.now - self.start)


def block_heterogeneity(params, entropy_q, blur_q):
    """Block entropy-based landscape type heterogeneity measure

    Parameters
    ----------
    params: dict
      Key value pairs holding all relevant data for the single worker
    entropy_q: multiprocessing.Queue
      The queue to push the entropy maps through
    blur_q: multiprocessing.Queue
      The queue to push the multi-band blurred land-cover types maps through
    """
    with TimedTask() as timer:
        view = params.pop('view')
        entropy_as_ubyte = params.pop('entropy_as_ubyte', False)
        blur_as_int = params.pop('blur_as_int', False)
        if entropy_as_ubyte:
            normed = True
            dtype = np.uint8
        else:
            normed = False
            dtype = None
        inner_view = params.pop('inner_view')
        # read out block from original file
        start = (view[0], view[1])
        size = (view[2], view[3])
        result = lbio.load_block(
            source,
            start=start,
            size=size
        )
        data = result.pop('data')
        # transform = result.pop('transform')
        # orig_profile = result.pop('orig_profile')
        # perform blur
        layers = copy(params.pop('layers'))
        blur_layers = lbproc.get_filtered_layers(
            data=data,
            layers=layers,
            img_filter=lbf_gauss.gaussian,
            sigma=params.pop('sigma'),
            truncate=params.pop('truncate')
        )
        if blur_as_int:
            _maxint = np.iinfo(np.uint8).max
            for k, data in blur_layers.items():
                blur_layers[k] = blur_layers[k] * _maxint
        # prepare parameter to send to blur writer
        blur_output = dict(layer_data=blur_layers,
                           inner_view=inner_view,)
        blur_q.put(blur_output)
        # calculate entropy
        entropy_layer = lbproc.compute_entropy(
            filtered_data_layers=blur_layers,
            normed=normed,
            dtype=dtype,
        )
        usable_block = np.copy(
            lbprep.get_view(entropy_layer,
                            lbprep.relative_view(view, inner_view))
        )
        # # write out result to partial file
        # entropy_output = dict(
        #     blurred_block=Path('tif with blurred block'),
        #     metadata=dict(
        #         view=(),  # (x, y, h, w)
        #         border_size=0,  # what to cut
        #         # ...
        #     ),
        # )
        # TODO: this will pickle the data, it might be better to temporally
        #       store it (see above)
        entropy_output = dict(data=usable_block,
                              inner_view=inner_view)
        entropy_q.put(entropy_output)
        # TODO: check if the pool is empty (i.e. this is the last task) and
        # write signal = 'kill' into the output
        print(f"Processed block\n\t{view=}")
    return timer


def output_filename(base_name: str, out_type: str, blur_params: dict):
    """Construct the filename for the specific output type.

    Parameters
    ----------
    base_name: str
      The basic output name in the form <name>.tif
    out_type: str
      The type of output that will be saved.
      This should be either 'blur' or 'entropy' but any string is accepted
    blur_params: dict
      Output of `get_blur_params`, so 'sigma', 'truncate' and 'diameter'
      are expected keys.

    Returns
    str:
      The resulting filename of the form
      '<name>_<out_type>_sig_<{sigma}>_diam_<{diameter}>_trunc_<{truncate}>.tif'
    """
    _base_name, _ext = os.path.splitext(base_name)
    sig = blur_params['sigma']
    diam = blur_params['diameter']
    trunc = blur_params['truncate']
    _blur_string = f"sig_{sig}_diam_{diam}_trunc_{trunc}"
    return f"{_base_name}_{out_type}_{_blur_string}{_ext}"


def combine_blurred_land_cover_types(output_params: dict, blur_q):
    """Listen to queue (blur_q) and write blurred blocks to a single file
    """
    with TimedTask() as timer:
        output = output_params.pop('output')
        blur_params = output_params.pop('blur_params')
        as_int = output_params.pop('as_int', False)
        output_file = output_filename(base_name=output,
                                      out_type='lct_blurred',
                                      blur_params=blur_params)
        print(f"{output_file=}")
        print(f"{as_int=}")
        profile = output_params.pop('profile')
        profile['dtype'] = rio.float64
        if as_int:
            profile['dtype'] = rio.uint8
        # TODO: This should not be hard-coded but determined when probing the
        #       source data for the number of distinct land-cover types.
        profile['count'] = 12
        with rio.open(output_file, 'w', **profile) as dst:
            while True:
                output = blur_q.get()
                signal = output.get('signal', None)
                if signal:
                    if signal == "kill":
                        print(f"\n\nClosing: {output_file}\n\n")
                        break
                layer_data = output.pop('layer_data')
                inner_view = copy(output.pop('inner_view'))
                # load block tif
                # get the relevant block (i.e. remove borders)
                # write to output file
                w = Window(inner_view[0],
                           inner_view[1],
                           inner_view[2],
                           inner_view[3])
                for band, data in layer_data.items():
                    dst.write(data,
                              window=w, indexes=band+1)
                print(f"Wrote out bands for blurred block {inner_view=}")
                timer.new_lab()
    return timer


def combine_entropy_blocks(output_params: dict, entropy_q):
    """Listen to queue (entropy_q) and write computed block to single file
    """
    with TimedTask() as timer:
        output = output_params.pop('output')
        blur_params = output_params.pop('blur_params')
        output_file = output_filename(base_name=output,
                                      out_type='entropy',
                                      blur_params=blur_params)
        print(f"{output_file=}")
        as_ubyte = output_params.pop('as_ubyte')
        profile = output_params.pop('profile')
        profile['dtype'] = rio.float64
        if as_ubyte:
            profile['dtype'] = rio.uint8
        with rio.open(output_file, 'w', **profile) as dst:
            while True:
                # load the entropy_q
                output = entropy_q.get()
                signal = output.get('signal', None)
                if signal:
                    if signal == "kill":
                        print(f"\n\nClosing: {output_file}\n\n")
                        break
                data = output.pop('data')
                inner_view = copy(output.pop('inner_view'))
                # load block tif
                # get the relevant block (i.e. remove borders)
                # write to output file
                w = Window(inner_view[0],
                           inner_view[1],
                           inner_view[2],
                           inner_view[3])
                dst.write(data,
                          window=w, indexes=1)
                # lbio.export_to_tif(destination=output_file, data=data,
                #                    start=start, orig_profile=profile)
                # delete partial block tif
                print(f"Wrote out entropy block {inner_view=}")
                timer.new_lab()
    plot_entropy(output_file, start=(0, 0),
                 size=(profile['width'], profile['height']),
                 output=f"{output_file}.preview.png")
    return timer


def get_blur_params(diameter, sigma, truncate):
    """
    .. note::
        The default of truncate is 3

    """
    # use default value of 3 for truncate
    truncate = truncate or 3
    if diameter:
        if sigma:
            truncate = 0.5 * diameter / sigma
        else:
            if truncate:
                sigma = 0.5 * diameter / truncate
    else:
        if sigma:
            diameter = 2 * sigma * truncate
        else:
            # TODO: this test should be done when parsing the input arguments
            raise TypeError("Either the `diameter` or the `sigma` parameter "
                            " need to be provided.")
    return dict(diameter=diameter, sigma=sigma, truncate=truncate)


def get_lct_heterogeneity(source: str, output: str, scale: float,
                          block_size: int,
                          blur_params: dict,
                          layers: list = None,
                          **params):
    """Compute the entropy-based heterogeneity from a map of land cover types.

    Parameters
    ----------
    source : str
        Path to the land cover type tif file
    output : str
        Path to where the heterogeneity tif should be saved
    scale : float
        Size of a single pixel in the same units as `diameter` and `sigma`
    layers: list
        Specify which of the land-cover types to use as layers.
        If not provided then all the land-cover types are used.
    block_size: tuple of int
        Size (width, height) in #pixel of the block that a single job processes
    blur_params : dict
        Parameters for the Gaussian blur. It must contain at least either
        `diameter` or `sigma` in a in meters or any other measure of distance.
    """
    # ###
    # prepare the input for the blocks
    # ###
    # read the metadata from source tif
    dataset = rio.open(source)
    width = dataset.width
    height = dataset.height
    profile = copy(dataset.profile)
    print("The chosen source tif has a dimension of:"
          f"\n\t{width=}\n\t{height=}")

    # get the diffusion kernel size in pixels
    psigma = blur_params['sigma'] / scale  # get sigma in pixels
    pdiameter = blur_params['diameter'] / scale  # get sigma in pixels
    truncate = blur_params['truncate']
    # get the distance from center to border of the Gaussian kernel
    ksize = lbf_gauss.get_kernel_size(sigma=psigma,
                                      truncate=truncate)
    print("Chosen parameters in distance units and corresponting pixels):\n"
          f"\t- sigma: {blur_params['sigma']} => {psigma} pixels\n"
          f"\t- diameter: {blur_params['diameter']} => {pdiameter} pixels\n"
          f"\t- truncate: {blur_params['truncate']} (in sigmas)")
    # the border size of a block should be at least as large as the kernel size
    # TODO: this should be a computed term, rather than simply set
    # set the block size in pixels
    view_size = block_size
    print(f"The block size without border is {view_size=} pixels")
    border = (ksize, ksize)
    print(f"The resulting border size is {border=} pixels")
    # Should the entropy be normalized and returned as ubyte?
    entropy_as_ubyte = params.pop('entropy_as_ubyte', False)
    blur_as_int = params.pop('blur_as_int', False)

    # now let's prepare the output parameters:
    blur_output_params = dict(
        blur_params=blur_params,
        output=output,
        profile=profile,
        as_int=blur_as_int,
    )
    entropy_output_params = dict(
        blur_params=blur_params,
        output=output,
        profile=profile,
        as_ubyte=entropy_as_ubyte,
    )
    views, inner_views = lbprep.create_views(view_size=view_size,
                                             border=border,
                                             size=(width, height))
    block_params = []
    for view, inner_view in zip(views, inner_views):
        bparams = dict(source=source,
                       layers=layers,
                       view=view,
                       inner_view=inner_view,
                       sigma=psigma,
                       entropy_as_ubyte=entropy_as_ubyte,
                       blur_as_int=blur_as_int,
                       truncate=truncate)
        block_params.append(bparams)

    # ###
    # prepare multiprocessing
    # ###
    manager = mproc.Manager()
    entropy_q = manager.Queue()
    blur_q = manager.Queue()
    # get number of cpu's
    nbr_cpus = params.pop('nbrcpu', mproc.cpu_count())
    print(f"using {nbr_cpus=}")
    pool = mproc.Pool(nbr_cpus)

    # start the blurred layer writer task
    blur_combiner = pool.apply_async(combine_blurred_land_cover_types,
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
    total_duration = max(entropy_combiner.get().get_duration(),
                         blur_combiner.get().get_duration())
    print(f"{total_duration=}")
    print(f"maximal duration of single job: {max(job_timers)=}")


if __name__ == "__main__":
    ap = ArgumentParser(prog='lct-heterogeneity',
                        usage=__doc__,
                        description=__doc__,
                        add_help=False,
                        formatter_class=ArgumentDefaultsHelpFormatter)

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

    # TODO: allow to select the layers (comma separated list)
    layers = list(range(11))

    inargs = vars(ap.parse_args())
    print(inargs)
    source = inargs.pop('source')
    output = inargs.pop('output')
    scale = inargs.pop('scale')
    diameter = inargs.pop('diameter')
    sigma = inargs.pop('sigma')
    truncate = inargs.pop('truncate')
    entropy_ubyte = inargs.pop('entropy_ubyte')
    blur_int = inargs.pop('blur_int')
    nbrcpu = inargs.pop('nbrcpu')
    bwidth = inargs.pop('bwidth')
    bheight = inargs.pop('bheight')

    get_lct_heterogeneity(
        source=source,
        scale=scale,
        block_size=(bwidth, bheight),
        layers=layers,
        blur_params=get_blur_params(diameter, sigma, truncate),
        output=output,
        entropy_as_ubyte=entropy_ubyte,
        blur_as_int=blur_int,
        nbrcpu=nbrcpu,
    )
