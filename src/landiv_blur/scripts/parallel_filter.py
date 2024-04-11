"""Script to calculate an entropy based land cover type heterogeneity.

Note
----

    When used as a script, you must either provide a value for the `diameter`,
    or for the `sigma` parameter.
    We truncate the Gaussian kernel after a certain distance from the center.
    The `diameter` is given by twice this truncation distance.
    In practical terms, the truncation distance, `truncate` must be expressed
    in units of the scale parameter, `sigma`, of the Gaussian kernel, therefore
    we have the relation:

        diameter = 2 * truncate * sigma

    As a consequence, two out of the three parameters, (`sigma`, `truncate`,
    `diameter`), need to be provided.
    If all 3 are provided, then `diameter` and `sigma` take precedence, meaning
    that the value provided for `truncate` is overwritten by
    `0.5 * \frac{diameter}{sigma}`
    If `truncate` is not provided, then the default value of `3` is used.

"""
import numpy as np
import multiprocessing as mproc
import rasterio as rio

from copy import copy
from time import perf_counter
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from skimage.util import img_as_ubyte
from rasterio.windows import Window


from landiv_blur import io as lbio
from landiv_blur import prepare as lbprep
from landiv_blur import processing as lbproc
from landiv_blur.filters import gaussian as lbf_gauss


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


def block_heterogeneity(params, q):
    """Block entropy-based landscape type heterogeneity measure
    """
    with TimedTask() as timer:
        view = params.pop('view')
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
        entropy_layer = lbproc.get_entropy(data=data,
                                           layers=copy(params.pop('layers')),
                                           # since the max is per block
                                           normed=False,
                                           img_filter=lbf_gauss.gaussian,
                                           sigma=params.pop('sigma'),
                                           truncate=params.pop('truncate'))
        usable_block = np.copy(
            lbprep.get_view(entropy_layer,
                            lbprep.relative_view(view, inner_view))
        )
        # # write out result to partial file
        # output = dict(
        #     blurred_block=Path('tif with blurred block'),
        #     metadata=dict(
        #         view=(),  # (x, y, h, w)
        #         border_size=0,  # what to cut
        #         # ...
        #     ),
        # )
        # TODO: this will pickle the data, it might be better to temporally
        #       store it (see above)
        output = dict(data=usable_block,
                      inner_view=inner_view)
        q.put(output)
        # TODO: check if the pool is empty (i.e. this is the last task) and
        # write signal = 'kill' into the output
    return timer


def combine_blocks(output_params: dict, q):
    """Listen to queue (q) and write computed block to single file
    """
    with TimedTask() as timer:
        output_file = output_params.pop('output')
        profile = output_params.pop('profile')
        as_ubyte = output_params.pop('as_ubyte', False)
        # TODO: we might need to adapt dtypes! (dtype=rasterio.ubyte...)
        profile['dtype'] = rio.float64
        if as_ubyte:
            profile['dtype'] = rio.ubyte
        with rio.open(output_file, 'w', **profile) as dst:
            while True:
                # load the q
                output = q.get()
                signal = output.get('signal', None)
                if signal:
                    if signal == "kill":
                        break
                data = output.pop('data')
                if as_ubyte:
                    # convert to ubyte
                    # TODO: This is an ugly hack to globally normalize
                    max_entropy = 2  # np.nanmax(data)
                    #       and to put in range [-1, 1] before conversion
                    data = 2 * data / max_entropy - 1
                    data = img_as_ubyte(data)
                    print(data)
                inner_view = output.pop('inner_view')
                # load block tif
                # get the relevant block (i.e. remove borders)
                # write to output file
                start = (inner_view[0], inner_view[1])
                size = (inner_view[2], inner_view[3])
                dst.write(data, window=Window(*start, *size), indexes=1)
                # lbio.export_to_tif(destination=output_file, data=data,
                #                    start=start, orig_profile=profile)
                # delete partial block tif
                timer.new_lab()
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


def get_lct_heterogeneity(source: str, output: str, scale: float, block_size: int,
                          layers: list, blur_params: dict, **params):
    """Compute the entropy-based heterogeneity from a map of land cover types.

    Parameters
    ----------
    source : str
        Path to the land cover type tif file
    output : str
        Path to where the heterogeneity tif should be saved
    scale : float
        Size of a single pixel in the same units as `diameter` and `sigma`
    block_size: int
        Size in # pixel of the square that a single job should process
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
    bsize = (block_size, block_size)
    print(f"The block size without border is {bsize=} pixels")
    border = (ksize+5, ksize+5)
    print(f"The resulting border size is {border=} pixels")
    # TODO: we truncate the map size here to a multiple of the block sizes
    _width = width - width % bsize[0]
    _height = height - height % bsize[1]
    # the resulting tif should have these dimensions, thus:
    profile['width'] = _width
    profile['height'] = _height

    # now let's prepare the output parameters:
    output_params = dict(
        output=output,
        profile=profile,
        as_ubyte=params.pop('as_ubyte', False)  # allow conversion to ubyte
    )
    nbr_views = (int(_width / bsize[0]), int(_height / bsize[1]))
    # compute the set of views
    views, inner_views = lbprep.create_views(nbr_views=nbr_views,
                                             border=border,
                                             size=(_width, _height))
    block_params = []
    for view, inner_view in zip(views, inner_views):
        bparams = dict(source=source,
                      layers=layers,
                      view=view,
                      inner_view=inner_view,
                      sigma=psigma,
                      truncate=truncate)
        block_params.append(bparams)

    # ###
    # prepare multiprocessing
    # ###
    manager = mproc.Manager()
    q = manager.Queue()
    # get number of cpu's
    nbr_cpus = params.pop('nbrcpu', mproc.cpu_count())
    print(f"using {nbr_cpus=}")
    pool = mproc.Pool(nbr_cpus)

    # start the writer task
    combiner = pool.apply_async(combine_blocks, (output_params, q, ))

    # start the block processing
    all_jobs = []
    for bparams in block_params:
        all_jobs.append(pool.apply_async(block_heterogeneity, (bparams, q)))
    # collect results
    job_timers = []
    for job in all_jobs:
        # await for the jobs to return (i.e. complete) by calling .get
        # get the duration from the timer object that is returned by .get()
        job_timers.append(job.get().get_duration())

    # once we have all the blocks, add a last element to the queue to stop
    # the combination process
    q.put(dict(signal='kill'))
    pool.close()
    # wait for the combiner task to finish
    pool.join()
    total_duration = combiner.get().get_duration()
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
    ap.add_argument("--ubyte", default=False, type=bool,
                    help='Set if the resulting heterogeneity map should be in '
                         'ubyte (i.e. 0-255 or as float)')
    ap.add_argument("--nbrcpu", default=2, type=int,
                    help='Set the number of cpus the script considers')
    ap.add_argument("--blocksize", default=1000, type=int,
                    help='The size of the block (i.e. square) in pixels to be '
                    'processed in a single job')

    # TODO: allow to select the layers (comma separated list)
    layers = list(range(8))

    inargs = vars(ap.parse_args())
    print(inargs)
    source = inargs.pop('source')
    output = inargs.pop('output')
    scale = inargs.pop('scale')
    diameter = inargs.pop('diameter')
    sigma = inargs.pop('sigma')
    truncate = inargs.pop('truncate')
    ubyte = inargs.pop('ubyte')
    nbrcpu = inargs.pop('nbrcpu')
    bsize = inargs.pop('blocksize')

    get_lct_heterogeneity(
        source=source,
        scale=scale,
        block_size=bsize,
        layers=layers,
        blur_params=get_blur_params(diameter, sigma, truncate),
        output=output,
        as_ubyte=ubyte,
        nbrcpu=nbrcpu,
    )
