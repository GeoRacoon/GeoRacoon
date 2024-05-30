"""
This module contains various helper functions to parallelize the application
of filters on a tif

"""
from __future__ import annotations

from copy import copy

import rasterio as rio
import multiprocessing as mproc

from rasterio.windows import Window

from .timing import TimedTask
from .plotting import plot_entropy
from .processing import view_blurred, view_entropy



def combine_blurred_land_cover_types(output_params: dict, blur_q):
    """Listen to queue (blur_q) and write blurred blocks to a single file
    """
    with TimedTask() as timer:
        as_int = output_params.pop('as_int', False)
        output_file = output_params.pop('output_file')
        print(f"{output_file=}")
        print(f"{as_int=}")
        profile = output_params.pop('profile')
        profile['dtype'] = rio.float64
        if as_int:
            profile['dtype'] = rio.uint8
        profile['count'] = output_params.get('count', profile['count'])
        with rio.open(output_file, 'w', **profile) as dst:
            while True:
                output = blur_q.get()
                signal = output.get('signal', None)
                if signal:
                    if signal == "kill":
                        print(f"\n\nClosing: {output_file}\n\n")
                        break
                layer_data = output.pop('data')
                inner_view = copy(output.pop('view'))
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
        output_file = output_params.pop('output_file')
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
                inner_view = copy(output.pop('view'))
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
                 output=f"{output_file}.preview.pdf")
    return timer

def runner_call(queue, callback, params):
    """Put the results of callback using parameter into the queue

    """
    output = callback(**params)
    queue.put(output)
    return output

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
        # this is only neede for the entropy part below
        view = params.get('view')
        entropy_as_ubyte = params.pop('entropy_as_ubyte', False)
        blurred_view = runner_call(
            blur_q,
            view_blurred,
            params
        )
        blur_layers = blurred_view['data']
        view = blurred_view['view']
        entropy_layer = runner_call(
            entropy_q,
            view_entropy,
            dict(
                blur_layers=blur_layers,
                view=view,
                entropy_as_ubyte=entropy_as_ubyte
            )
        )
    return timer



