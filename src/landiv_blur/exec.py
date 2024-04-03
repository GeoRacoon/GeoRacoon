"""
This module defines a command line interface.
"""
import os
from skimage.filters import gaussian
from .plotting import (
    plot_landtypes,
    plot_layers,
    plot_entropy_full
)


def main_cli():
    """Command line entry point.
    """
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--do",
                        type=str,
                        default='show',
                        help="Tell landiv what it is you want to do."
                             "\n Options are:\n"
                             "- show: to create a plot\n"
                             "- export: to export the map to a new file")
    parser.add_argument("--source",
                        type=str,
                        help="Path to the file to load")
    parser.add_argument("--output",
                        type=str,
                        help="Where to save the image")
    parser.add_argument("--size",
                        type=int,
                        default=10000,
                        help="Number of pixels for with & height")
    parser.add_argument("--hstart",
                        type=int,
                        default=0,
                        help="Where to start horizontally")
    parser.add_argument("--vstart",
                        type=int,
                        default=0,
                        help="Where to start vertically")
    parser.add_argument("--type",
                        type=str,
                        default="entropy",
                        help="What type of action to do."
                             "\n Options are:\n"
                             "- entropy: perform Gaussian blur on each layer"
                             " then compute the per cell entropy")
    parser.add_argument("--sigma",
                        type=float,
                        default=1,
                        help="standard deviation for gaussian kernel")
    # parse the arguments
    args = parser.parse_args()
    if args.do == 'show':
        fname, fextension = os.path.splitext(args.output)
        if args.type == 'entropy':

            plot_entropy_full(
                args.source,
                (args.hstart, args.vstart),
                (args.size, args.size),
                output=f"{ fname }_layers_entropy_{args.sigma}{ fextension }",
                img_filter=gaussian,
                params=dict(sigma=args.sigma,)
            )
        else:
            plot_landtypes(
                args.source,
                (args.hstart, args.vstart),
                (args.size, args.size),
                output=f"{ fname }{ fextension }",
            )
            plot_layers(
                args.source,
                (args.hstart, args.vstart),
                (args.size, args.size),
                output=f"{ fname }_layers{ fextension }",
            )
            # now with filter
            plot_layers(
                args.source,
                (args.hstart, args.vstart),
                (args.size, args.size),
                output=f"{ fname }_layers_filtered_{args.sigma}{ fextension }",
                img_filter=gaussian,
                params=dict(sigma=args.sigma,)
            )
    elif args.do == 'export':
        pass
    else:
        raise Exception('You need to set a valid option for --do')


if __name__ == "__main__":
    main_cli()
