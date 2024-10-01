"""
This module defines a command line interface.
"""
import os
from skimage.filters import gaussian
from .plotting import (
    plot_categories,
    figure_categories,
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
                             "- entropy: perform Gaussian blur on each category"
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
                source=args.source,
                output=f"{ fname }_category_entropy_{args.sigma}{ fextension }",
                view=(args.hstart, args.vstart, args.size, args.size),
                img_filter=gaussian,
                filter_params=dict(sigma=args.sigma,)
            )
        else:
            plot_categories(
                source=args.source,
                output=f"{ fname }{ fextension }",
                view=(args.hstart, args.vstart, args.size, args.size),
            )
            figure_categories(
                source=args.source,
                view=(args.hstart, args.vstart, args.size, args.size),
                fig_params = dict(
                    output=f"{ fname }_category{ fextension }",
                )
            )
            # now with filter
            figure_categories(
                source=args.source,
                view=(args.hstart, args.vstart, args.size, args.size),
                fig_params = dict(
                    output=f"{ fname }_categories_filtered_{args.sigma}{ fextension }",
                ),
                img_filter=gaussian,
                params=dict(sigma=args.sigma,)
            )
    elif args.do == 'export':
        pass
    else:
        raise Exception('You need to set a valid option for --do')


if __name__ == "__main__":
    main_cli()
