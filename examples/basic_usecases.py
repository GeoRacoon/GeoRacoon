from skimage.filters import gaussian
from landiv_blur import (
    plot_categories,
    figure_categories,
    plot_entropy_full
)

def main(args):
    """Generate all plots
    """
    plot_categories(
        args.source,
        (args.hstart, args.vstart),
        (args.size, args.size),
        output=f"{ args.output }.{ args.format }",
    )
    figure_categories(
        args.source,
        (args.hstart, args.vstart),
        (args.size, args.size),
        output=f"{ args.output }_categories.{ args.format }",
    )
    # now with filter
    figure_categories(
        args.source,
        (args.hstart, args.vstart),
        (args.size, args.size),
        output=f"{ args.output }_categories_filtered_{args.sigma}.{ args.format }",
        img_filter=gaussian,
        params=dict(sigma=args.sigma,)
    )
    plot_entropy_full(
        args.source,
        (args.hstart, args.vstart),
        (args.size, args.size),
        output=f"{ args.output }_categories_entropy_{args.sigma}.{ args.format }",
        img_filter=gaussian,
        filter_params=dict(sigma=args.sigma,)
    )

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
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
    parser.add_argument("--format",
                   type=str,
                   default='png',
                   help="What format to use")
    parser.add_argument("--sigma",
                   type=float,
                   default=1,
                   help="standard deviation for gaussian kernel")
    # parse the arguments
    args = parser.parse_args()
    main(args)
