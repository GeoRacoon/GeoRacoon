import os
import pickle
import numpy as np

from rasterio.errors import RasterioIOError

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from landiv_blur import io as lbio
from landiv_blur import prepare as lbprep

# ###
# Data sources
# ###
data_path = '../data/landcover'
interim_path = '../results/'
orig_fname = 'reclass_GLC_FCS30_2015_{utm_zone}.tif'
blurred_fname = 'lct_heterogeneity_{utm_zone}_{res_type}_sig_{sigma}_diam_{diameter}_trunc_{truncate}.tif'
# ###
# Output
# ###
results_path = '../results/landcover'
lct_counts_orig_fname = "land_cover_type_counts_binary_{utm_zone}.p"
lct_counts_blurred_fname = "land_cover_type_counts_blurred_{utm_zone}_diam_{diameter}_trunc_{truncate}.p"
# ###
# Parameters
# ###
utm_zone = 'utm32U'
res_type = 'lct_blurred'
truncate = 3.
# diameters
diams = [100., 500., 1000., 5000., 10000., 30000.]
# since we have uint8 maps, we use it all
max_dom = 255
dominance_scores = range(max_dom + 1)


# ###
# Plot the % difference in coverage as a function of the gaussian kernel size
# for each land-cover type
#
# - select land-cover type

# ###
# Load the original data
#
# - load the tif
source_file = os.path.join(data_path, orig_fname.format(utm_zone=utm_zone))
orig_map = lbio.load_block(source=source_file)
lc_types = np.unique(orig_map['data'])
print(f"Land-cover types found: {lc_types=}")
# - select for specific layer
# - count the # pixels
# ##
# check if pickle exists
# ##
output_file = os.path.join(
        results_path,
        lct_counts_orig_fname.format(utm_zone=utm_zone)
)
if not os.path.isfile(output_file):
    print("Counting in as binary")
    lc_types, counts = np.unique(orig_map['data'], return_counts=True)
    abs_counts = {lct: count for lct, count in zip(lc_types, counts)}
    # - export the count as pickle
    output_file = os.path.join(
            results_path,
            lct_counts_orig_fname.format(utm_zone=utm_zone)
    )
    with open(output_file, 'wb') as fobj:
        pickle.dump(abs_counts, fobj)
else:
    print(f"Using pre-compiled data. Remove '{output_file}' to recompute ")

# Load the blurred output data
#
print("Counting predominant pixels in blurred maps")
for diameter in diams:
    abs_count_blurred = dict()
    blur_params = lbprep.get_blur_params(diameter=diameter, truncate=truncate)
    # - load the tif with all the blurred land-cover types as bands
    source_file = os.path.join(
        interim_path,
        blurred_fname.format(
            utm_zone=utm_zone.lower(),
            res_type=res_type,
            **blur_params
        )
    )
    output_file = os.path.join(
            results_path,
            lct_counts_blurred_fname.format(utm_zone=utm_zone, **blur_params)
    )
    if not os.path.isfile(output_file):
        print(f"loading {source_file=}")
        for lct in lc_types:
            # - select band
            lct_blurred_map = lbio.load_block(source=source_file,
                                              indexes=[lct+1])
            print(f"\tCounting {lct=}'s")
            # - count the # pixels
            domin, count = np.unique(lct_blurred_map['data'],
                                     return_counts=True)
            # - filter for >= threshold
            abs_count_blurred[lct] = {
                dom/max_dom: np.sum(np.where(domin >= dom, count, 0))
                for dom in dominance_scores}
            print(abs_count_blurred[lct])
        # - export the counts as pickle
        with open(output_file, 'wb') as fobj:
            pickle.dump(abs_count_blurred, fobj)
    else:
        print(f"Using pre-compiled data. Remove '{output_file}' to recompute ")

# Compare 
#
# - load pickles from previous steps
bincount_file = os.path.join(
        results_path,
        lct_counts_orig_fname.format(utm_zone=utm_zone)
)
with open(bincount_file, 'rb') as fobj:
    abs_counts = pickle.load(fobj)
per_diam_abs_count_blurred = dict()
for diameter in diams:
    blur_params = lbprep.get_blur_params(diameter=diameter, truncate=truncate)
    blurcount_file = os.path.join(
        results_path,
        lct_counts_blurred_fname.format(utm_zone=utm_zone, **blur_params)
    )
    with open(blurcount_file, 'rb') as fobj:
        per_diam_abs_count_blurred[diameter] = pickle.load(fobj)

# ###
# Plot the % difference in coverage as a function of the threshold value to
# determine dominance for the different values of the Gaussian kernel size
# Note: this is a plot for each land-cover type
# ###
fig = plt.figure(tight_layout=True, figsize=(128, 98))
max_row = 3
max_col = 4
gs = gridspec.GridSpec(max_row, max_col)
for lct in lc_types:
    # create the axis for plotting
    row = int((lct - (lct % max_col)) / max_col)
    col = int(lct % max_col)
    ax = fig.add_subplot(gs[row, col])
    # x: dominance threshold
    # y: count/binary_count - 1 per diameter
    bin_count = abs_counts[lct]
    for diam in diams:
        perdominance_blur_counts = per_diam_abs_count_blurred[diam][lct]
        x = list(perdominance_blur_counts.keys())
        y = [100*(blrc/bin_count - 1)
             for blrc in perdominance_blur_counts.values()]
        ax.set_title(f"{lct=}".upper())
        ax.axhline(y=0, linestyle='dotted')
        ax.plot(x, y, label=diam)
        ax.set_ylim(max(min(y), -100), min(max(y), 500))
        ax.set_xlabel('pre-dominance threshold')
        ax.set_ylabel('area % diff to binary')

# - compute relative difference (blurred/binary - 1)
# - plot relative difference as fct of kernel size for all l-c types
lines_labels = [fig.axes[0].get_legend_handles_labels()]
# lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels, loc='upper center', ncol=4,
           title='Kernel diameter [m]')
plt.show()
fig.savefig(os.path.join(results_path, 'pre_dominance_coverage_change.pdf'))
fig.savefig(os.path.join(results_path, 'pre_dominance_coverage_change.png'))
