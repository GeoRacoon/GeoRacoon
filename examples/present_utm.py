import os
from rasterio.enums import Resampling
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from landiv_blur import io as lbio
from landiv_blur import prepare as lbprep
from landiv_blur import plotting as lbplot

utm_zone = 'utm32U'
# we are going to re-scale the maps by changing the resolution by this factor:
scaling = 0.5
# the land-cover types to plot
lc_types = [1, 2, 8]  # , 10]


# this is going to be our figure
fig = plt.figure(tight_layout=False, figsize=(128, 98))
gs = gridspec.GridSpec(6, 8)

# fig, axs = plt.subplots(2, 4, figsize=(128, 128))

data_path = '../data/landcover'
filename = 'reclass_GLC_FCS30_2015_{utm_zone}.tif'
source_file = os.path.join(data_path, filename.format(utm_zone=utm_zone))
# first plot the land-cover types as different colors
print(f"plotting overview map for\n{source_file=}")
ax = fig.add_subplot(gs[1:5, :2])
lbplot.plot_block(source=source_file, start=None, size=None, ax=ax,
                  scaling=scaling)

# first we get the individual layers from the source file
# for the layers we increase visibility:
s_method_layer = Resampling.nearest

# to plot the layers individually we have a little helper function
_axs = lbplot.plot_layers(source_file, None, None, scaling=scaling,
                          scaling_params=dict(scaling_method=s_method_layer),
                          # see #31 for the layer combination
                          layers=lc_types,
                          fig_params=dict(fig=fig, gs=gs,
                                          gsr=0, gsc=2, rl=1,
                                          rstep=2, cstep=2))
                          # layers=[[1, 4], 2, [8, 9], 10], axs=axs)

# now we load the blurred layers
result_map = 'lct_heterogeneity_{utm_zone}_{res_type}_sig_{sigma}_diam_{diameter}_trunc_{truncate}.tif'
results_path = '../results/'
diameter = 30000.
truncate = 3.
scaling_method = Resampling.nearest
blur_params = lbprep.get_blur_params(diameter=diameter, truncate=truncate)
blurred_map = result_map.format(utm_zone=utm_zone.lower(),
                                res_type='lct_blurred',
                                **blur_params)
source_file = os.path.join(results_path, blurred_map)
print(f"plotting blurred map:\n{source_file=}")

for i, lct in enumerate(lc_types):
    print(f"{source_file=}")
    print(f"{scaling=}")
    print(f"{lct=}")
    ax = fig.add_subplot(gs[2*i:2*i+2, 4:6])
    utm_map = lbio.load_block(source_file,
                              indexes=lct+1,
                              )
                              # scaling=scaling)
    # encoded the layers starting from 0
    lbplot.show_layer(utm_map['data'], layer=lct,
                      transform=utm_map['transform'],
                      ax=ax)

# now we get the entropy map and plot it
# TODO: for illustration purposes we show the entropy map for a smaller kernel
#       for larger kernels (constant block size) there is still an issue in the
#       recombination procedures
diameter = 1000.
blur_params = lbprep.get_blur_params(diameter=diameter, truncate=truncate)
entropy_map = result_map.format(utm_zone=utm_zone.lower(),
                                res_type='entropy',
                                **blur_params)
source_file = os.path.join(results_path, entropy_map)
print(f"printing entropy map\n{entropy_map=}")
ax = fig.add_subplot(gs[1:5, 6:])
ax, plot_params = lbplot.plot_entropy(source=source_file, size=None,
                                      start=None,
                                      scaling=scaling,
                                      fig_params=dict(fig=fig, ax=ax))
# adding the colormap
# fig.colorbar(plot_params[0], ax=ax)
# finally we save it
fig.savefig(os.path.join(results_path, 'utm_32u.pdf'))
# fig.savefig('single_lct.pdf', dpi=lbplot.DPI)
