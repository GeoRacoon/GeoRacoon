from __future__ import annotations

import rasterio



import rasterio as rio
from rasterio.mask import mask

from shapely.geometry import box as shbox
import geopandas as gpd

from .helper import (
    outfile_suffix,
)
# is_needed
# needs_work (module should be moved, needs docs)
# is_tested (at least partially)
# usedin_both

# TODO: Adapt this to convster (or similar)
# this is our namespace for tags
NS = 'LANDIV'

def clip_to_ecoregion(source, shapefile, ecoregion_number, output=None, buffer_meter=None):
    # not_needed (used in example)
    # no_work
    # not_tested
    """Clip raster file to eco-region boundary (vector-data) for given eco-region number

    Parameters
    ----------
    source: str
      The path to the tif file you want to clip
    shapefile: str
      The path to the shapefile (.shp) with the eco-region polygons for clipping
    ecoregion_number: int
      The number of the respective eco-region to clip the source to
    output: str (optional)
      The path to write the clipped map to
    buffer_meter: float, int
      The buffer in meters to apply to the eco-region polygon before clipping.
      Needs to be negative for reducing the polygon e.g. -1000

    Returns
    -------
    str:
      The name of the file that holds eco-region clipped map
    """
    if output is None:
        output = outfile_suffix(source, "eco-clip")

    with rasterio.open(source) as src:
        bounds = src.bounds
        bbox_geom = shbox(*bounds)

    gdf = gpd.read_file(shapefile, bbox=bbox_geom)
    gdf = gdf.loc[gdf.ecoregion == ecoregion_number]
    geometry = gdf["geometry"]

    # Buffer
    # Note: before buffering cut shapefiler bigger than the raster to expanded smaler chunk to avoid long processing.
    # Therfore the bbox of the raster plus the buffer distance (absolute) is used to avoid losing any areas later.
    if buffer_meter is not None:
        exp_bbox_geom = buffer_geometries_metric(gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[bbox_geom]),
                                                 buffer_meter=abs(buffer_meter),
                                                 source_crs=gdf.crs)
        exp_geometry = gpd.clip(geometry, exp_bbox_geom, keep_geom_type=True)

        geometry = buffer_geometries_metric(exp_geometry,
                                            buffer_meter=buffer_meter,
                                            source_crs=gdf.crs)

    # Clip to bbox
    geometry_clip = gpd.clip(geometry, bbox_geom, keep_geom_type=True)

    # Make sure there are no Multilinestrings
    geom_types = [t for t in geometry_clip.geom_type.unique()]
    if len(geom_types) != 1:
        if {'MultiPolygon', 'Polygon'} == set(geom_types):
            geometry_clip = geometry_clip.explode(ignore_index=True)
            if 'MultiPolygon' == geometry_clip.geom_type.unique():
                raise ValueError("Issue in transforming MultiPolygons to Polygons using geopandas.explode")
        else:
            raise ValueError(f'Unvalid geometry types in clipping json {geom_types}')

    # Clip files using shapely geometries as list and rasterio
    cutline_shape = [geom for geom in geometry_clip]

    with rio.open(source) as src:
        out_array, out_transform = mask(src, cutline_shape, crop=False) # False to keep original dimension of raster
        profile = src.profile.copy()

        profile.update({"driver": "GTiff",
                         "height": out_array.shape[1],
                         "width": out_array.shape[2],
                         "transform": out_transform})

        with rio.open(output, "w", **profile) as dst:
            dst.write(out_array)
    return output
