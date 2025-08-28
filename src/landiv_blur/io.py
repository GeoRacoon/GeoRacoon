from __future__ import annotations

import os
import glob
import rasterio


from math import floor

from typing import Any

import rasterio as rio
from rasterio.io import DatasetWriter
from rasterio.enums import ColorInterp
from rasterio.windows import Window
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    transform_bounds
)

from shapely.geometry import box as shbox
import geopandas as gpd

from numpy.typing import NDArray

from ._exceptions import (
    BandSelectionNoMatchError,
    BandSelectionAmbiguousError,
    SourceNotSavedError,
    UnknownExtensionError,
)
from ._helper import (
    check_crs_raster,
    outfile_suffix,
    serialize,
    deserialize,
    sanitize,
    match_all,
    view_to_window,
    get_scale_factor,
)
# is_needed
# needs_work (module should be moved, needs docs)
# is_tested (at least partially)
# usedin_both

# this is our namespace for tags
NS = 'LANDIV'

# TODO: General Idea - maybe we can merge some of these into io_.py class structure - so we avoid having both.
#  --> yet it is nice to have the function by themselves as well without direct need of class structures

def write_band(src:DatasetWriter,
               data:NDArray,
               bidx:int=1,
               window:Window|None=None,
               **tags):
    # TODO: is_needed (only 1 use) - needs_work - is_tested - usedin_both
    """Write data to a specific band of a tif file and set the tags


    Parameters
    ----------
    src:
      An opened file to write into
    data:
      The array to write into the file
    window:
      An optional window to specify an area to write
    **tags:
      Arbitrary number of keyword arguments that will be set as tags.
      The value provided is converted to a string with `helper.serialize`
      before the tag is written to the file.
    """
    # is_needed
    # no_work
    # not_tested
    # usedin_processing
    src.write(data, indexes=bidx, window=window)
    set_tags(src, bidx=bidx, **tags)


def update_band(src:DatasetWriter,
                data:NDArray,
                window:Window|None=None,
                **tags):
    # TODO: not_needed
    """Find a specific band and update it with data

    ..Note::
      If no band with the matching tags is found a
      `BandSelectionNoMatchError` is raised.

    Parameters
    ----------
    src:
      An opened file to write into
    data:
      The array to write into the file
    window:
      An optional window to specify an area to write
    **tags:
      Arbitrary number of keyword arguments that will be used to find
      the band to write into
    """
    # not_needed (could be useful though)
    # no_work
    # not_tested
    # usedin_both (could be)
    try:
        bidx = get_bidx(src=src, **tags)
    except BandSelectionNoMatchError as e:
        raise BandSelectionNoMatchError(
            "There was no band with matching tags. "
            "Either adapt the tags or use `write_band` "
            "with a specific band index instead if you "
            "want to write and also update the tags."
        ) from e
    else:
        src.write(data, indexes=bidx, window=window)


def export_to_tif(destination:str,
                  data:NDArray,
                  orig_profile:dict,
                  start=(0, 0),
                  **pparams):
    # TODO: not_needed
    """Export a np.array to tif, only updating a window if data is smaller

    .. note::
      This function will overwrite the dtype of the destination tif with the
      value provided in `pparams` or the data type of `data`.

    Parameters
    ----------
    destination: str
        location to export save the .tif file
    data: np.array
        The map to export
    start: tuple
      horizontal and vertical starting coordinate
    orig_profile: dict
        the profile of the original map
        (see https://rasterio.readthedocs.io/en/stable/topics/profiles.html)
    **pparams:
        further parameter to be added to the profile
    """
    # not_needed (could be useful though)
    # no_work
    # not_tested
    # usedin_both (pot.)
    profile = orig_profile.copy()
    # Note: we no longer update the size automatically as for Windows this is
    # not correct, pass height and width explicitly to update via pparams
    # # update for the correct dimensions
    # profile['height'] = data.shape[1]
    # profile['width'] = data.shape[0]
    # set the dtype explicitly of get it from the data
    profile['dtype'] = pparams.pop('dtype', str(data.dtype))
    profile.update(pparams)
    # write it:
    size = data.shape[::-1]  # since positions are inverted in numpy
    with rasterio.open(destination, "w", **profile) as dest:
        dest.write(data, window=Window(*start, *size), indexes=1)


def project_to(source, reference, output=None, nodata=None)->str | None:
    # TODO: not_needed
    """Re-projects the source map into the coordinate system of a reference map

    Parameters
    ----------
    source: str
      The path to the tif file you want to change projection
    reference: str
      The path to the tif file with the projection to apply
    output: str (optional)
      The path to write the re-projected map to.
    nodata: float, int (optional)
      The `nodata` value to set for the output (e.g. np.nan or integer)

    ..note::
       If not provided, the output file will take the name of the input file
       and add the CRS of the new projection at the end of the name.

    Returns
    -------
    str:
      The name of the file that hold the re-projected map
    """
    # not_needed (used in examples)
    # no_work
    # not_tested
    # usedin_both (potentially)
    with rio.open(reference) as ref:
        dst_crs = str(ref.profile['crs'])
    with rio.open(source) as src:
        src_crs = str(src.crs)
        if src_crs == dst_crs:
            print(f"There is nothing to project! {src_crs=} to {dst_crs=}")
            return None

        (transform, width, height) = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds
        )
        kwargs = src.meta.copy()
        # prepare the resulting profile
        kwargs.update({
          'crs': dst_crs,
          'transform': transform,
          'width': width,
          'height': height
        })
        if nodata is not None:
            kwargs['nodata'] = nodata

        if output is None:
            _base_name, _ext = os.path.splitext(source)
            output = f"{_base_name}_{dst_crs}{_ext}"
        with rio.open(output, 'w', **kwargs) as dst:
            for bidx in src.indexes:
                reproject(
                    source=rio.band(src, bidx),
                    destination=rio.band(dst, bidx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
        return output


def clip_to_bounds(source, reference, output=None):
    # TODO: not_needed
    """Clip raster to bounding box of reference raster

    Parameters
    ----------
    source: str
      The path to the tif file you want to clip
    reference: str
      The path to the tif file with the extent to use as clipping bounding box
    output: str (optional)
      The path to write the bounding box clipped map to

    Returns
    -------
    str:
      The name of the file that holds clipped map
    """
    # not_needed (used in example only)
    # no_work
    # not_tested
    # usedin_both (potentially)
    if not check_crs_raster(source, reference):
        raise ValueError("Cannot clip by BBOX - projections are not the same.")

    if output is None:
        output = outfile_suffix(source, "bounds")

    with rasterio.open(reference) as ref:
        bounds = ref.bounds
        bbox_geom = shbox(*bounds)

    with rasterio.open(source) as src:
        out_image, out_transform = mask(src, [bbox_geom], crop=True)

        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

    with rasterio.open(output, 'w', **out_meta) as dst:
        dst.write(out_image)
    return output


def _coregister_raster(source, reference, output=None):
    # TODO: needed (Internal funciton only - move this to riogrande for now)
    """Align raster to have identical resolution.

    Resolution will be calculated automatically from bounds and height/width of reference raster.

    Parameters
    ----------
    source: str
      The path to the tif file you want to co-register
    reference: str
      The path to the tif file with the pixel registration to use as reference for co-registration
    output: str (optional)
      The path to write the co-registered map to

    Returns
    -------
    str:
      The name of the file that holds co-registered map
    """
    # is_needed (in tests only)
    # needs_work (format doc)
    # not_tested
    # usedin_both

    if not check_crs_raster(source, reference):
        raise ValueError("Cannot co-register sources - projections are not the same.")

    if output is None:
        output = outfile_suffix(source, "coreg")

    with rasterio.open(source) as src:
        src_transform = src.transform
        src_nodata = src.nodata

        with rasterio.open(reference) as refsrc:
            dst_crs = refsrc.crs

            (dst_transform,
             dst_width,
             dst_height) = calculate_default_transform(src.crs,
                                                       dst_crs,
                                                       refsrc.width,
                                                       refsrc.height,
                                                       *refsrc.bounds)

        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": src_nodata})

        with rasterio.open(output, "w", **dst_kwargs) as dst:
            for bidx in src.indexes:
                reproject(
                    source=rasterio.band(src, bidx),
                    destination=rasterio.band(dst, bidx),
                    src_transform=src_transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    return output


def buffer_geometries_metric(geom_geoseries, buffer_meter, source_crs):
    # TODO: not_needed
    """ Applies a buffer to the geometries in GeoSeries given.

    ..Note: This function re-projects the GeoSeries to the respective UTM zone in order to use metric buffer and best
    distance calculations. Further empty geometries are dropped before handing back the results.

    Parameters
    ----------
    geom_geoseries: GeoPandas GeoSeries
      The geoseries holding the polygons to perform the buffer on
    buffer_meter: float, int
      The buffer in meters to apply to the ecoregion polygon before clipping.
      Needs to be negative for reducing the polygon e.g. -1000
    source_crs:
      The coordinate system of the inptu GeoSeries (taken from GeoDataframe before by user) to project to after buffer.

    Returns
    -------
    GeoSeries object:
      The buffered GeoSeries object
    """
    # is_needed (internal only)
    # needs_work (docs)
    # not_tested
    # usedin_both (io module)
    geom_utm = geom_geoseries.to_crs(geom_geoseries.estimate_utm_crs())
    geom_buff = geom_utm.buffer(buffer_meter,
                                resolution=10, cap_style='round', join_style='round')
    geom_buff = geom_buff[geom_buff.area > 0]
    return geom_buff.to_crs(source_crs)


def clip_to_ecoregion(source, shapefile, ecoregion_number, output=None, buffer_meter=None):
    # TODO: not_needed
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
    # not_needed (used in example)
    # no_work
    # not_tested
    # usedin_processing
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


def compress_tif(source, output:str|None=None, compression:str|None='lzw'):
    # TODO: is_needed (technically not_needed but still present in some bash-scripts (Folder: scripts)
    # TODO: once this goes, we may also remove the outfile_suffix I believe (but double-check)
    """Compress tif file with LZW compression

    Parameters
    ----------
    source: str
      The path to the tif file you want to compress
    output:
      Optional path to output file.
      If not set, the resulting file will inherit the filename from `source` and get
      a `_compress` appended to the filename.
      If compression is `'none'`, i.e. no compression the appendix will be '_decompressed'

    Returns
    -------
    str:
      The name of the compressed file
    """
    # is_needed
    # needs_work (docs)
    # is_tested
    # usedin_processing (part of IO module, could be used in both)
    if compression is None:
        compression = 'none'
    overwrite = False
    if output is None:
        if compression != 'none':
            output = outfile_suffix(source, "compress")
        else:
            output = outfile_suffix(source, "decompressed")
    elif output == source:
        overwrite = True
        output = outfile_suffix(source, 'tmp')

    with rasterio.Env():
        with rasterio.open(source) as src:
            profile = src.profile
            profile.update(compress=compression)

            with rasterio.open(output, 'w', **profile) as dst:
                set_tags(src=dst, bidx=None, **get_tags(src=src, bidx=None))
                for i in range(1, src.count + 1):
                    for ji, window in src.block_windows(i):
                        array = src.read(i, window=window)
                        dst.write(array, i, window=window)
                    tags = get_tags(src, bidx=i)
                    set_tags(dst, bidx=i, **tags)
                    band_names = src.descriptions[(i - 1)]
                    dst.set_band_description(i, band_names)
    if overwrite:
        os.remove(source)
        os.rename(src=output, dst=source)
        output = source
    return output
