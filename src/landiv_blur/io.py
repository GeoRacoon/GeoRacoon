from __future__ import annotations

import os
import glob
import rasterio


from math import floor

import rasterio as rio
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
from osgeo import gdal, ogr
import geopandas as gpd

from .exceptions import (
    BandSelectionNoMatchError,
    BandSelectionAmbiguousError,
    SourceNotSavedError,
    UnknownExtensionError,
)
from .helper import (
    check_crs_raster,
    outfile_suffix,
    serialize,
    deserialize,
    sanitize,
    match_all,
    get_scale_factor,
)

# this is our namespace for tags
NS = 'LANDIV'

def set_tags(src, bidx=None, ns=NS, **tags):
    """Update tags for a dataset or a single band of a dataset.

    Since metadata in a tif file is stored as a string the value of each tag is
    serialized and converted to a string with `helper.serialize`.

    ..Example::

      Setting the tags

      - 'category': 1
      - foo: 'bar'

      on band with index 2 in some opened tif file (`src`) is done with:

      ```python
      set_tags(src=src, bidx=2, category=1, foo='bar')
      ```

    ..Note::
      A tag name must satisfy the python variable naming convention and
      must be different from `src`, `bidx` and `ns` as these are reserved
      for the arguments of this function.

    ..Note::
      Existing tags are either kept or updated

    Parameters
    ----------
    src:
      `tif` file openend with `rasterio.open` in write mode (i.e. "w" or "r+")
    bidx:
      Index of the band to set tags for (starting from 1 as is the convention
      in rasterio). If set to `None` then the tags are set for the entire
      dataset.
    ns:
      The namespace to set the tags in.
      
      ..Note::
        It is dicouraged to change this value from the default as all tagging
        related methods of this package use the same default namespace.
    **tags:
      Arbitrary number of keyword arguments that will be set as tags.
      The value provided is converted to a stirng with `helper.serialize`
      before the tag is written to the file.
    """
    if bidx is None:
        bidx = 0
    # serialize the tag values:
    serialized_tags = serialize(tags)
    src.update_tags(ns=ns, bidx=bidx, **serialized_tags)

def get_tags(src, bidx=None, ns=NS)->dict:
    """Get all the tags and deserialize the values

    Parameters
    ----------
    src:
      `tif` file openend with `rasterio.open`
    bidx:
      Index of the band to get tags from (starting from 1 as is the convention
      in rasterio). If set to `None` then the tags for the entire dataset are
      returned.
    ns:
      The namespace to get the tags from.
      
      ..Note::
        It is dicouraged to change this value from the default as all tagging
        related methods of this package use the same default namespace.
    """
    if bidx is None:
        bidx = 0
    return deserialize(src.tags(bidx=bidx, ns=ns))

def find_bidxs(src, ns=NS, **tags):
    """Find all bands in src for which all tags match

    Parameters
    ----------
    src:
      `tif` file openend with `rasterio.open`
    ns:
      The namespace to set the tags in.

      ..Note::
        It is dicouraged to change this value from the default as all tagging
        related methods of this package use the same default namespace.
    **tags:
      Arbitrary number of keyword arguments that will be compared to the tags
      of the bands in the dataset.
    """
    _tags = sanitize(tags) 
    matching_bidxs = []
    for bidx in src.indexes:
        b_tags = get_tags(src=src, bidx=bidx, ns=ns)
        if match_all(targets=_tags, tags=b_tags):
            matching_bidxs.append(bidx)
    return matching_bidxs

def get_bidx(src, ns=NS, **tags)->None|int:
    """Get the index of the band with matching tags

    You can specify an arbitrary number of tags by passing keyword arguments
    to this selector.
    Make sure that the provided tags identify one and only one specific band.

    ..Example::

      Get the band with the tags

      - 'category': 1
      - foo: 'bar'

      ```python
      bidx = get_bidx(src=src, category=1, foo='bar')
      ```

    ..Note::
      If multiple bands match the provided tags a
      `BandSelectionAmbiguousError` is raised.
      If you want to get all bands that match the criterion use the
      `get_bands` method instead.

      If no band with matching tags if sound a
      `BandSelectionNoMatchError` is raised.

    ..Note::
      The values of the provided tags are first serialized and then
      deserialized again with `helper.serialize`, resp. `helper.deserialize`,
      before comparing to the tags from the provided file.

      The reason for this procedure is the fact that the values of tags are
      converted to and stored as strings in the tif metadata.
      Serializing the values with `helper.serialize` allows us to know how arbitrary
      python objects are converted.
      As a consequence, we serialize/deserialize the values of the provided
      tags to bring them into the form they will we when loading them from
      the tif.

    Parameters
    ----------
    src:
      `tif` file openend with `rasterio.open`
    ns:
      The namespace to set the tags in.

      ..Note::
        It is dicouraged to change this value from the default as all tagging
        related methods of this package use the same default namespace.
    **tags:
      Arbitrary number of keyword arguments that will be compared to the tags
      of the bands in the dataset.
    """
    # serialize/deserialize tags
    _tags = sanitize(tags) 
    matching_bidxs = find_bidxs(src=src, ns=ns, **_tags)
    matches = len(matching_bidxs)
    if matches > 1:
        raise BandSelectionAmbiguousError(
            f"The selection\n\t{_tags}\nleads to multiple matches:\n\t{matching_bidxs}"
        )
    elif matches == 0:
        raise BandSelectionNoMatchError(
            f"No band matches the tags: {_tags}"
        )
    return matching_bidxs[0]

def get_bands(source:str, ns=NS, **tags)->list[tuple[str,int]]:
    """Find all bands that match specific tags

    This method check the metadata (including those of bands)
    in one or several tif files and returns the file paths, as well as,
    the band indexes for all bands with matching tags.

    Whenever a band has tags that match, the name of the tif file,
    as well as, the band index are added to the list of returned
    matches.

    If the tags are found in the metadata of a dataset the path
    to the file and a band index of None are added to the list of
    returned matches (this different form the rasterio convention to
    that uses bidx=0 for "all bands" - I find that confusing).

    Parameters
    ----------
    source:
      A string that is fed to `glob.glob` leading to (potentially) multiple
      source files that will be checked
    ns:
      The namespace to search the tags in.

      ..Note::
        It is dicouraged to change this value from the default as all tagging
        related methods of this package use the same default namespace.

    **tags:
      Arbitrary number of keyword arguments that will be compared to the tags
      of each tif file
    """
    _tags = sanitize(tags)
    _sources = glob.glob(source)
    matches = []
    for source in _sources:
        with rasterio.open(source, "r") as src:
            ds_tags = get_tags(src=src, bidx=None, ns=ns)
            bidxs = find_bidxs(src=src, ns=ns, **_tags)
            if match_all(targets=_tags, tags=ds_tags):
                bidxs.append(None)  # use bidx None to indicate tags of the file
        for bidx in bidxs:
            matches.append((source, bidx))
    return matches


def load_map(source, indexes=None)->dict:
    """Load a map from a tif

    Returns
    -------
    dict:
       Returns the callback of
       `load_block(source=source, start=None, size=None, indexes=indexes)`
    """
    return load_block(source=source, start=None, size=None, indexes=indexes)


def load_block(source, start=None, size=None, indexes=None, scaling=None,
               **params)->dict:
    """Get a block from a *.tif file along with the transform

    Parameters
    ----------
    source: str
      The path to the tif file to load
    start: tuple
      horizontal and vertical starting coordinate

      If not provided (or set to `None`) then the coordinate (0,0) is used
    size: tuple
      width and height of the block to extract

      If not provided the entire map is loaded.
    indexes: list of int, int or None
      If a list is provided a 3D array is returned, if not a 2D array.

      ..note::
        The index of the first band is 1 not 0!

    scaling: tuple[float] | None
      Factors to rescale the number of pixels. Values >1 will upscale.

      ..note::
        If scaling is provided, the keyword argument `scaling_method` should
        also be given and identify a method from `rasterio.enums.Resampling`
        to apply for the scaling

    Returns
    -------
    dict:
       data: holding a numpy array with the actual data
       transform: an ???.Affine object that encodes the transformation used
       orig_meta: The meta information of the original .tif file
       orig_profile: The profile information of the original .tif file
    """
    with rasterio.open(source) as img:
        # TODO: rasterio Window allows using slices. In doing so we could
        #       harmonize what we call blocks and views and just work with
        #       slices.
        # Lookup table for the color space in the source file
        colorspace = dict(zip(img.colorinterp, img.indexes))

        if len(colorspace.keys()) == 3:
            # Read the image in the proper order so the numpy array is RGB
            idxs = [
                colorspace[ci]
                for ci in (ColorInterp.red,
                           ColorInterp.green,
                           ColorInterp.blue)
            ]
        elif indexes is not None:
            idxs = indexes
        else:
            idxs = img.indexes
        if any((start, size)):
            assert all((start, size)), \
                   f"{start=} and {size=} both need to be set or both None"
            riow = Window(*start, *size)
            transform = img.window_transform(riow)
            width = size[0]
            height = size[1]
        else:
            width = img.width
            height = img.height
            riow = None
            transform = img.transform

        # perform a re-scaling if needed
        if scaling:
            out_shape = (
                img.count,
                floor(img.height * scaling[0]),
                floor(img.width * scaling[1])
            )
            print(out_shape)
            print(width)
            resampling = params.get('scaling_method', Resampling.bilinear)
        else:
            out_shape = None
            resampling = Resampling.nearest
        # read out the desired part
        data = img.read(idxs,
                        window=riow,
                        out_shape=out_shape,
                        resampling=resampling)
        if scaling:
            # scale image transform
            transform = transform * transform.scale(
                (width / data.shape[-1]),
                (height / data.shape[-2])
            )
        return {
            'data': data,
            'transform': transform,
            'orig_profile': img.profile.copy()
        }


def export_to_tif(destination, data, orig_profile, start=(0, 0),  **pparams):
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
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
        return output


def clip_to_bounds(source, reference, output=None):
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


def coregister_raster(source, reference, output=None):
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

    if not check_crs_raster(source, reference):
        raise ValueError("Cannot co-register sources - projections are not the same.")

    if output is None:
        output = outfile_suffix(source, "coreg")

    with rasterio.open(source) as src:
        src_transform = src.transform
        src_nodata = src.nodata

        with rasterio.open(reference) as refsrc:
            dst_crs = refsrc.crs

            dst_transform, dst_width, dst_height = calculate_default_transform(src.crs,
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
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src_transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    return output


def buffer_geometries_metric(geom_geoseries, buffer_meter, source_crs):
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
    geom_utm = geom_geoseries.to_crs(geom_geoseries.estimate_utm_crs())
    geom_buff = geom_utm.buffer(buffer_meter,
                                resolution=10, cap_style='round', join_style='round')
    geom_buff = geom_buff[geom_buff.area > 0]
    return geom_buff.to_crs(source_crs)


def clip_to_ecoregion(source, shapefile, ecoregion_number, output=None, buffer_meter=None):
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

    # Write to temporary geojson
    geojson_name = "tmp_processing_ecoregion.geojson"
    geometry_clip.to_file(geojson_name, driver='GeoJSON')

    # Clip actual file
    gdal.Warp(output, source,
              cutlineDSName=geojson_name,
              cutlineLayer="tmp_processing_ecoregion",
              cropToCutline=False,  # if True, this extent the raster to the larger shapefile area outside the
              copyMetadata=True)

    if os.path.exists(geojson_name):
        os.remove(geojson_name)

    return output


def compress_tif(source, output=None):
    """Compress tif file with LZW compression

    Parameters
    ----------
    source: str
      The path to the tif file you want to compress

    Returns
    -------
    str:
      The name of the compressed file
    """
    if output is None:
        output = outfile_suffix(source, "compress")

    with rasterio.Env():
        with rasterio.open(source) as src:
            profile = src.profile
            profile.update(compress="lzw")

            with rasterio.open(output, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    array = src.read(i)
                    dst.write(array, i)
                    tags = get_tags(src, bidx=i)
                    set_tags(dst, bidx=i, **tags)
                    band_names = src.descriptions[(i - 1)]
                    dst.set_band_description(i, band_names)
    return output
