import os
import urllib
import zipfile
import math
from functools import lru_cache
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from contextlib import contextmanager

import folium

import numpy
import geopandas
import pandas
import pandas_gbq
import rasterio
import rasterio.features
from ipyleaflet import GeoData, CircleMarker
from shapely import geometry
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from whitebox import WhiteboxTools

import altair
from ipywidgets import widgets


DEFAULT_PROJECT = "stormwaterheatmap-hydrology"
SQM_TO_ACRES = 4046.8564224
_VALID_SOIL_TYPES = {0: 'A/B', 1: 'C', 2: 'D'}
_VALID_LANDUSES = {0: 'forest', 1: 'pasture', 2: 'lawn', 5: 'impervious'}
_VALID_SLOPE_CLASSES = {0: 'flat', 1: 'moderate', 2: 'steep'}
ALL_HRUs = [
    f'hru{soil}{landuse}{slope}'
    for soil, landuse, slope in product(_VALID_SOIL_TYPES, _VALID_LANDUSES,
                                        _VALID_SLOPE_CLASSES)
    if (landuse != 5) or (landuse == 5 and soil == 2)
]
ALL_MONTHS = list(range(1, 13))

def _get_soil(hruname):
    stype = int(hruname[0])
    return _VALID_SOIL_TYPES[stype]


def _get_land_cover(hruname):
    lutype = int(hruname[1])
    return _VALID_LANDUSES[lutype]


def _get_slope(hruname):
    slopetype = int(hruname[2])
    return _VALID_SLOPE_CLASSES[slopetype]

# 'hru description': lambda df: df.apply(lambda row: (row['soil'])+", "+(row['land cover'])+", "+(row['slope']), axis=1),
def _hru_descr(hruname):
    return ', '.join([fxn(hruname) for fxn in (_get_soil, _get_land_cover, _get_slope)])

@contextmanager
def get_back_workingdir(workingdir):
    """
    path. It's dumb. This is how we work around it while it's
    not fixed.
    """
    yield workingdir
    os.chdir(workingdir)


def dd_to_meters (dd_value:None):
    return 6378137.0*(3.14159)*dd_value/180.0 ##approximation of degrees to meters


def dd_to_acres (dd_value:None):
    dd_root = math.sqrt(dd_value)
    sqft = (20925646.33*(3.14159)*dd_root/180.0)**2 ##converts an area
    acres = sqft/43560
    return acres


def near(point, source_points=None): #.geometry.unary_union):
    # find the nearest point and return the corresponding Place valud
    """
    find the nearest point and return the corresponding Place value
    Parameters
    ----------
    point: point of interest
    source_points : points to get info from

    """
    nearest = source_points.geometry == nearest_points(point, source_points.geometry.unary_union)[1]
    return source_points.loc[nearest, 'DBAssetID'].to_numpy()[0]


def get_HRUs_in_watershed(ee, watershed_gdf, tile_scale=10, image_file=None):
    """
    Retrieve a list of HRUs within a watershed from

    Parameters
    ----------
    ee : Initialized Earth Engine object
    watershed_geom : shapely Polygon
        Shapely object representing the watershed
    tile_ecale : int (optional, default = 10)
        Scale to which the detail of the image file should be reduced
    image_file : str (optional)
        EE image name that contains the HRUs. When not provided, falls back to:
        "users/stormwaterheatmap/hruOut_fixed".

    Returns
    -------
    HRUs : dict
        Dictionary where the keys are the HRU and the values are the areas in
        square meters

    """
    if not image_file:
        image_file = "users/stormwaterheatmap/public/hrusJan2020Mode"
    watershed_layer = GeoData(geo_dataframe=watershed_gdf, name='watershed')
    watershed = ee.FeatureCollection((watershed_layer.data)["features"])
    hrus = ee.Image(image_file)
    area = ee.Image.pixelArea()
    img = ee.Image.cat(area,hrus)
    regionReduce = img.reduceRegion(
        ee.Reducer.sum().group(1, 'hru'),
        watershed, 2
    ).get('groups').getInfo()
    df = pandas.DataFrame(regionReduce).rename(columns={"sum": "sq.m"})
    return df


def process_HRUs(df):
    hru_mapper = dict(zip(
        [40, 41, 42, 140, 141, 142, 240, 241, 242, 50, 51, 52, 150, 151, 152],
        [250, 251, 252, 250, 251, 252, 250, 251, 252, 250, 251, 252, 250, 251, 252]
    ))

    hru_df = (
        df.assign(hru=lambda df: df['hru'].replace(hru_mapper))
        .assign(hruname=lambda df: df['hru'].map(lambda x: f'{x:03d}'))
        .loc[lambda df: ~df['hruname'].str.contains('3')]
        .groupby(by=['hru', 'hruname']).sum()
        .reset_index()
        .assign(**{
            'acres': lambda df: df['sq.m'] / SQM_TO_ACRES,
            'soil': lambda df: df['hruname'].apply(_get_soil),
            'land cover': lambda df: df['hruname'].apply(_get_land_cover),
            'slope': lambda df: df['hruname'].apply(_get_slope),
            'hru description': lambda df: df['hruname'].apply(_hru_descr),
        })
    )
    return hru_df


@lru_cache(maxsize=128)
def downloadDEM(x,
                y,
                filename="users/stormwaterheatmap/public/hydrodem_wgs84",
                scale=10,
                outputdir=None,
                ee=None):
    """
    1. Using earthengine, this function gets a dem (default is the usgs nhd plus raster)
    2. Clips it to the relevant HUC12 boundary
    3. Generates a url for download
    4. Downloads and unzips it
    5. returns a dictionary with the huc12 information

    Parameters
    ----------
    x: longitude of the pour point
    y: latittude of the pour point
    filename: earthengine dem asset
    scale: analysis scale in sq.m

    Returns
    -------
    filename of tif

    """
    geometry = ee.Geometry.Point(x, y)
    image = ee.Image(filename)
    image = image.reduceResolution(ee.Reducer.mean())

    #//get HUC12 boundary the point lies in
    HUC12 = ee.FeatureCollection("USGS/WBD/2017/HUC12")
    HUCBounds = HUC12.filterBounds(geometry)

    #Clip dem to the huc12 boundary
    clipBoundary = ee.Feature(HUCBounds.first()).bounds()

    clippedDEM = image.clip(clipBoundary)

    #generate url
    url = clippedDEM.getDownloadURL({'params': {'name': 'dem', 'scale': scale}})
    ## extract file from url
    #zipurl = url
    # Download the file from the URL
    zipresp = urllib.request.urlopen(url)
    # Create a new file on the hard drive
    tempzip = open("/tmp/tempfile.zip", "wb")
    # Write the contents of the downloaded file into the new file
    tempzip.write(zipresp.read())
    # Close the newly-created file
    tempzip.close()
    # Re-open the newly-created file with ZipFile()
    zf = zipfile.ZipFile("/tmp/tempfile.zip")
    tif_file_name = [k for k in zf.namelist() if '.tif' in k] #gets the name of the tif file
    # Extract its contents into <extraction_path>
    # note that extractall will automatically create the path
    zf.extractall(outputdir)
    #close the ZipFile instance
    zf.close()

    return tif_file_name


def generate_watershed_polygon(outputdir=None):
    """
    Converts a raster of a watershed to a geodataframe

    Parameters
    ----------
    raster_filename: location of the local watershed raster file (tif)

    Returns
    -------
    watershed_polygon : Pandas GeoDataFrame with the watershed boundary
    """
    with rasterio.open(str(outputdir / 'watershed.tif')) as dataset:
        # Read the dataset's valid data mask as a ndarray.
        data = [(int(value), Polygon(geom['coordinates'][0]))
                for (geom, value) in rasterio.features.shapes(
                    dataset.read(1), transform=dataset.transform)]
        gdf = geopandas.GeoDataFrame(
            data, columns=['value', 'geometry'], crs=dataset.crs)
    watershed_polygon = (gdf[gdf.value > 0]).dissolve(by="value")
    return watershed_polygon


def watershed_full_workflow(wbt, point, demfilename=None, outputdir=None):
    """
    performs full watershed raster workflow:
    1. gets a shapely point file and saves it as an ESRI shapefile
    2. Does raster filling, flow accumulation
    3. Snaps pour point to raster
        #To Do: right now the snapping distance is hard coded, this
        can be extracted as a variable.
    4. Saves rasters to directory.


    Parameters
    ----------
    point: pour point (shapely point)
    dempath: path to local dem

    Returns
    -------
    No function returns. Saves rasters to local directory

    """
    get_back_workingdir(wbt.work_dir)
    pt = xy_to_gdf(point.x,point.y)
    shpfile = str(outputdir / 'pnt.shp')
    pt.to_file(shpfile)
    dempath = outputdir / demfilename
    wbt.flow_accumulation_full_workflow(
        dempath,
        out_dem=str(outputdir / 'filled_dem.tif'),
        out_pntr=str(outputdir / 'flow_dir.tif'),
        out_accum=str(outputdir / 'flow_accum.tif'))
    wbt.snap_pour_points(
        pour_pts=str(outputdir / "pnt.shp"),
        flow_accum=str(outputdir / "flow_accum.tif"),
        output=str(outputdir / "snapped.shp"),
        snap_dist=0.001)
    wbt.watershed(
        d8_pntr=str(outputdir / "flow_dir.tif"),
        pour_pts=str(outputdir / "snapped.shp"),
        output=str(outputdir / "watershed.tif"))


def xy_to_gdf(x=None,y=None):
    """
    takes x,y information and creates a pandas geodataframe with
    geometry information.

    Returns
    -------
    geodataframe

    """
    df = pandas.DataFrame({"x": [x], "y": [y]})
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.x, df.y))
    return gdf


def add_ee_layer(self, ee, ee_image_object, vis_params, name):
    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr=
        'Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True).add_to(self)


def run_query(datadir=None, x=None, y=None):
    #Authenticate to bigquery
    key_path = datadir / "stormwaterheatmap-hydrology-8308154a1232.json"
    pandas_gbq.context.project = "stormwaterheatmap-hydrology"
    pandas_gbq.context.credentials = service_account.Credentials.from_service_account_file(
        str(key_path))

    #Generate query statement
    gridquery = """
        with location as (
            SELECT
            st_contains(st_geogfromtext(wkt_geom),
                st_geogpoint(
                    {lon},
                    {lat}
                    ) ) as grid,
                    geohash
            FROM
            `stormwaterheatmap-hydrology.geometry.geohash` )
            select geohash from location
            where grid
        """.format(lon=x, lat=y)
        #run query to find precip grid geohash
    project_id = "stormwaterheatmap-hydrology"
    bqOutput = pandas.read_gbq(gridquery, project_id=project_id)
    gridID = bqOutput.values[0][0]
    namesAreas = []
    for hru in hrustoQuery.keys():
        namesAreas.append(
        hru + "*" + str(hrustoQuery.get(hru))
        )
    quantstmt = """
    approx_quantiles( ' + '
    + '.join({namesAreas}) + ', '
    +str(numofQuantiles)+ ')
    AS Quantiles'
    """.format(namesAreas=namesAreas)
    #generate quantile query statement
    qry = """
        Select
        {selectstmt}
        FROM   `stormwaterheatmap-hydrology.gfdl.{grid_id}`
        WHERE  year BETWEEN {year0} AND    {yearN}
        AND    month IN {months}
        """.format(
            grid_id=gridID, selectstmt=quantstmt, year0=year0, yearN=yearN, months=months)
    #Run query
    bqOutput = pandas.read_gbq(qry, project_id=project_id)
    return bqOutput


def process_quantiles(flow_quantiles, name):
    flows = numpy.array(flow_quantiles.Quantiles[0]) * 9.80962964e-6  # mm/hr*m2 to cfs
    #flows = numpy.delete(flows, -1)  # remove the 0 value
    n = (flows.shape[0]-1)
    probs = numpy.arange(0, 1+(1 / n), 1 / n)
    excd = (1 - probs).round(2)
    return pandas.DataFrame({'Exceedance': excd, name: flows})


def onclick_handler(
        # from event
        event=None, type=None, coordinates=None,
        id=None, feature=None, properties=None,
        #required
        sites=None,  mapobj=None, id_box=None,
        lat_box=None, lng_box=None, accept_button=None,
        watershed_dict=None, wbt=None):

    if coordinates is None:
        return
    else:
        pour_point = Point(coordinates[::-1])
        #write pour point to dictionary
        #watershed_dict["pour_point"] = xy_to_gdf(8,5)
        lat_box.value = pour_point.y
        lng_box.value = pour_point.x
        #convert to gdf
        pour_point_gdf = xy_to_gdf(pour_point.x,pour_point.y) #xy_to_gdf(pour_point.x,pour_point.y)
        watershed_dict["pour_point"] = pour_point_gdf
        sites['Nearest'] = pour_point_gdf.apply(
            lambda row: near(row.geometry, sites), axis=1)
        id_box.value = str(
            pour_point_gdf.apply(lambda row: near(row.geometry, sites), axis=1)[0])
        accept_button.disabled = False
        accept_button.icon = ''
        accept_button.button_style = "info"


def remove_existing_layer(mapobj, layername):
    all_names = [x.name for x in mapobj.layers]
    if layername in all_names:
        mapobj.remove_layer(mapobj.layers[all_names.index(layername)])


def accept_button_click_handler(button, mapobj=None, watershed_dict=None, wbt=None,
                                ee=None, outputdir=None, info_panel=None):
    button.disabled = True
    center = (watershed_dict['pour_point'].loc[0, 'y'], watershed_dict['pour_point'].loc[0, 'x'])
    mapobj.zoom = 12
    mapobj.center = center

    circle = CircleMarker(location=center, radius=50, fill_color="blue", stroke=False, name='Buffer')
    remove_existing_layer(mapobj, circle.name)
    mapobj.add_layer(circle)
    info_panel.layout.visibility = 'visible'
    info_panel.value = '...fetching DEM'

    dem_path = downloadDEM(x=center[1], y=center[0], outputdir=outputdir, ee=ee)
    info_panel.value = '...delineating watershed'

    watershed_full_workflow(wbt, point=watershed_dict['pour_point'].loc[0, 'geometry'],
                            demfilename=dem_path[0], outputdir=outputdir)

    info_panel.value = '...extracting watershed'
    watershed_gdf = generate_watershed_polygon(outputdir)
    watershed_dict["watershed_geometry"] = watershed_gdf
    watershed_dict["centroid"] = watershed_gdf.centroid

    wshd_layer = GeoData(geo_dataframe=watershed_gdf, name='watershed')
    button.button_style = "success"
    button.icon = "check"
    remove_existing_layer(mapobj, wshd_layer.name)
    mapobj.add_layer(wshd_layer)
    info_panel.value = f'''
    DBID: \n\n
    Done! Please proceed to the next cell.'''
    #Todo; update DBID in report


def hru_barchart(hru_df):
    chart = altair.Chart(hru_df)
    c1 = chart.mark_text().encode(
        y=altair.Y('hru description', axis=None)

    )
    c4 = chart.mark_bar().encode(
        x =altair.X('acres:Q'),
        y=altair.Y('hru description:N', sort=altair.EncodingSortField(field='acres', order='descending')),
        color = 'hru description:N'
        #color='land use'
    ).properties(width=300)
    #desc = c1.encode(text='hru description:N').properties(title='descriptions')
    #ac = c1.encode(text='acres:Q').properties(title='acres')
    #tab = altair.hconcat(desc, ac)
    return c4


def get_geohash_of_point(centroid, project_id=None):
    if not project_id:
        project_id = DEFAULT_PROJECT

    sql = '''with test as (
    SELECT
    st_contains(st_geogfromtext(wkt_geom),
        st_geogpoint({lon},
            {lat}) ) as grid,
            geohash
    FROM
    `stormwaterheatmap-hydrology.geometry.geohash` )
    select geohash from test
    where grid
    '''.format(lon=centroid.x, lat=centroid.y)
    return pandas.read_gbq(sql, project_id=project_id).geohash[0]


def get_flow_quantiles(hru_df, geohash, year0, yearN, nquantiles,
                       *months, project_id=None):
    if not project_id:
        project_id = DEFAULT_PROJECT

    names_and_areas = hru_df[['hruname', 'sq.m']].apply(
        lambda r: f"(hru{r['hruname']} * {r['sq.m']})", axis=1
    ).tolist()

    quantstmt = 'approx_quantiles(({names_areas}), {nquantiles}) AS Quantiles'.format(
        names_areas=' + '.join(names_and_areas),
        nquantiles=nquantiles
    )

    qry = """
    SELECT
    {selectstmt}
    FROM   `stormwaterheatmap-hydrology.gfdl.{grid_id}`
    WHERE  year BETWEEN {year0} AND {yearN}
    AND    month IN ({months})
    """.format(
        grid_id=geohash, selectstmt=quantstmt,
        year0=year0, yearN=yearN,
        months=(', '.join(str(x) for x in months))
    )
    return pandas.read_gbq(qry, project_id=project_id)



def prob_plot(quants):
    ymax = math.ceil(quants['Qhigh'].max(0))
    qplot = altair.Chart(quants).transform_fold(
        ['Qlow', 'Qhigh']
    ).mark_line().encode(
        x=altair.X('Exceedance:Q', axis=altair.Axis(format='%')),
        y=altair.Y(
            'value:Q',
            scale=altair.Scale(
                type='log',
                domain=[1,ymax],
                clamp=True),
                axis=altair.Axis(
                    title = 'Discharge (cfs)'
                )),
        color='key:N',
        tooltip=['Exceedance:Q', 'value:Q'], 
        )
    return qplot


def hilo_flow_widgets(quants):
    low = (quants.loc[quants['Exceedance'] == 0.95]['Qlow']).values[0]
    high =round((quants.loc[quants['Exceedance'] == 0.10]['Qhigh']).values[0],2)

    low_label = widgets.Label(
        value=f'Low Flow: {low:.3g} cfs',
        description='Low Flow (cfs)',
    )

    high_label = widgets.Label(
        value=f'High Flow: {round(high,3)} cfs',
        description='Low Flow (cfs)',
    )

    return widgets.VBox([
        widgets.Label('Results'),
        low_label,
        high_label
    ])


def detail_map(watershed_dict, ee=None):
    #add ee method to folium
    folium.Map.add_ee_layer = add_ee_layer

    # Set visualization parameters.
    LCvis_params = {'min': 0, 'max': 5,
                    "palette":["55775e","dacd7f","7e9e87","b3caff","844c8b","ead1ff"]};
    soils_vis = {'min': 0, 'max': 2,
                    "palette":['#564138', '#69995D', '#F06543']}
    slope_vis = {'min': 0, 'max': 2, 'dimensions': 400,
                    "palette": ['#009B9E', '#F1F1F1', '#C75DAB']}

    #Read watershed information from the watershed dictionary
    center = watershed_dict["pour_point"]
    watershed_gdf = GeoData(geo_dataframe=(watershed_dict["watershed_geometry"]), name='watershed')
    watershed = ee.FeatureCollection((watershed_gdf.data)["features"])

    # Create a folium map object.
    eeMap = folium.Map(location=[center.y, center.x], zoom_start=13)

    #download stormwaterheatmap data from earth engine

    #landcover is a stormwaterheatmap asset:
    landcover = ee.Image("users/stormwaterheatmap/landcover_2m").clip(watershed)

    #soils data is a stormwaterheatmap asset:
    soils = ee.Image("users/stormwaterheatmap/soils2m").clip(watershed)

    #we generate slope information from the USGS National Elevation dataset:
    NED = ee.Image("USGS/NED").clip(watershed)

    #classify by WWHM thresholds:
    thresholds = ee.Image([5.0, 15, 100]);
    slopeZone = ee.Terrain.slope(NED).gt(thresholds).reduce('sum');
    slope = slopeZone.clip(watershed)

    #add to the folium Map
    eeMap.add_ee_layer(ee, soils, soils_vis, 'Soils')
    eeMap.add_ee_layer(ee, slope, slope_vis, 'Slope')
    eeMap.add_ee_layer(ee, landcover, LCvis_params, 'Land cover')

    # Add a layer control panel to the map.
    eeMap.add_child(folium.LayerControl())
    return eeMap
