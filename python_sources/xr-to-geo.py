
import os 
import xarray as xr 
import geoviews.util as gu 
import geoviews as gv 
import shapely as sh
from geojson import FeatureCollection,Feature
import matplotlib.cm as cm
from matplotlib.colors import rgb2hex
import numpy as np 
from collections.abc import Iterable
import pandas as pd 

  
def get_valid_polygon(poly): 
    """
    Modifie la geometrie d'entree de telle sorte a avoir un polygone valide en sortie. 
    """
    p = sh.geometry.asPolygon(poly)
    if p.is_valid: 
        return p 
    pb_bis = p.buffer(1e-5).buffer(-1e-5)
    if pb_bis.area>0:
        return pb_bis 

    line_non_simple = sh.geometry.LineString(poly)
    mls = sh.ops.unary_union(line_non_simple)
    polygons = list(sh.ops.polygonize(mls))
    if len(polygons)> 1: 
        "Case with multiple polygons"
        polyf= sh.geometry.Polygon(polygons[0])
        for i in np.arange(1,len(polygons)): 
            polyf=polyf.union(sh.geometry.Polygon(polygons[i]))
        return polyf
    elif polygons != []: 
        "Case with one polygon"
        return  sh.geometry.Polygon(polygons[0])
    else: 
        return sh.geometry.Polygon([])
    return sh.geometry.Polygon([])
    
def change_invalid_polygon(geo_contour):
    import shapely.ops 
    l_poly = []
    if geo_contour.geom_type == 'MultiPolygon':
        for poly in geo_contour.geoms:
            if poly.is_valid:
                l_poly.append(poly)
            else: 
                validpoly = get_valid_polygon(poly.exterior)            
                if validpoly.area > 0 :
                    if validpoly.geom_type == 'MultiPolygon':
                        for poly_p in validpoly.geoms: 
                            l_poly.append(poly_p)
                    else:
                        l_poly.append(validpoly)
    elif geo_contour.geom_type == "Polygon":
        l_poly.append(get_valid_polygon(geo_contour.exterior))
    else:
        raise(ValueError("Case not handled for geomety of type %s"%geo_contour.geom_type))
    valid_multipolygon = sh.geometry.asMultiPolygon(l_poly)
    if valid_multipolygon.is_valid:
        return valid_multipolygon 
    else:
        "It should be cause by the fact that some polygons intersect. So changing them."
        temp_poly = sh.geometry.Polygon([])
        for poly in l_poly: 
            temp_poly = temp_poly.union(poly)
        return temp_poly
    
def get_contour(ds,lat_name ="latitude",lon_name="longitude",levels=10,**kwargs):
    """
    get_contour [summary]
    
    Arguments:
        ds {DataArray} -- A 2D dataArray
    
    Keyword Arguments:
        lat_name {str} -- Dimension name for latitude (default: {"latitude"})
        lon_name {str} -- Dimension name for longitude (default: {"longitude"})
        levels {int,list} -- Number of levels or list of levels for contours  (default: {10})
    """
    if not isinstance(ds,xr.core.dataarray.DataArray): 
        raise(ValueError("In get_geojson_contour, input dataset should be a DataArray. Get :"%type(ds)))
        
    if len(ds.dims) != 2:
        raise(ValueError("Dataset should be 2D"))
        
    if lat_name not in ds.dims or lon_name not in ds.dims: 
        raise(ValueError("Latitude or longitude name are not present. Should get %s %s and get %s"%(lat_name,lon_name,ds.dims)))
    
    gv.extension("bokeh")  
    hv_ds = gv.Dataset(ds,[lon_name,lat_name])
    contours = gv.operation.contours(hv_ds,filled=True,levels=levels)
    
    polygon_list=list() 
    dict_poly = gu.polygons_to_geom_dicts(contours)    
    cmap = kwargs.get("cmap",cm.RdBu)
    mini = kwargs.get("mini",list(contours.data[0].values())[0])
    maxi = kwargs.get("maxi",list(contours.data[-1].values())[0])

    for i in range(len(dict_poly)):
        list_poly=[]
        for holes in dict_poly[i]["holes"]: 
            l_p = [sh.geometry.Polygon(x) for x in holes]
            if len(l_p)>0:
                list_poly.extend(l_p)
        if len(list_poly):
            mp_holes = sh.geometry.MultiPolygon(list_poly)
            mp_init = dict_poly[i]["geometry"]
            if not mp_init.is_valid: 
                mp_init = change_invalid_polygon(mp_init)
            if not mp_holes.is_valid: 
                mp_holes = change_invalid_polygon(mp_holes)   
            mp_final = mp_init - mp_holes
        else:
            if not dict_poly[i]["geometry"].is_valid:
                mp_final = change_invalid_polygon(dict_poly[i]["geometry"])
            else:
                mp_final = dict_poly[i]["geometry"]
        
        if kwargs.get("qualitative",False) and not mp_final.is_empty:
            buffer_arg = kwargs.get("buffer",5e-4)
            mp_temp = mp_final.buffer(-buffer_arg).buffer(1.1*buffer_arg)    
            if mp_temp.area > 0:
                mp_diff = (mp_final - mp_temp)
                if mp_diff.area > 0 :
                    mp_final = mp_final - mp_diff
            else:
                mp_final = sh.geometry.Polygon([])
        
        ## On stock le resultat dans un geojson 
        if not mp_final.is_empty:
            try:
                res = Feature( geometry = mp_final)
            except Exception: 
                print("buffering anyway")
                mp_temp = mp_final.buffer(-1e-5).buffer(1.1*1e-5)
                res = Feature(geometry = mp_temp)
 
            if isinstance(levels,Iterable): 
                value = list(contours.data[i].values())[0]
                descending_list = np.sort(levels)[::-1]
                bound_min = descending_list[np.argmax(descending_list < value)]
                bound_max = levels[np.argmax(levels >=value)]
  

                res["properties"] = {
                    "value_min":bound_min*1.0,
                    "value_max":bound_max*1.0,
                    "units":ds.attrs.get('units'),
                    "name":ds.attrs.get("long_name",ds.name)
                }
            else:
                res["properties"] = {
                    "value":list(contours.data[i].values())[0],
                    "units":ds.attrs.get('units'),
                    "name":ds.attrs.get("long_name",ds.name)
                    }
        
            res["properties"]["cmap"] = {
                    "value":list(contours.data[i].values())[0],
                    "mini":mini*1.0,
                    "maxi":maxi*1.0,
                 }
            res["properties"]["style"] = {
                "fillColor": rgb2hex(cmap((list(contours.data[i].values())[0]-mini)/(maxi-mini))),
                "fillOpacity":0.9,
                "opacity": 0.2,
                "color": "#000000",
                "dashArray": '5',
                "weight": 2,
                }
            polygon_list.append(res)
        else:
            print("Empty polygon for ",list(contours.data[i].values())[0])
            
    feature_collection = FeatureCollection(polygon_list)
    return feature_collection 

def generate_colorbar(geo_contour,colorbar_title):
    from matplotlib.colors import ListedColormap
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    l_color = []
    l_values = [] 
    for contour in geo_contour["features"]: 
        l_color.append(contour["properties"]["style"]["fillColor"])
        l_values.append(contour["properties"]["value"])
    fig = plt.figure(figsize = (2,10))
    ax1 = fig.add_axes([0.05, 0.12, 0.2, 0.87])
    newcmp = ListedColormap(l_color) 
    N = len(l_color)
    bounds=np.linspace(0,N,N+1)
    ticks_new=bounds[:-1]+0.5 
    cb=mpl.colorbar.ColorbarBase(ax1,cmap=newcmp,boundaries=bounds,ticks=ticks_new)      
    cb.ax.set_yticklabels(l_values)
    cb.ax.tick_params(labelsize=13)
    plt.xticks(rotation=45)
    plt.savefig(colorbar_title)
    plt.close()