#!/usr/bin/env python
# coding: utf-8

# this notebook is a demo of the MeteoNet dataset and is part of the DataViz challenge
# the notebook contains two sections :
# the first one will convert ground station data to netcdf format
# the second one will create an interactive mini-application that allows an unser to select a ground station and compare the actual data with the data predicted by the model.
# 

# In[ ]:


import matplotlib.pyplot as plt
import datetime as dt
import xarray as xr
import numpy as np
from datetime import datetime
import dask.dataframe as dd
import time
import plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objects as go
import pandas as pd


# # 1 Section convert to netcdf

# this cell is an example that will take the month of Febuary 2016 and convert it from its csv file to netcdf

# In[ ]:


# In[2]:


zone, year = 'NW','2016'
fname = '/kaggle/input/meteonet/'+zone+"_Ground_Stations/"+zone+"_Ground_Stations/"+zone+"_Ground_Stations_"+year+".csv"
df = pd.read_csv(fname,parse_dates=[4],infer_datetime_format=True)
df


# In[3]:


df["lat"].isnull().sum()


#openinng the csv file and filter it for year 2016 and region Notrh-West

# In[4]:


df2 = df.drop_duplicates(subset=["number_sta","date"],keep="last",inplace=False)


# In[5]:


df2

#df2 is a dataframe where we brutally removed the duplicates entries

# In[6]:


df2["lat"].isnull().sum()


# In[7]:


df2_filter=df2[(df2["date"]>="2016-02-01") & (df2["date"]<"2016-03-01")]
df2_filter


# In[8]:


df2_filter["lat"].isnull().sum()


# we only keep the month Febuary

# In[9]:


df_reindex=df2_filter.set_index(["number_sta","date"])
df_reindex


# In[10]:


df_reindex["lat"].isnull().sum()

#reset index on station number and date

# In[11]:


dd = df2_filter[(df2_filter["number_sta"]!=28239002)&
            (df2_filter["number_sta"]!=36128001)&
            (df2_filter["number_sta"]!=56240003)&
            (df2_filter["number_sta"]!=86027001)]


# In[12]:


dd[dd["number_sta"]==28239002]


# In[13]:


dd=dd.set_index(["number_sta","date"])
dd


# In[14]:


xarr = dd.to_xarray()
xarr


# In[15]:


def build_stations_list():
    return xarr["number_sta"].values
import math
def build_lat_long():
    for stanum in build_stations_list():
        lat = xarr.sel(number_sta=stanum)["lat"].values[0]
        lon = xarr.sel(number_sta=stanum)["lon"].values[0]
        if math.isnan(lat) or math.isnan(lon):
            print("found nan in station "+str(stanum))
    print("ok")
build_lat_long()


# we build the xarray

# In[16]:


xarr.to_netcdf('2016-02.netcdf')


# In[ ]:


#open custom netcdf
c_ncdf = xr.open_dataset("/kaggle/input/201602/2016-02.netcdf")
#c_ncdf.sel(number_sta=14066001,date=slice(datetime.strptime("2016-02-01","%Y-%m-%d"),datetime.strptime("2016-02-02","%Y-%m-%d")))


# In[ ]:


def __convert_to_datetime(d):
    '''convert a date'''
    return datetime.strptime(np.datetime_as_string(d,unit='s'), '%Y-%m-%dT%H:%M:%S')


# In[ ]:


def getDataSets(date,station,variable):
    '''returns sensor data, prediction values for a date, a station id and one variable given'''
    #get variable
    if variable == "temperature":
        v1 = "t"
        level = "2m"
        param2 = "t2m"
    elif variable == "wind_speed":
        v1 = "ff"
        level = "10m"
        param2 = "ws"
    elif variable == "wind_direction":
        v1 = "dd"
        level = "10m"
        param2= "p3031"
    elif variable == "precipitation":
        v1 = "precip"
        level = "PRECIP"
        param2 = "tp"
    elif variable == "humidity":
        v1 = "hu"
        level = "2m"
        param2 = "r"
    elif variable == "dew_point":
        v1 = "td"
        level = "2m"
        param2 = "d2m"
    elif variable == "pression":
        v1 = "psl"
        level = "P_sea_level"
        param2 = "msl"
    else:
        raise Exception("wrong parameter")
    
    c_ncdf = xr.open_dataset("/kaggle/input/201602/2016-02.netcdf")
    ground_station = c_ncdf.sel(number_sta=station,date=slice(date,date+dt.timedelta(days=1)))[v1].values
    gs_dates = c_ncdf.sel(number_sta=station,date=slice(date,date+dt.timedelta(days=1)))["date"].values
    
    zone = "NW"
    model = "arome"
    MODEL = "AROME"
    
    #open weather model
    directory = '/kaggle/input/meteonet/' + zone + '_weather_models_2D_parameters_' + str(date.year) + str(date.month).zfill(2) + '/' + str(date.year) + str(date.month).zfill(2) + '/'
    fname = directory + f'{MODEL}/{level}/{model}_{level}_{zone}_{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}000000.nc'
    data = xr.open_dataset(fname)
    
    l_lat = data["latitude"].values
    l_lon = data["longitude"].values
   
    
    lat=c_ncdf.sel(number_sta=station)["lat"].values[0]
    lon = c_ncdf.sel(number_sta=station)["lon"].values[0]
    steplat = l_lat[1]-l_lat[0]
    steplon = l_lon[1]-l_lon[0]
    series = data.sel(latitude=slice(lat-steplat/2,lat+steplat/2),longitude=slice(lon-steplon/2,lon+steplon/2))[param2].values
    
    
    series_axis = data["valid_time"].values
    
    unit = data.sel(latitude=slice(lat-steplat/2,lat+steplat/2),longitude=slice(lon-steplon/2,lon+steplon/2))[param2].attrs["units"]
    
    #return ((ground_station,gs_dates),(series.reshape(25),series_axis))
    
    #convert dates
    a = []
    b = []
    
    for d in gs_dates:
        a.append(__convert_to_datetime(d))
        
    for d in series_axis:
        b.append(__convert_to_datetime(d))
    
    return {"donnees_capteur":(ground_station,a),"prediction":(series.reshape(25),b),"unit":unit}


# measure the time taken by one call of "getDataSets" (which includes opening netcdf files)

# In[ ]:


t1 = time.time() 
t = getDataSets(datetime.strptime("2016-02-01","%Y-%m-%d"),14066001,"temperature")
t2 = time.time()
print ("time to open"+str(t2-t1))
t


# In[ ]:


import ipywidgets as w


# cree une figure et deux des trois widgets

# create a figure with two of the three widgets

# In[ ]:


fig = go.FigureWidget()
c1 = w.DatePicker(description="select the date")
#c2 = w.IntText(description="station id")
c3 = w.Dropdown(options=["temperature","wind_speed","wind_direction","precipitation","humidity","dew_point","pression"])


# In[ ]:


def plot(date,station,variable):
    '''replot the graph with respect to the given parameters'''
    d = getDataSets(date,station,variable)
    
    (a,b) = d["donnees_capteur"]
    (c,e) = d["prediction"]
    
    unit = d["unit"]
    
    t1 = go.Scatter(x=b,y=a,name="sensor data")
    t2 = go.Scatter(x=e,y=c,name="prediction")
    fig.data=[]
    fig.add_trace(t1)
    fig.add_trace(t2)
    fig.update_layout(title="comparing the sensor data with the prediction model for the day",
                     yaxis_title=unit)


# In[ ]:


def build_stations_list():
    '''build the station list'''
    data = xr.open_dataset("/kaggle/input/201602/2016-02.netcdf")
    return data["number_sta"].values

c2 = w.Dropdown(options=build_stations_list(),description="select the station")


# In[ ]:


class p:
    #class that contains plot parameteres
    date = datetime.strptime("01-02-2016","%d-%m-%Y")
    station = 14066001
    variable = "temperature"
    
class dateObject:
    #default datepicker settings
    year=2016
    month=2
    day=1

    
#the three widgets listeners
def hook_c1(change):
    print(change)
    try:
        date = change["new"]["value"]
        p.date = datetime(int(date["year"]),int(date["month"])+1,int(date["date"]))
        plot(p.date,p.station,p.variable)
    except Exception:
        pass
def hook_c2(change):
    try:
        ar = build_stations_list()
        p.station = ar[change["new"]["index"]]
        #p.station = change["new"]["value"]
        plot(p.date,p.station,p.variable)
    except Exception:
        pass
def hook_c3(change):
    ar = ["temperature","wind_speed","wind_direction","precipitation","humidity","dew_point","pression"]
    p.variable = ar[change["new"]["index"]]
    plot(p.date,p.station,p.variable)
    
c1.observe(hook_c1)
c2.observe(hook_c2)
c3.observe(hook_c3)
c1.value = dateObject
plot(p.date,p.station,p.variable)


# In[ ]:


wbox = w.VBox([c1,c2,c3,fig])
wbox


# # 2 Section IpyLeaflet 

# hack ipyleaflet

# In[ ]:


get_ipython().system('pip install geoviews --upgrade')
get_ipython().system('pip install ipyleaflet==0.12.3')
get_ipython().system('jupyter nbextension enable --py --sys-prefix ipyleaflet --user')


# refresh the notebook

# testing ipyleaflet. If the hack worked, you should see a map

# In[ ]:


import ipyleaflet as ipyl
from ipyleaflet import Map, Marker, MarkerCluster
mapi = ipyl.Map(center=(45,0),zoom=4)
mapi


# In[ ]:


stations_list = build_stations_list()
stations_list


# In[ ]:


import math
def build_lat_long():
    '''build without returning it a list of coordinates but raise an exception if it has found nan values'''
    for stanum in build_stations_list():
        lat = c_ncdf.sel(number_sta=stanum)["lat"].values[0]
        lon = c_ncdf.sel(number_sta=stanum)["lon"].values[0]
        if math.isnan(lat) or math.isnan(lon):
            raise Exception("nan found")
    print("ok")
build_lat_long()


# In[ ]:


class MyMark(Marker):
    '''Custom marker that, when clicked, update the station id in the plot'''
    def __init__(self,sta_num):
        self.sta_num = sta_num
        lat = c_ncdf.sel(number_sta=sta_num)["lat"].values[0]
        lon = c_ncdf.sel(number_sta=sta_num)["lon"].values[0]
        super().__init__(location=(lat,lon),draggable=False)
        super().on_click(self.printsta)
    def printsta(self,*args,**kwargs):
        p.station = self.sta_num
        c2.value = p.station
        plot(p.date,p.station,p.variable)


# In[ ]:


def build_marker_list():
    '''build a cluster of markers'''
    c_ncdf = xr.open_dataset("/kaggle/input/201602/2016-02.netcdf")
    station_list = build_stations_list()
    m_list = []
    for stanum in stations_list:
        lat = c_ncdf.sel(number_sta=stanum)["lat"].values[0]
        lon = c_ncdf.sel(number_sta=stanum)["lon"].values[0]
        marker = MyMark(stanum)
        m_list.append(marker)
        
    m_list=tuple(m_list)
    print(m_list)
    return MarkerCluster(markers=m_list)


# In[ ]:


ml = build_marker_list()


# add the marker cluster to the map

# In[ ]:


mapi.add_layer(ml)
mapi


# put them together

# In[ ]:


wboxv2 = w.VBox([mapi,wbox])
wboxv2


# In[ ]:




