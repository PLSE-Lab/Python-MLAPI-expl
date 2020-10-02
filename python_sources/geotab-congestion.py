#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import math
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import shapely.geometry as geom
import geopandas as gpd

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
    #for filename in filenames:
    #    print(os.path.join(dirname, filename))

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


get_ipython().system('pip install git+https://github.com/jonbarron/robust_loss_pytorch')
get_ipython().system('pip install --user torchcontrib')
get_ipython().system('cp -r /kaggle/input/optimizers/lookahead.pytorch-master/* /kaggle/working/lookahead')
get_ipython().system('cp -r /kaggle/input/optimizers/RAdam-master/* /kaggle/working/RAdam')


# In[ ]:


df_train = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/train.csv')
df_test = pd.read_csv('/kaggle/input/bigquery-geotab-intersection-congestion/test.csv')
df_train = reduce_mem_usage(df_train)
df_test = reduce_mem_usage(df_test)


# In[ ]:


list(df_train.columns.values)


# In[ ]:


df_train.fillna("Unknown",inplace=True)
df_test.fillna("Unknown",inplace=True)


# ## Create UID Intersection

# In[ ]:


df_train["IntersectionUID"] = df_train["City"] + df_train["IntersectionId"].astype(str)
df_test["IntersectionUID"] = df_test["City"] + df_test["IntersectionId"].astype(str)


# In[ ]:


df_train.sample(5)


# In[ ]:


df_test.sample(5)


# In[ ]:


#df_train["IntersectionUID"] = df_train[["City", "IntersectionId"]].apply(lambda row: row.City+str(row.IntersectionId), axis=1)
#df_test["IntersectionUID"] = df_test[["City", "IntersectionId"]].apply(lambda row: row.City+str(row.IntersectionId), axis=1)
uid_dict=pd.DataFrame(data=range(len(df_train["IntersectionUID"].unique())), index=df_train["IntersectionUID"].unique()).to_dict()[0]
df_train["IntersectionUID"] = df_train["IntersectionUID"].map(uid_dict)
df_test["IntersectionUID"] = df_test["IntersectionUID"].map(uid_dict)
df_test["IntersectionUID"].fillna(int(4796), inplace=True)
df_test["IntersectionUID"] = df_test["IntersectionUID"].astype("int32")


# In[ ]:


print(len(df_train["IntersectionUID"].unique()))
print(df_train["IntersectionUID"].unique())
print(len(df_test["IntersectionUID"].unique()))
print(df_test["IntersectionUID"].unique())


# ## Create Street ID

# In[ ]:


street_uid_dict=pd.DataFrame(data=range(len(df_train["EntryStreetName"].append(df_train["ExitStreetName"]).unique())), index=df_train["EntryStreetName"].append(df_train["ExitStreetName"]).unique()).to_dict()[0]
df_train["EntryStreetID"] = df_train["EntryStreetName"].map(street_uid_dict)
df_test["EntryStreetID"] = df_test["EntryStreetName"].map(street_uid_dict)
df_train["ExitStreetID"] = df_train["ExitStreetName"].map(street_uid_dict)
df_test["ExitStreetID"] = df_test["ExitStreetName"].map(street_uid_dict)
df_test["EntryStreetID"].fillna(int(1), inplace=True)
df_test["ExitStreetID"].fillna(int(1), inplace=True)
df_train["ExitStreetID"] = df_train["ExitStreetID"].astype(np.int32)
df_test["ExitStreetID"] = df_test["ExitStreetID"].astype(np.int32)
df_train["EntryStreetID"] = df_train["EntryStreetID"].astype(np.int32)
df_test["EntryStreetID"] = df_test["EntryStreetID"].astype(np.int32)


# ## Add Street Type

# In[ ]:


print(len(df_train["EntryStreetName"]))
#print(df_train[~df_train["ExitStreetName"].str.contains("Street|St|Avenue|Ave|Boulevard|Road|Drive|Lane|Tunnel|Highway|Way|Parkway|Parking|Oval|Unknown|Square|Place|Bridge", regex=True)]["ExitStreetName"])
#print(df_train[~df_train["EntryStreetName"].str.contains("Street|St|Avenue|Ave|Boulevard|Road|Drive|Lane|Tunnel|Highway|Way|Parkway|Parking|Oval|Unknown|Square|Place|Bridge", regex=True)]["EntryStreetName"])
#print(df_train[df_train["EntryStreetName"].str.contains("Place", regex=True)]["EntryStreetName"])

street_type = ["Street", "St", "Avenue", "Ave", "Boulevard", "Road", "Drive", "Lane",
              "Tunnel", "Highway", "Way", "Parkway", "Parking", "Oval", "Square",
              "Place", "Bridge", "Unknown"]

street_type_num = [0, 0, 1, 1, 2, 3, 4, 5,
                     6, 7, 8, 9, 9, 10, 11,
                     12, 13, 14]

def get_street_type(row, column):
    
    for i, s in enumerate(street_type):
        if s in row[column]:
            return street_type_num[i]
    return 15

def extract_street_type(df, column):
    df[column.replace("StreetName", "Type")] = df.apply(lambda row: get_street_type(row, column), axis=1)


# In[ ]:


extract_street_type(df_train, "EntryStreetName")
extract_street_type(df_train, "ExitStreetName")
extract_street_type(df_test, "EntryStreetName")
extract_street_type(df_test, "ExitStreetName")
df_train.head()


# ## Add Turn direction

# In[ ]:


directions={"E":0, "SE":1, "S":2, "SW":3, "W":4, "NW":5, "N":6, "NE":7,
           0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7}

df_train["EntryHeading"] = df_train["EntryHeading"].map(directions).astype(np.int32)
df_train["ExitHeading"] = df_train["ExitHeading"].map(directions).astype(np.int32)

df_test["EntryHeading"] = df_test["EntryHeading"].map(directions).astype(np.int32)
df_test["ExitHeading"] = df_test["ExitHeading"].map(directions).astype(np.int32)

def calc_turn(dataframe):
    dataframe["RightTurn"] = ((dataframe["EntryHeading"] - dataframe["ExitHeading"]) % 8 == 2).astype(np.int32)
    dataframe["LeftTurn"] = ((dataframe["EntryHeading"] - dataframe["ExitHeading"]) % 8 == 6).astype(np.int32)
    dataframe["PassThru"] = ((dataframe["EntryHeading"] - dataframe["ExitHeading"]) % 8 == 4).astype(np.int32)
    dataframe["UTurn"] = (dataframe["EntryHeading"] - dataframe["ExitHeading"] == 0).astype(np.int32)
    
    dataframe["RightSide"] = ((dataframe["ExitHeading"] - dataframe["EntryHeading"]) % 8 > 5).astype(np.int32)
    dataframe["LeftSide"] = ((dataframe["ExitHeading"] - (dataframe["EntryHeading"] + 1)) % 8 < 3).astype(np.int32)
    dataframe["Direction"] = ((dataframe["ExitHeading"] - (dataframe["EntryHeading"])) % 8).astype(np.int32)

calc_turn(df_train)
calc_turn(df_test)


# In[ ]:


df_train[["EntryHeading", "ExitHeading", "RightTurn", "LeftTurn", "PassThru", "UTurn", "RightSide", "LeftSide", "Direction"]].sample(10)


# ## Rush Hours

# In[ ]:


df_train["RushHour1"] = 8 - df_train["Hour"]
df_train["RushHour2"] = 17 - df_train["Hour"]
df_test["RushHour1"] = 8 - df_test["Hour"]
df_test["RushHour2"] = 17 - df_test["Hour"]


# In[ ]:


df_train.sample(10)


# ## Add Climate Data [1](https://www.weather-us.com/en/)

# In[ ]:


df_high_temp = pd.DataFrame({"Atlanta":[52.3, 56.6, 64.6, 72.5, 79.9, 86.4, 89.1, 88.1, 82.2, 72.7, 63.6, 54],
                              "Boston":[35.8, 38.7, 45.4, 55.6, 66.0, 75.9, 81.4, 79.6, 72.4, 61.4, 51.5, 41.2],
                              "Chicago": [31.5, 35.8, 46.8, 59.2, 70.2, 79.9, 84.2, 82.1, 75.3, 62.8, 48.6, 35.3],
                              "Philadelphia":[40.3, 43.8, 52.7, 63.9, 73.8, 82.7, 87.1, 85.3, 78.0, 66.6, 56.0, 44.8]})

df_low_temp = pd.DataFrame({"Atlanta":[34.3, 37.7, 44.1, 51.5, 60.3, 68.2, 71.3, 70.7, 64.8, 54.0, 44.5, 36.5],
                           "Boston":[22.2, 24.7, 31.1, 40.6, 49.9, 59.5, 65.4, 64.6, 57.4, 46.5, 38.0, 28.2],
                           "Chicago":[18.1, 21.7, 30.9, 41.7, 51.6, 62.1, 67.5, 66.2, 57.5, 45.7, 34.5, 22.7],
                           "Philadelphia":[25.6, 27.7, 33.4, 44.1, 54.0, 62.8, 69.2, 67.9, 60.3, 48.4, 39.2, 30.1]})

df_snowfall = pd.DataFrame({"Atlanta":[1.3, 0.4, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4],
                              "Boston":[12.9, 10.9, 7.8, 1.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.3, 9.0],
                              "Chicago":[11.5, 9.1, 5.4, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.3, 8.7],
                              "Philadelphia":[6.5, 8.8, 2.9, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 3.4]})
                                                                                   
df_rainfall = pd.DataFrame({"Atlanta":[4.2, 4.7, 4.8, 3.4, 3.7, 4.0, 5.3, 3.9, 4.5, 3.4, 4.1, 3.9],
                              "Boston":[3.4, 3.3, 4.3, 3.7, 3.5, 3.7, 3.4, 3.4, 3.4, 3.9,4.0, 3.8],
                              "Chicago":[2.1, 1.9, 2.7, 3.6, 4.1, 4.1, 4.0, 4.0, 3.3, 3.2, 3.4, 2.6],
                              "Philadelphia":[3.0, 2.7, 3.8, 3.6, 3.7, 3.4, 4.4, 3.5, 3.8, 3.2, 3.0, 3.6]})
                                                                                   
df_daylight = pd.DataFrame({"Atlanta":[10, 11, 12, 13, 14, 14, 14, 13, 12, 11, 10, 10],
                             "Boston":[9, 11, 12, 13, 15, 15, 15, 14, 12, 11, 10, 9],
                             "Chicago":[10, 11, 12, 13, 15, 15, 15, 14, 12, 11, 10, 9],
                             "Philadelphia":[10, 11, 12, 13, 14, 15, 15, 14, 12, 11, 10, 9]})         
                                                                                  
df_sunshine = pd.DataFrame({"Atlanta":[5.3, 6.1, 7.1, 8.7, 9.3, 9.5, 8.8, 8.3, 7.6, 7.7, 6.2, 5.3], 
                             "Boston":[5.3, 6.0, 6.9, 7.6, 8.6, 9.6, 9.7, 8.9, 7.9, 6.7, 4.8, 4.6],
                             "Chicago":[4.4, 4.9, 6.0, 7.2, 9.1, 10.4, 10.3, 9.1, 7.6, 6.2, 3.6, 3.4],  
                             "Philadelphia":[5.0, 5.5, 6.5, 7.2, 7.9, 9.0, 8.9, 8.4, 7.3, 6.6, 5.2, 4.4]})


# In[ ]:


def get_climate_dict(df):
    df["Index"] = range(1, 13)
    melt_df = pd.melt(df, value_vars=["Atlanta", "Boston", "Chicago", "Philadelphia"], id_vars="Index")
    melt_df["Index"] = melt_df["variable"] + melt_df["Index"].astype(str)
    melt_df = melt_df.drop(columns=["variable"])
    melt_df = melt_df.set_index("Index")
    return melt_df.to_dict()["value"]


# In[ ]:


high_temp_dict = get_climate_dict(df_high_temp)
low_temp_dict = get_climate_dict(df_low_temp)
snowfall_dict = get_climate_dict(df_snowfall)
rainfall_dict = get_climate_dict(df_rainfall)
daylight_dict = get_climate_dict(df_daylight)
sunshine_dict = get_climate_dict(df_sunshine)


# In[ ]:


def get_climate(df, climate_dict, column):
    df[column] = df["City"] + df["Month"].astype(str)
    df[column] = df[column].map(climate_dict)


# In[ ]:


get_climate(df_train, high_temp_dict, "HighTemp")
get_climate(df_train, low_temp_dict, "LowTemp")
get_climate(df_train, snowfall_dict, "SnowFall")
get_climate(df_train, rainfall_dict, "RainFall")
get_climate(df_train, daylight_dict, "DayLight")
get_climate(df_train, sunshine_dict, "SunShine")

get_climate(df_test, high_temp_dict, "HighTemp")
get_climate(df_test, low_temp_dict, "LowTemp")
get_climate(df_test, snowfall_dict, "SnowFall")
get_climate(df_test, rainfall_dict, "RainFall")
get_climate(df_test, daylight_dict, "DayLight")
get_climate(df_test, sunshine_dict, "SunShine")


# ## School break

# In[ ]:


df_train["SummerBreak"] = ((df_train["Month"] >= 6) & (df_train["Month"] <= 8)).astype(int)
df_test["SummerBreak"] = ((df_test["Month"] >= 6) & (df_test["Month"] <= 8)).astype(int)
df_train["WinterBreak"] = (df_train["Month"] == 12).astype(int)
df_test["WinterBreak"] = (df_test["Month"] == 12).astype(int)


# ## Distance to City Center

# In[ ]:


cities={"Atlanta":0, "Boston":1, "Chicago":2, "Philadelphia":3, 0:0, 1:1, 2:2, 3:3}

df_train["City"] = df_train["City"].map(cities).astype(np.int32)
df_test["City"] = df_test["City"].map(cities).astype(np.int32)


# In[ ]:


def get_geo_dist(df_train, df_test):
    
    MAX_DIR = 0.002
    
    def get_dist_dict(atlanta_df, boston_df, chicago_df, philadelphia_df, column):
    
        atlanta_dict = atlanta_df[["IntersectionUID", column]].set_index("IntersectionUID").to_dict()[column]
        boston_dict = boston_df[["IntersectionUID", column]].set_index("IntersectionUID").to_dict()[column]
        chicago_dict = chicago_df[["IntersectionUID", column]].set_index("IntersectionUID").to_dict()[column]
        philadelphia_dict = philadelphia_df[["IntersectionUID", column]].set_index("IntersectionUID").to_dict()[column]

        dist_dict = {**atlanta_dict, **boston_dict}
        dist_dict = {**dist_dict, **chicago_dict}
        dist_dict = {**dist_dict, **philadelphia_dict}
        
        return dist_dict
    
    def min_dist(point1, point2):
        
        dist = point2.distance(point1)
        min_dist = (dist.sort_values().reset_index()).iloc[1,1]        
        return min_dist
    
    def min_dir(point1, point2, dir):

        if dir == 0:
            #dir_vector = Point(1, 0)
            dir_point2 = point2[point2.x > point1.x]
        elif dir == 1:
            #dir_vector = Point(-1, 0)
            dir_point2 = point2[point2.x < point1.x]
        elif dir == 2: 
            #dir_vector = Point(0, 1)
            dir_point2 = point2[point2.y > point1.y]
        elif dir == 3:
            #dir_vector = Point(0, -1)
            dir_point2 = point2[point2.y < point1.y]
        elif dir == 4:
            #dir_vector = Point(1, 1)
            dir_point2 = point2[point2.y < point1.y]
        elif dir == 5:
            #dir_vector = Point(1, -1)
            dir_point2 = point2[point2.y < point1.y]
        elif dir == 6:
            #dir_vector = Point(-1, -1)
            dir_point2 = point2[point2.y < point1.y]
        elif dir == 7:
            #dir_vector = Point(-1, 1)
            dir_point2 = point2[point2.y < point1.y]
        else:
            exit(1)
        
        #vector=(point2 - point1)
        #cos_similarity = (vector * dir_vector) / (vector * vector + dir_vector * dir_vector)
        #dir_point2 = point2[cos_similarity > 0.8]
        
        if len(dir_point2) == 0:
            dir_dist = MAX_DIR
        else:
            dir_dist = dir_point2.distance(point1)
            dir_dist = (dir_dist.sort_values().reset_index()).iloc[0, 1]
            dir_dist = min(MAX_DIR, dir_dist)
            
        return dir_dist
    
    def min_inter(point1, point2, dir):
        if dir == 0:
            #dir_vector = Point(1, 0)
            dir_point2 = point2[point2.x > point1.x]
        elif dir == 1:
            #dir_vector = Point(-1, 0)
            dir_point2 = point2[point2.x < point1.x]
        elif dir == 2: 
            #dir_vector = Point(0, 1)
            dir_point2 = point2[point2.y > point1.y]
        elif dir == 3:
            #dir_vector = Point(0, -1)
            dir_point2 = point2[point2.y < point1.y]
        elif dir == 4:
            #dir_vector = Point(1, 1)
            dir_point2 = point2[point2.y < point1.y]
        elif dir == 5:
            #dir_vector = Point(1, -1)
            dir_point2 = point2[point2.y < point1.y]
        elif dir == 6:
            #dir_vector = Point(-1, -1)
            dir_point2 = point2[point2.y < point1.y]
        elif dir == 7:
            #dir_vector = Point(-1, 1)
            dir_point2 = point2[point2.y < point1.y]
        else:
            exit(1)

        #vector=(point2 - point1)
        #cos_similarity = (vector * dir_vector) / (vector * vector + dir_vector * dir_vector)
        #dir_point2 = point2[cos_similarity > 0.8]

        if len(dir_point2) == 0:
            dir_inter = 4096
        else:
            dir_dist = dir_point2.distance(point1)
            dir_dist = dir_dist.sort_values().reset_index().iloc[0, 1]

            if dir_dist > MAX_DIR:
                dir_inter = 4096
            else:
                dir_iter = 0

        return dir_inter

    def apply_map(df, dictionary, column):
        
        df[column] = df["IntersectionUID"]
        df[column] = df[column].map(dictionary)
    
    df = pd.concat((df_train[["IntersectionUID", "Latitude", "Longitude", "City"]],                     df_test[["IntersectionUID", "Latitude", "Longitude", "City"]]), axis=0)

    df_geo = df.drop_duplicates(subset="IntersectionUID")
    df_geo = gpd.GeoDataFrame(df_geo, geometry=gpd.points_from_xy(df_geo.Latitude, df_geo.Longitude))

    atlanta_df = df_geo.loc[df_geo["City"] == 0]
    boston_df = df_geo.loc[df_geo["City"] == 1]
    chicago_df = df_geo.loc[df_geo["City"] == 2]
    philadelphia_df = df_geo.loc[df_geo["City"] == 3]
    
    atlanta_df["CenterDist"] = atlanta_df.geometry.distance(geom.Point(33.753746, -84.386330))
    boston_df["CenterDist"] = boston_df.geometry.distance(geom.Point(42.361145, -71.057083))
    chicago_df["CenterDist"] = chicago_df.geometry.distance(geom.Point(41.881832, -87.623177))
    philadelphia_df["CenterDist"] = philadelphia_df.geometry.distance(geom.Point(39.952583, -75.165222))

    center_dist_dict = get_dist_dict(atlanta_df, boston_df, chicago_df, philadelphia_df, "CenterDist")
    apply_map(df_train, center_dist_dict, "CenterDist")
    apply_map(df_test, center_dist_dict, "CenterDist")
    
    atlanta_df["MinDist"] = atlanta_df.geometry.apply(min_dist, args=(atlanta_df.geometry, ))
    boston_df["MinDist"] = boston_df.geometry.apply(min_dist, args=(boston_df.geometry, ))
    chicago_df["MinDist"] = chicago_df.geometry.apply(min_dist, args=(chicago_df.geometry, ))
    philadelphia_df["MinDist"] = philadelphia_df.geometry.apply(min_dist, args=(philadelphia_df.geometry, ))

    for i, column in enumerate(["NDist", "SDist", "WDist", "EDist"]):
        atlanta_df[column] = atlanta_df.geometry.apply(min_dir, args=(atlanta_df.geometry, i, ))
        boston_df[column] = boston_df.geometry.apply(min_dir, args=(boston_df.geometry, i, ))
        chicago_df[column] = chicago_df.geometry.apply(min_dir, args=(chicago_df.geometry, i, ))
        philadelphia_df[column] = philadelphia_df.geometry.apply(min_dir, args=(philadelphia_df.geometry, i, ))

    min_dist_dict = get_dist_dict(atlanta_df, boston_df, chicago_df, philadelphia_df, "MinDist")
    n_dist_dict = get_dist_dict(atlanta_df, boston_df, chicago_df, philadelphia_df, "NDist")
    s_dist_dict = get_dist_dict(atlanta_df, boston_df, chicago_df, philadelphia_df, "SDist")
    w_dist_dict = get_dist_dict(atlanta_df, boston_df, chicago_df, philadelphia_df, "WDist")
    e_dist_dict = get_dist_dict(atlanta_df, boston_df, chicago_df, philadelphia_df, "EDist")
    
    apply_map(df_train, min_dist_dict, "MinDist")
    apply_map(df_test, min_dist_dict, "MinDist")
    
    apply_map(df_train, n_dist_dict, "NDist")
    apply_map(df_test, n_dist_dict, "NDist")
    
    apply_map(df_train, s_dist_dict, "SDist")
    apply_map(df_test, s_dist_dict, "SDist")
    
    apply_map(df_train, w_dist_dict, "WDist")
    apply_map(df_test, w_dist_dict, "WDist")
    
    apply_map(df_train, e_dist_dict, "EDist")
    apply_map(df_test, e_dist_dict, "EDist")


# In[ ]:


get_ipython().run_line_magic('time', 'get_geo_dist(df_train, df_test)')
df_train.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(12,12)) 
sns.heatmap(data=(df_train).corr(), square=True, ax=ax)


# In[ ]:


def count_plot(df, value, ax):
    sns.countplot(x=value, data=df, ax=ax)


# In[ ]:


fig = plt.figure(figsize = (20, 12)) # width x height
ax1 = fig.add_subplot(3, 3, 1) # row, column, position
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)
ax5 = fig.add_subplot(3, 3, 5)
ax6 = fig.add_subplot(3, 3, 6)
ax7 = fig.add_subplot(3, 3, 7)
ax8 = fig.add_subplot(3, 3, 8)
ax9 = fig.add_subplot(3, 3, 9)

sns.countplot(data=df_train, x="City", ax=ax1)
sns.countplot(data=df_train, x="Month", ax=ax2)
sns.countplot(data=df_train, x="Hour", ax=ax3)
sns.countplot(data=df_train, x="Weekend", ax=ax4)
sns.countplot(data=df_train, x="EntryHeading", ax=ax5)
sns.countplot(data=df_train, x="ExitHeading", ax=ax6)
sns.countplot(data=df_train, x="Direction", ax=ax7)
sns.countplot(data=df_train, x="EntryType", ax=ax8)
sns.countplot(data=df_train, x="ExitType", ax=ax9)


# In[ ]:


fig = plt.figure(figsize = (20, 12)) # width x height
ax1 = fig.add_subplot(3, 3, 1) # row, column, position
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)
ax5 = fig.add_subplot(3, 3, 5)
ax6 = fig.add_subplot(3, 3, 6)
ax7 = fig.add_subplot(3, 3, 7)
ax8 = fig.add_subplot(3, 3, 8)
ax9 = fig.add_subplot(3, 3, 9)

sns.countplot(data=df_test, x="City", ax=ax1)
sns.countplot(data=df_test, x="Month", ax=ax2)
sns.countplot(data=df_test, x="Hour", ax=ax3)
sns.countplot(data=df_test, x="Weekend", ax=ax4)
sns.countplot(data=df_test, x="EntryHeading", ax=ax5)
sns.countplot(data=df_test, x="ExitHeading", ax=ax6)
sns.countplot(data=df_test, x="Direction", ax=ax7)
sns.countplot(data=df_test, x="EntryType", ax=ax8)
sns.countplot(data=df_test, x="ExitType", ax=ax9)


# In[ ]:


def plot_6(df, plot):
    fig = plt.figure(figsize = (20,8)) # width x height
    ax1 = fig.add_subplot(2, 3, 1) # row, column, position
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    
    plot(df, "TotalTimeStopped_p20", ax1)
    plot(df, "TotalTimeStopped_p50", ax2)
    plot(df, "TotalTimeStopped_p80", ax3)
    plot(df, "DistanceToFirstStop_p20", ax4)
    plot(df, "DistanceToFirstStop_p50", ax5)
    plot(df, "DistanceToFirstStop_p80", ax6)


# In[ ]:


def plot_week_hour(df, value, ax):
    data = df.groupby(["Weekend", "Hour"]).mean()[value].unstack(level=0).reset_index().rename(columns={0:"Weekday", 1:"Weekend"}).set_index("Hour")
    sns.lineplot(data=data, ax=ax, legend="brief")
    ax.set(ylabel=value)
    ax.legend(loc=2)
    
plot_6(df_train, plot_week_hour)


# In[ ]:


def plot_month_hour(df, value, ax):
    data = df.groupby(["Month", "Hour"]).mean()[value].unstack(level=0).reset_index()
    data.fillna(0, inplace=True)
    data = data.melt(id_vars="Hour", value_vars=[1,5,6,7,8,9,10,11,12], value_name=value)
    #sns.catplot(x="Hour", y=value, hue="Month", data=data, ax=ax, kind="strip")
    sns.factorplot(x="Hour", y=value, hue="Month", data=data, ax=ax)
    ax.set(ylabel=value)
    ax.legend(loc=2)
    plt.close(2)
    
plot_6(df_train, plot_month_hour)


# In[ ]:


def plot_city_hour(df, value, ax):
    data = df.groupby(["City", "Hour"]).mean()[value].unstack(level=0).reset_index()
    data.fillna(0, inplace=True)
    data = data.melt(id_vars="Hour", value_vars=[0,1,2,3], value_name=value)
    #sns.catplot(x="Hour", y=value, hue="Month", data=data, ax=ax, kind="strip")
    sns.factorplot(x="Hour", y=value, hue="City", data=data, ax=ax)
    ax.set(ylabel=value)
    ax.legend(loc=2)
    plt.close(2)

plot_6(df_train, plot_city_hour)


# In[ ]:


def plot_direction_city(df, value, ax):
    data = df.groupby(["Direction", "City"]).mean()[value].unstack(level=0).reset_index()
    data.fillna(0, inplace=True)
    data = data.melt(id_vars="City", value_vars=[0,1,2,3,4,5,6,7], value_name=value)
    #sns.catplot(x="Hour", y=value, hue="Month", data=data, ax=ax, kind="strip")
    sns.factorplot(x="City", y=value, hue="Direction", data=data, ax=ax, kind="bar")
    ax.set(ylabel=value)
    ax.legend(loc=2)
    plt.close(2)

plot_6(df_train, plot_direction_city)


# ## Map

# In[ ]:


def plot_map_count(df, city):
    count=df.groupby(['City','Latitude','Longitude'])["IntersectionUID"].count().reset_index()
    
    fig = px.scatter_mapbox(count[count["City"]==city], 
                            lat="Latitude", lon="Longitude",size="IntersectionUID",size_max=10,
                            color="IntersectionUID", color_continuous_scale=px.colors.sequential.Inferno, zoom=11)
    fig.update_layout(mapbox_style="stamen-terrain")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()


# In[ ]:


#plot_map_count(df_train, 0)


# In[ ]:


#plot_map_count(df_test, 0)


# In[ ]:


#plot_map_count(df_train, 3)


# In[ ]:


#plot_map_count(df_test, 3)


# In[ ]:


def plot_map(city, feature, hour=17):
    TotalTimeStopped=df_train.groupby(['City','Latitude','Longitude', 'Hour'])[feature].mean().reset_index()
    TotalTimeStopped = TotalTimeStopped[TotalTimeStopped["Hour"]== hour]
    
    fig = px.scatter_mapbox(TotalTimeStopped[TotalTimeStopped["City"]==city], 
                            lat="Latitude", lon="Longitude",size=feature,size_max=10,
                            color=feature, color_continuous_scale=px.colors.sequential.Inferno, zoom=11)
    fig.update_layout(mapbox_style="stamen-terrain")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()


# In[ ]:


#plot_map(0, "DistanceToFirstStop_p80")


# In[ ]:


#plot_map(3, "DistanceToFirstStop_p80")


# In[ ]:


drop_features = ['IntersectionId', 'EntryStreetName', 'ExitStreetName', 'Path']
drop_unused_targets = ['TotalTimeStopped_p40', 'TotalTimeStopped_p60',
                        'TimeFromFirstStop_p20', 'TimeFromFirstStop_p40', 'TimeFromFirstStop_p50',
                        'TimeFromFirstStop_p60', 'TimeFromFirstStop_p80', 'DistanceToFirstStop_p40',
                         'DistanceToFirstStop_p60']

input_train = df_train.drop(drop_features + drop_unused_targets, axis=1)
input_test = df_test.drop(drop_features, axis=1)
print("Total      - {}".format(len(input_train)))
input_val = input_train.sample(frac=0.2, random_state=0)
input_train = input_train.drop(input_val.index)
print("Training   - {}".format(len(input_train)))
print("Validation - {}".format(len(input_val)))
print("Test       - {}".format(len(input_test)))


# In[ ]:


def min_max(df, min, max):
    #min = base_df.min()
    #max = base_df.max()

    return (df - min) / (max - min)

def z_score(df, base_df):
    mean = base_df.mean()
    std = base_df.std()
    return (df - mean) / std

def preprocess_data(input_train, input_val, input_test):

    #input_test["Latitude"] = min_max(input_test["Latitude"], input_train["Latitude"])
    #input_test["Longitude"] = min_max(input_test["Longitude"], input_train["Latitude"])
    #input_test["CenterDistance"] = min_max(input_test["CenterDistance"], input_train["Latitude"])

    #input_val["Latitude"] = min_max(input_val["Latitude"], input_train["Latitude"])
    #input_val["Longitude"] = min_max(input_val["Longitude"], input_train["Longitude"])
    #input_val["CenterDistance"] = min_max(input_val["CenterDistance"],input_train["CenterDistance"])

    #input_train["Latitude"] = min_max(input_train["Latitude"], input_train["Latitude"])
    #input_train["Longitude"] = min_max(input_train["Longitude"], input_train["Longitude"])
    #input_train["CenterDistance"] = min_max(input_train["CenterDistance"], input_train["CenterDistance"])
    
    input_test["Latitude"] = z_score(input_test["Latitude"], input_train["Latitude"])
    input_test["Longitude"] = z_score(input_test["Longitude"], input_train["Longitude"])
    input_test["CenterDist"] = z_score(input_test["CenterDist"], input_train["CenterDist"])
    input_test["MinDist"] = z_score(input_test["MinDist"], input_train["MinDist"])
    input_test["NDist"] = z_score(input_test["NDist"], input_train["NDist"])
    input_test["SDist"] = z_score(input_test["SDist"], input_train["SDist"])
    input_test["WDist"] = z_score(input_test["WDist"], input_train["WDist"])
    input_test["EDist"] = z_score(input_test["EDist"], input_train["EDist"])
    input_test["HighTemp"] = z_score(input_test["HighTemp"], input_train["HighTemp"])
    input_test["LowTemp"] = z_score(input_test["LowTemp"], input_train["LowTemp"])
    input_test["SnowFall"] = z_score(input_test["SnowFall"], input_train["SnowFall"])
    input_test["RainFall"] = z_score(input_test["RainFall"], input_train["RainFall"])
    input_test["DayLight"] = z_score(input_test["DayLight"], input_train["DayLight"])
    input_test["SunShine"] = z_score(input_test["SunShine"], input_train["SunShine"])

    input_val["Latitude"] = z_score(input_val["Latitude"], input_train["Latitude"])
    input_val["Longitude"] = z_score(input_val["Longitude"], input_train["Longitude"])
    input_val["CenterDist"] = z_score(input_val["CenterDist"], input_train["CenterDist"])
    input_val["MinDist"] = z_score(input_val["MinDist"], input_train["MinDist"])
    input_val["NDist"] = z_score(input_val["NDist"], input_train["NDist"])
    input_val["SDist"] = z_score(input_val["SDist"], input_train["SDist"])
    input_val["WDist"] = z_score(input_val["WDist"], input_train["WDist"])
    input_val["EDist"] = z_score(input_val["EDist"], input_train["EDist"])
    input_val["HighTemp"] = z_score(input_val["HighTemp"], input_train["HighTemp"])
    input_val["LowTemp"] = z_score(input_val["LowTemp"], input_train["LowTemp"])
    input_val["SnowFall"] = z_score(input_val["SnowFall"], input_train["SnowFall"])
    input_val["RainFall"] = z_score(input_val["RainFall"], input_train["RainFall"])
    input_val["DayLight"] = z_score(input_val["DayLight"], input_train["DayLight"])
    input_val["SunShine"] = z_score(input_val["SunShine"], input_train["SunShine"])
    
    input_train["Latitude"] = z_score(input_train["Latitude"], input_train["Latitude"])
    input_train["Longitude"] = z_score(input_train["Longitude"], input_train["Longitude"])
    input_train["CenterDist"] = z_score(input_train["CenterDist"], input_train["CenterDist"])
    input_train["MinDist"] = z_score(input_train["MinDist"], input_train["MinDist"])
    input_train["NDist"] = z_score(input_train["NDist"], input_train["NDist"])
    input_train["SDist"] = z_score(input_train["SDist"], input_train["SDist"])
    input_train["WDist"] = z_score(input_train["WDist"], input_train["WDist"])
    input_train["EDist"] = z_score(input_train["EDist"], input_train["EDist"])
    input_train["HighTemp"] = z_score(input_train["HighTemp"], input_train["HighTemp"])
    input_train["LowTemp"] = z_score(input_train["LowTemp"], input_train["LowTemp"])
    input_train["SnowFall"] = z_score(input_train["SnowFall"], input_train["SnowFall"])
    input_train["RainFall"] = z_score(input_train["RainFall"], input_train["RainFall"])
    input_train["DayLight"] = z_score(input_train["DayLight"], input_train["DayLight"])
    input_train["SunShine"] = z_score(input_train["SunShine"], input_train["SunShine"])
    
    input_train["Hour"] = min_max(input_train["Hour"], 0, 23)
    input_train["RushHour1"] = min_max(input_train["RushHour1"], -15, 8)
    input_train["RushHour2"] = min_max(input_train["RushHour2"], -6, 17)
    input_train["Month"] = min_max(input_train["Month"], 1, 12)
    
    input_val["Hour"] = min_max(input_val["Hour"], 0, 23)
    input_val["RushHour1"] = min_max(input_val["RushHour1"], -15, 8)
    input_val["RushHour2"] = min_max(input_val["RushHour2"], -6, 17)
    input_val["Month"] = min_max(input_val["Month"], 1, 12)
    
    input_test["Hour"] = min_max(input_test["Hour"], 0, 23)
    input_test["RushHour1"] = min_max(input_test["RushHour1"], -15, 8)
    input_test["RushHour2"] = min_max(input_test["RushHour2"], -6, 17)
    input_test["Month"] = min_max(input_test["Month"], 1, 12)

preprocess_data(input_train, input_val, input_test)


# In[ ]:


from random import randint
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from RAdam.radam import RAdam
from lookahead.lookahead import Lookahead
from torchcontrib.optim import SWA
from robust_loss_pytorch.adaptive import AdaptiveLossFunction


# In[ ]:


BATCH_SIZE = 512


# In[ ]:


input_train.head()


# In[ ]:


INTERSEC_IN = 4797
STREET_IN = 1831
DIR_IN = 8
CITY_IN = 4
STREET_TYPE_IN = 16


# In[ ]:


def to_one_hot(targets, num_classes):
    return torch.from_numpy(np.eye(num_classes)[targets]).float()

def prepare_batch(batch, training=True):
    batch_x = {}

    batch_x["InDir"] = to_one_hot(batch["EntryHeading"].values, DIR_IN)
    batch_x["OutDir"] = to_one_hot(batch["ExitHeading"].values, DIR_IN)
    batch_x["City"] = to_one_hot(batch["City"].values, CITY_IN)
    batch_x["InType"] = to_one_hot(batch["EntryType"].values, STREET_TYPE_IN)
    batch_x["OutType"] = to_one_hot(batch["ExitType"].values, STREET_TYPE_IN)
    
    batch_x["InStreet"] = to_one_hot(batch["EntryStreetID"].values, STREET_IN)
    batch_x["OutStreet"] = to_one_hot(batch["ExitStreetID"].values, STREET_IN)
    
    if randint(0, 7) == 0 and training:
        uid = batch["IntersectionUID"].values
        uid[0] = INTERSEC_IN - 1
        batch_x["UID"] = to_one_hot(uid, INTERSEC_IN)
    else:
        batch_x["UID"] = to_one_hot(batch["IntersectionUID"].values, INTERSEC_IN)
        
    batch_x["Hour"] = torch.from_numpy(batch["Hour"].values).float().unsqueeze(1)
    batch_x["Weekend"] = torch.from_numpy(batch["Weekend"].values).float().unsqueeze(1)
    batch_x["Month"] = torch.from_numpy(batch["Month"].values).float().unsqueeze(1)
    batch_x["Latitude"] = torch.from_numpy(batch["Latitude"].values).float().unsqueeze(1)
    batch_x["Longitude"] = torch.from_numpy(batch["Longitude"].values).float().unsqueeze(1)
    
    batch_x["RightTurn"] = torch.from_numpy(batch["RightTurn"].values).float().unsqueeze(1)
    batch_x["LeftTurn"] = torch.from_numpy(batch["LeftTurn"].values).float().unsqueeze(1)
    batch_x["PassThru"] = torch.from_numpy(batch["PassThru"].values).float().unsqueeze(1)
    batch_x["UTurn"] = torch.from_numpy(batch["UTurn"].values).float().unsqueeze(1)
    batch_x["Direction"] = torch.from_numpy(batch["Direction"].values).float().unsqueeze(1)
    batch_x["LeftSide"] = torch.from_numpy(batch["LeftSide"].values).float().unsqueeze(1)
    batch_x["RightSide"] = torch.from_numpy(batch["RightSide"].values).float().unsqueeze(1)
    
    batch_x["HighTemp"] = torch.from_numpy(batch["HighTemp"].values).float().unsqueeze(1)
    batch_x["LowTemp"] = torch.from_numpy(batch["LowTemp"].values).float().unsqueeze(1)
    batch_x["SnowFall"] = torch.from_numpy(batch["SnowFall"].values).float().unsqueeze(1)
    batch_x["RainFall"] = torch.from_numpy(batch["RainFall"].values).float().unsqueeze(1)
    batch_x["DayLight"] = torch.from_numpy(batch["DayLight"].values).float().unsqueeze(1)
    batch_x["SunShine"] = torch.from_numpy(batch["SunShine"].values).float().unsqueeze(1)
    batch_x["RushHour1"] = torch.from_numpy(batch["RushHour1"].values).float().unsqueeze(1)
    batch_x["RushHour2"] = torch.from_numpy(batch["RushHour2"].values).float().unsqueeze(1)
    
    batch_x["CenterDist"] = torch.from_numpy(batch["CenterDist"].values).float().unsqueeze(1)
    batch_x["MinDist"] = torch.from_numpy(batch["MinDist"].values).float().unsqueeze(1)
    batch_x["NDist"] = torch.from_numpy(batch["NDist"].values).float().unsqueeze(1)
    batch_x["SDist"] = torch.from_numpy(batch["SDist"].values).float().unsqueeze(1)
    batch_x["EDist"] = torch.from_numpy(batch["EDist"].values).float().unsqueeze(1)
    batch_x["WDist"] = torch.from_numpy(batch["WDist"].values).float().unsqueeze(1)

    
    if "TotalTimeStopped_p20" in batch:
        batch_y = torch.from_numpy(pd.concat([batch["TotalTimeStopped_p20"], batch["TotalTimeStopped_p50"],
                                                 batch["TotalTimeStopped_p80"], batch["DistanceToFirstStop_p20"],
                                                 batch["DistanceToFirstStop_p50"], batch["DistanceToFirstStop_p80"]],
                                                 axis=1).values).float()
        return batch_x, batch_y
    else:
        return batch_x, None


# In[ ]:


batch = input_train.sample(5)
batch_x, batch_y = prepare_batch(batch)
print(batch_x.keys())


# In[ ]:


class TraficPred(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(TraficPred, self).__init__()
         
        INTERSEC_EMB_CH = 128
        STREET_EMB_CH = 64
        DIR_EMB_CH = 8
        CITY_EMB_CH = 4
        STREET_TYPE_CH = 16

        NUM_OUT = 6
        
        self.uid_emb = nn.Linear(INTERSEC_IN, INTERSEC_EMB_CH, bias=True)
        self.city_emb = nn.Linear(CITY_IN, CITY_EMB_CH, bias=True)

        self.street_emb = nn.Linear(STREET_IN, STREET_EMB_CH, bias=True)
        self.s_type_emb = nn.Linear(STREET_TYPE_IN, STREET_TYPE_CH, bias=True)

        self.dir_emb = nn.Linear(DIR_IN, DIR_EMB_CH, bias=True)
        
        EMB_CH = 2 * DIR_EMB_CH + CITY_EMB_CH + 2 * STREET_TYPE_CH + INTERSEC_EMB_CH + 2 * STREET_EMB_CH
        self.linear_block = nn.Sequential(
                                nn.Linear(EMB_CH + 21, 256, bias=True),
                                nn.PReLU(),
                                nn.Linear(256, 512, bias=True),
                                nn.PReLU(),
                                nn.Linear(512, 256, bias=True),
                                nn.PReLU(),
                                nn.Dropout(dropout_rate),
                                nn.Linear(256, NUM_OUT, bias=True),
                                nn.ReLU())
        
    def forward(self, input):
        
        uid_emb = self.uid_emb(input["UID"])
        city_emb = self.city_emb(input["City"])
        
        in_street_emb = self.street_emb(input["InStreet"])
        out_street_emb = self.street_emb(input["OutStreet"])
        
        in_type_emb = self.s_type_emb(input["InType"])
        out_type_emb = self.s_type_emb(input["OutType"])
        
        in_dir_emb = self.dir_emb(input["InDir"])
        out_dir_emb = self.dir_emb(input["OutDir"])
        
        emb = torch.cat((uid_emb, in_dir_emb, out_dir_emb, city_emb, in_type_emb, out_type_emb,
                                                         in_street_emb, out_street_emb), dim=1)
        
        x = torch.cat((emb, input["Latitude"], input["Longitude"], input["Hour"],
                            input["Weekend"], input["CenterDist"],# input["Month"],
                            #input["NDist"], input["SDist"], input["WDist"], input["EDist"],
                            input["HighTemp"], input["LowTemp"], input["SnowFall"],
                            input["RainFall"], input["DayLight"], input["SunShine"],
                            input["RightTurn"], input["LeftTurn"], input["PassThru"],
                            input["UTurn"], input["Direction"], input["LeftSide"],
                            input["RightSide"], input["RushHour1"], input["RushHour2"],
                            input["MinDist"]), dim=1)
        
        return self.linear_block(x)

    
class MSELoss(nn.Module): #
    def __init__(self):
        super(MSELoss, self).__init__()
        
        self.mse = nn.MSELoss()
    
    def forward(self, input, target):
        
        return self.mse(input, target)

    
class L1Loss(nn.Module): #
    def __init__(self):
        super(L1Loss, self).__init__()
        
        self.l1 = nn.L1Loss()
    
    def forward(self, input, target):
        
        return self.l1(input, target)

class BarronLoss(nn.Module): #
    def __init__(self):
        super(BarronLoss, self).__init__()
            
        self.barron = AdaptiveLossFunction(6, torch.float32, "cpu")
    
    def forward(self, input, target):
        
            return torch.mean(self.barron.lossfun(input - target))


# In[ ]:


def train(model, loss_fn, optimizer, input, num_epoch=30):
    
    best_loss = float("INF")
    
    for epoch in range(num_epoch):
        print("{}/{}: LR-{}".format(epoch+1, num_epoch,
                        optimizer.param_groups[0]["lr"]))
        
        sum_loss = 0
        model.train()

        with tqdm(total=1000, iterable=range(1000),
                 dynamic_ncols=False, unit="it", unit_scale=True,
                 desc="Training", postfix={"Loss":0.0}, file=sys.stdout) as tqdm_bar:
            
            for it in tqdm_bar:
                batch = input.sample(BATCH_SIZE)
                batch_x, batch_y = prepare_batch(batch)

                output = model(batch_x)
                loss = loss_fn(output, batch_y)
                sum_loss += loss.item()
                
                del batch_x, batch_y
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                tqdm_bar.set_postfix(oredered_dict={"Loss":"{0:.4f}"                            .format(sum_loss / (it + 1))}, refres=True)
                del output, loss
                
        sum_loss = 0        
        model.eval()
        with tqdm(total=400, iterable=range(400),
                 dynamic_ncols=False, unit="it", unit_scale=True,
                 desc="Validation", postfix={"Loss":0.0}, file=sys.stdout) as tqdm_bar:
            
            for it in tqdm_bar:
                batch = input.sample(BATCH_SIZE)
                batch_x, batch_y = prepare_batch(batch)

                output = model(batch_x)
                loss = loss_fn(output, batch_y)
                sum_loss += loss.item()
                
                del batch_x, batch_y
            
                tqdm_bar.set_postfix(oredered_dict={"Loss":"{0:.4f}"                            .format(sum_loss / (it + 1))}, refres=True)
                del output, loss

        if sum_loss / 400 < best_loss:
            model_state = {"weights": model.state_dict(),
                         "loss": sum_loss / 400,
                         "epoch": epoch + 1}
            best_loss = sum_loss / 400
            
            torch.save(model_state, "weights.pth.tar")
            


# In[ ]:


model = TraficPred()
optimizer = Lookahead(RAdam(model.parameters(), lr=0.001), k=5, alpha=0.5)
#optimizer = Adam(model.parameters(), lr=0.01)
train(model, MSELoss(), optimizer, input_train)


# In[ ]:


model_state = torch.load("weights.pth.tar")
print("Loading weights from epoch:{}".format(model_state["epoch"]))
model.load_state_dict(model_state["weights"])


# In[ ]:


def pred(model, input):
    model.eval()
    preds = np.zeros([input.shape[0]] + [6])
    num_samples = len(input.index)
    with tqdm(total=num_samples//BATCH_SIZE, iterable=range(num_samples//BATCH_SIZE),
             dynamic_ncols=False, unit="it", unit_scale=True,
             file=sys.stdout) as tqdm_bar:

        for it in tqdm_bar:
            batch = input[it * BATCH_SIZE:(it + 1) * BATCH_SIZE]
            batch_x, _ = prepare_batch(batch, training=False)

            output = model(batch_x)
            out_np = output.detach().numpy()
            preds[it * BATCH_SIZE:it * BATCH_SIZE + out_np.shape[0]] = out_np
            del batch_x, output
    
    return preds


# In[ ]:


def test(pred, target):
    dif = pred - target
    return math.sqrt(np.mean(dif ** 2, axis=(0,1)))


# In[ ]:


columns=["pred1", "pred2", "pred3", "pred4", "pred5", "pred6"]

trains = pd.DataFrame(data=pred(model, input_train), columns=columns, index=input_train["RowId"])
vals = pd.DataFrame(data=pred(model, input_val), columns=columns, index=input_val["RowId"])
print("Training  : {}".format(test(trains.values, input_train[["TotalTimeStopped_p20", "TotalTimeStopped_p50",
                                                             "TotalTimeStopped_p80", "DistanceToFirstStop_p20",
                                                             "DistanceToFirstStop_p50", "DistanceToFirstStop_p80"]].values)))

print("Validation: {}".format(test(vals.values, input_val[["TotalTimeStopped_p20", "TotalTimeStopped_p50",
                                                             "TotalTimeStopped_p80", "DistanceToFirstStop_p20",
                                                             "DistanceToFirstStop_p50", "DistanceToFirstStop_p80"]].values)))


# In[ ]:


input_train.head()


# In[ ]:


input_test.head()


# In[ ]:


preds = pd.DataFrame(data=pred(model, input_test), columns=columns, index=input_test["RowId"])


# In[ ]:


preds.head()


# In[ ]:


preds.mean(axis=0)


# In[ ]:


means=pd.DataFrame({})
means["Training"] = trains.mean(axis=0)
means["Validation"] = vals.mean(axis=0)
means["Testing"] = preds.mean(axis=0)
means["Index"] = means.index
means = means.melt(id_vars="Index", value_vars=["Training", "Validation", "Testing"], )
sns.factorplot(x="Index", y="value", hue="variable", data=means, kind="bar")


# In[ ]:


def rmse(pred, target):
    dif = pred - target
    return np.sqrt(np.mean(dif ** 2, axis=(1)))

input_train["Error"] = rmse(trains.values, input_train[["TotalTimeStopped_p20", "TotalTimeStopped_p50",
                                                             "TotalTimeStopped_p80", "DistanceToFirstStop_p20",
                                                             "DistanceToFirstStop_p50", "DistanceToFirstStop_p80"]].values)

input_train["ErrorDistance"] = rmse(trains[["pred4", "pred5", "pred6"]].values, input_train[["DistanceToFirstStop_p20",
                                                             "DistanceToFirstStop_p50", "DistanceToFirstStop_p80"]].values)

input_train["ErrorDistance80"] = rmse(trains[["pred6"]].values, input_train[["DistanceToFirstStop_p80"]].values)

input_train["ErrorTime"] = rmse(trains[["pred1", "pred2", "pred3"]].values, input_train[["TotalTimeStopped_p20",
                                                        "TotalTimeStopped_p50", "TotalTimeStopped_p80"]].values)


# In[ ]:


def plot_map_error(df, city, error):
    count=df.groupby(['City','Latitude','Longitude'])[error].mean().reset_index()
    
    fig = px.scatter_mapbox(count[count["City"]==city], 
                            lat="Latitude", lon="Longitude",size=error,size_max=10,
                            color=error, color_continuous_scale=px.colors.sequential.Inferno, zoom=11)
    fig.update_layout(mapbox_style="stamen-terrain")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()


# In[ ]:


plot_map_error(input_train, 0, "Error")


# In[ ]:


plot_map_error(input_train, 0, "ErrorDistance80")


# In[ ]:


plot_map_error(input_train, 0, "ErrorTime")


# In[ ]:


preds = preds.values.flatten()

sub  = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")
sub["Target"] = preds.tolist()
sub.to_csv("pred.csv",index = False)

