#!/usr/bin/env python
# coding: utf-8

# # Objective
# 
# In this notebook i will perform a basic exploratory data analysis of the data, also we will make a simple model to calculate the emission factor coefficient of electricity that produce Green House Gases for Puerto Rico.
# 
# Here is a link of a public kernel that help me understand better the problem we are trying to solve:
# 
# https://www.kaggle.com/parulpandey/understanding-the-data-wip
# 
# * The model needs to produce a value for the an annual average historical grid-level electricity emissions factor (based on rolling 12-months of data from July 2018 - July 2019) for the sub-national region?
# 
# We also recieve bonuses for other objectives but we will not cover them in this initial notebook

# # General Equation for Emission Estimation
# 
# The general equation for calculating emission is the following.
# 
# E = A X EF X (1 - ER / 100)
# 
# * E = emissions
# * A = activity rate
# * EF = emission factor
# * ER = overall emission reduction efficiency %.

# In[ ]:


import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import glob
import cv2
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from plotly import tools, subplots
import plotly.offline as py
py.init_notebook_mode(connected = True)
import plotly.graph_objs as go
import plotly.express as px
pd.set_option('max_columns', 1000)
pd.set_option('max_rows', 1000)
import warnings
warnings.filterwarnings('ignore')
import gc

import rasterio as rio
import folium
import tifffile as tiff

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import preprocessing


# # Global Power Plant Database

# In[ ]:


gpp_df = pd.read_csv('../input/ds4g-environmental-insights-explorer/eie_data/gppd/gppd_120_pr.csv')
gpp_df.head()


# We have a lot of usefull data in this dataset, starting by the estimate annual electricity generation in gigawatt-hours.
# 
# Let's first check the location of this power plants. Im ussing a function extracted from this great kernel!.
# 
# https://www.kaggle.com/paultimothymooney/how-to-get-started-with-the-earth-engine-data

# In[ ]:


def plot_points_on_map(dataframe,begin_index,end_index,latitude_column,latitude_value,longitude_column,longitude_value,zoom):
    df = dataframe[begin_index:end_index]
    location = [latitude_value,longitude_value]
    plot = folium.Map(location=location,zoom_start=zoom)
    for i in range(0,len(df)):
        popup = folium.Popup(str(df.primary_fuel[i:i+1]))
        folium.Marker([df[latitude_column].iloc[i],df[longitude_column].iloc[i]],popup=popup).add_to(plot)
    return(plot)

def overlay_image_on_puerto_rico(file_name,band_layer,lat,lon,zoom):
    band = rio.open(file_name).read(band_layer)
    m = folium.Map([lat, lon], zoom_start=zoom)
    folium.raster_layers.ImageOverlay(
        image=band,
        bounds = [[18.6,-67.3,],[17.9,-65.2]],
        colormap=lambda x: (1, 0, 0, x),
    ).add_to(m)
    return m

def plot_scaled(file_name):
    vmin, vmax = np.nanpercentile(file_name, (5,95))  # 5-95% stretch
    img_plt = plt.imshow(file_name, cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()

def split_column_into_new_columns(dataframe,column_to_split,new_column_one,begin_column_one,end_column_one):
    for i in range(0, len(dataframe)):
        dataframe.loc[i, new_column_one] = dataframe.loc[i, column_to_split][begin_column_one:end_column_one]
    return dataframe


# In[ ]:


gpp_df = split_column_into_new_columns(gpp_df,'.geo','latitude',50,66)
gpp_df = split_column_into_new_columns(gpp_df,'.geo','longitude',31,48)
gpp_df['latitude'] = gpp_df['latitude'].astype(float)
a = np.array(gpp_df['latitude'].values.tolist()) # 18 insted of 8
gpp_df['latitude'] = np.where(a < 10, a + 10, a).tolist()
lat = 18.200178; lon = -66.664513
plot_points_on_map(gpp_df, 0, 425, 'latitude', lat, 'longitude', lon, 9)


# In this map we can clearly see that the plant are in Puerto Rico.
# 
# Let's check how this plants work, in other words what is the primary fuel for the generation of electricity
# 
# Also we are going to  inspect other variables likes the generation_gwh_2013, generation_gwh_2014 etc...

# In[ ]:


years = [2013, 2014, 2015, 2016, 2017]
print([(gpp_df[f'generation_gwh_{x}'].nunique()) for x in years])


# It's look that this columns have only one number (0). Let's continue

# In[ ]:


print('There are {} power plants'.format(gpp_df.shape[0]))


# In[ ]:


def bar_plot(df, column, title, width, height, n, get_count = True):
    if get_count == True:
        cnt_srs = df[column].value_counts()[:n]
    else:
        cnt_srs = df
        
    trace = go.Bar(
        x = cnt_srs.index,
        y = cnt_srs.values,
        marker = dict(
            color = '#1E90FF', 
        ), 
    )
    
    layout = go.Layout(
        title = go.layout.Title(
            text = title,
            x = 0.5
        ),
        font = dict(size = 14),
        width = width,
        height = height,
    )
    
    data = [trace]
    fig = go.Figure(data = data, layout = layout)
    py.iplot(fig, filename = 'bar_plot')
bar_plot(gpp_df, 'primary_fuel', 'Primary Fuel Distribution', 800, 500, 100)


# * Hydro followed by gas are the most common fuels of Puerto Rico electricity plants, coal is the less common
# * We have only one coal plant!

# In[ ]:


pf_generation = gpp_df.groupby('primary_fuel')['estimated_generation_gwh'].sum()
bar_plot(pf_generation, 'primary_fuel', 'Electricity Generation Sum by Primary Fuel', 800, 500, 100, False)


# * Coal is the fuel type that produce more energy, on the other hand is the one that power less plants! (1 plant). Coal in known to be an energy commodity
# 
# Commissioning year is important, older plants pollute more. Let's check this feature.

# In[ ]:


gpp_df['commissioning_year'].value_counts()


# * We have a lot of 0 values, where 0 can be unknown commissioning_year. Also we have a plant that was built in 1942!. 
# 
# Let'check the capacity mesured in mega watts

# In[ ]:


pf_capacity = gpp_df.groupby('primary_fuel')['capacity_mw'].sum()
bar_plot(pf_capacity, 'primary_fuel', 'Capacity Sum by Primary Fuel', 800, 500, 100, False)


# * Oil power plants have the biggest capacity (sum of all oil plants) followed by gas
# 
# Capacity and Generation stats

# In[ ]:


gpp_df.groupby(['primary_fuel']).agg({'estimated_generation_gwh': ['nunique', 'sum', 'mean', 'max', 'min'], 'capacity_mw' : ['nunique', 'sum', 'mean', 'max', 'min']}).reset_index()


# * Oil fuel power plants have the highest mean capacity, followed by the single Coal power plant

# Let's check the most important plants (electricity generation)

# In[ ]:


df = gpp_df[['name','latitude','longitude','primary_fuel','capacity_mw','estimated_generation_gwh', 'owner']].sort_values('estimated_generation_gwh', ascending = False)
df.head()


# * A.E.S Corp it the plant that generates more energy but's its capacity is not the biggest.

# # Let's Jump to Copernicus image data
# 
# * How do we read this data?

# In[ ]:


image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180701T161259_20180707T175356.tif'
latitude=18.1429005246921; longitude=-65.4440010699994
overlay_image_on_puerto_rico(image,band_layer=7,lat=latitude,lon=longitude,zoom=8)


# Let's check another picture.

# In[ ]:


image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180702T173526_20180708T192358.tif'
overlay_image_on_puerto_rico(image,band_layer=7,lat=latitude,lon=longitude,zoom=8)


# In[ ]:


image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180704T165720_20180710T184641.tif'
overlay_image_on_puerto_rico(image,band_layer=7,lat=latitude,lon=longitude,zoom=8)


# In[ ]:


image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180706T161914_20180712T200737.tif'
overlay_image_on_puerto_rico(image,band_layer=7,lat=latitude,lon=longitude,zoom=8)


# * Strange, it seems their is no red on this picture

# In[ ]:


image = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/s5p_no2_20180707T174140_20180713T191854.tif'
overlay_image_on_puerto_rico(image,band_layer=7,lat=latitude,lon=longitude,zoom=8)


# # Question so Far
# 
# Puerto Rico also offers a unique fuel mix and distinctive energy system layout that should make it easier to isolate pollution attributable to power generation in the remote sensing data.
# 
# * NO2 reading are only cause by the power plants electricity generation or more activities? I think they are more factors involved
# * How much pollution of the total pollution does power plants electricity generation produce?
# * How can we isolate the pollution attributable to power plants generation in the remote sensing data?

# # First Simple Model
# 
# Let's calculate the factor with the last image of NO2
# 
# We know that not all energy fuel types produce pollution, the first assumption is going to be that only Caol, Gas and Oil produce pollution.
# 
# Searching in google I found that 14% of the pollution is made from power plants electricity generation (im not entirely sure that that is correct).
# 
# We are going to use Simplified emission factor formula:
# 
# E / A = EF
# 
# * E = emissions
# * A = activity rate
# * EF = emission factor

# In[ ]:


tiff.imread(image).shape


# As you can see this photo we have 3 dimensions, the last dimension corresponds to the band. If you look in the data section, their is a description of the bands (last channel)
# 
# https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S5P_OFFL_L3_NO2
# 
# * NO2 emissions are measured in band1, band2, band3 and band4

# In[ ]:


p_fuel_types = ['Coal', 'Oil', 'Gas']
# only consider pollute fuel types
p_type_df = gpp_df[gpp_df['primary_fuel'].isin(p_fuel_types)]
# sum the electricity generation
p_type_sum = p_type_df['estimated_generation_gwh'].sum()
# sum the pollution of the last satellite picture
sum_no2_emission = np.sum(tiff.imread(image)[:, :, 0 : 4])
# consider 14% of pollution is made from power plants electricity
sum_no2_emission_oe = sum_no2_emission * 0.14
# use the simplified emission factor formula
factor = sum_no2_emission_oe / p_type_sum
print(f'Simplified emissions factor for Puerto Rico is {factor} mol * h / m^2 * gw')


# # What could be wrong with this simple methology:
# 
# * Remember that the model needs to reproduce an annual average historical grid-level electricity emissions factor (based on rolling 12-months of data from July 2018 - July 2019) for Puerto Rico.
# 
# * Pictures are different each day, meaning that using only one picture makes no sense at all.
# 
# * Consider 14% of pollution is made from power plants electricity? This is a general stat, we should some how use the information given in the power plant database to estimate better. I believe that fuel type and commissioning_year is a very important factor to consider. In other words a really important objective is to calculate the % of pollution made from Puerto Rico power plants!!!.
# 
# * Another important factor is to check each plant. Does all plants in Puerto Rico pollute?
# 
# Let's try and make better and more accurate baseline factor using all the NO2 emission pictures that we have!!

# In[ ]:


no2_path = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/s5p_no2/*'
no2_pictures_path = glob.glob(no2_path)
len(no2_pictures_path)
print('We have {} pictures of the Copernicus Sentinel'.format(len(no2_pictures_path)))


# Let's preprocess this data.

# In[ ]:


# this function will help us extract the no2 emission data in a tabular way
def read_s5p_no2_pictures_data(only_no2_emissions = True):
    s5p_no2_pictures = []
    for num, i in tqdm(enumerate(no2_pictures_path), total = 387):
        temp_s5p_no2_pictures = {'start_date': [], 'end_date': [], 'data': []}
        temp_s5p_no2_pictures['start_date'] = no2_pictures_path[num][76:84]
        temp_s5p_no2_pictures['end_date'] = no2_pictures_path[num][92:100]
        # only no2 emissions
        if only_no2_emissions:
            temp_s5p_no2_pictures['data'] = tiff.imread(i)[:, :, 0 : 4]
            temp_s5p_no2_pictures['no2_emission_sum'] = np.sum(tiff.imread(i)[:, :, 0 : 4])
            temp_s5p_no2_pictures['no2_emission_mean'] = np.average(tiff.imread(i)[:, :, 0 : 4])
            temp_s5p_no2_pictures['no2_emission_std'] = np.std(tiff.imread(i)[:, :, 0 : 4])
            temp_s5p_no2_pictures['no2_emission_max'] = np.max(tiff.imread(i)[:, :, 0 : 4])
            temp_s5p_no2_pictures['no2_emission_min'] = np.min(tiff.imread(i)[:, :, 0 : 4])
            s5p_no2_pictures.append(temp_s5p_no2_pictures)
        # all Copernicus data
        else:
            temp_s5p_no2_pictures['data'] = tiff.imread(i)
            s5p_no2_pictures.append(temp_s5p_no2_pictures)
    s5p_no2_pictures = pd.DataFrame(s5p_no2_pictures)
    s5p_no2_pictures['start_date'] = pd.to_datetime(s5p_no2_pictures['start_date'])
    s5p_no2_pictures['end_date'] = pd.to_datetime(s5p_no2_pictures['end_date'])
    s5p_no2_pictures.sort_values('start_date', inplace = True)
    s5p_no2_pictures.reset_index(drop = True, inplace = True)
    return s5p_no2_pictures

s5p_no2_pictures_df = read_s5p_no2_pictures_data()


# # NO2 Analysis
# 
# For speed purpose we are going to split the data in 2 dataframes

# In[ ]:


s5p_no2_pictures_stats = s5p_no2_pictures_df[[col for col in s5p_no2_pictures_df.columns if col not in ['data']]]
s5p_no2_pictures_data = s5p_no2_pictures_df[['data']]
del s5p_no2_pictures_df
s5p_no2_pictures_stats.head()


# There most common range between dates is 6 days.
# 
# We have some negatives values, is that correct?
# 
# MMMMMM we have some rows were the sum of NO2 emission is NaN. Let's check why

# In[ ]:


def check_arrays(df, row = 1):
    band1 = pd.DataFrame(s5p_no2_pictures_data['data'][row][:, :, 0])
    band2 = pd.DataFrame(s5p_no2_pictures_data['data'][row][:, :, 1])
    band3 = pd.DataFrame(s5p_no2_pictures_data['data'][row][:, :, 2])
    band4 = pd.DataFrame(s5p_no2_pictures_data['data'][row][:, :, 3])
    
    def check_nan(df):
        df_nan = df.isnull().values.sum()
        return df_nan
    
    band1_nan = check_nan(band1)
    band2_nan = check_nan(band2)
    band3_nan = check_nan(band3)
    band4_nan = check_nan(band4)
    
    print('From row {} we have {} nan values for band1'.format(row, band1_nan))
    print('From row {} we have {} nan values for band2'.format(row, band2_nan))
    print('From row {} we have {} nan values for band3'.format(row, band3_nan))
    print('From row {} we have {} nan values for band4'.format(row, band4_nan))

    return band1, band2, band3, band4

band1, band2, band3, band4 = check_arrays(s5p_no2_pictures_data, row = 4)


# We have some nan values in our images. 
# 
# * Why do we have some nan values in our images? 
# * Should we impute these values for better data quality or ignore them??
# 
# Lets ignore them for know, going to leave them in my backlog :)

# In[ ]:


# this function ignore nan values from the images
def read_s5p_no2_pictures_data_ignore_nan(only_no2_emissions = True):
    s5p_no2_pictures = []
    for num, i in tqdm(enumerate(no2_pictures_path), total = 387):
        temp_s5p_no2_pictures = {'start_date': [], 'end_date': [], 'data': []}
        temp_s5p_no2_pictures['start_date'] = no2_pictures_path[num][76:84]
        temp_s5p_no2_pictures['end_date'] = no2_pictures_path[num][92:100]
        # only no2 emissions
        if only_no2_emissions:
            temp_s5p_no2_pictures['data'] = tiff.imread(i)[:, :, 0 : 4]
            temp_s5p_no2_pictures['no2_emission_sum'] = np.nansum(tiff.imread(i)[:, :, 0 : 4])
            temp_s5p_no2_pictures['no2_emission_mean'] = np.nanmean(tiff.imread(i)[:, :, 0 : 4])
            temp_s5p_no2_pictures['no2_emission_std'] = np.nanstd(tiff.imread(i)[:, :, 0 : 4])
            temp_s5p_no2_pictures['no2_emission_max'] = np.nanmax(tiff.imread(i)[:, :, 0 : 4])
            temp_s5p_no2_pictures['no2_emission_min'] = np.nanmin(tiff.imread(i)[:, :, 0 : 4])
            s5p_no2_pictures.append(temp_s5p_no2_pictures)
        # all Copernicus data
        else:
            temp_s5p_no2_pictures['data'] = tiff.imread(i)
            s5p_no2_pictures.append(temp_s5p_no2_pictures)
    s5p_no2_pictures = pd.DataFrame(s5p_no2_pictures)
    s5p_no2_pictures['start_date'] = pd.to_datetime(s5p_no2_pictures['start_date'])
    s5p_no2_pictures['end_date'] = pd.to_datetime(s5p_no2_pictures['end_date'])
    s5p_no2_pictures.sort_values('start_date', inplace = True)
    s5p_no2_pictures.reset_index(drop = True, inplace = True)
    return s5p_no2_pictures

s5p_no2_pictures_df_ig_nan = read_s5p_no2_pictures_data_ignore_nan()


# In[ ]:


s5p_no2_pictures_stats_ig_nan = s5p_no2_pictures_df_ig_nan[[col for col in s5p_no2_pictures_df_ig_nan.columns if col not in ['data']]]
del s5p_no2_pictures_df_ig_nan
s5p_no2_pictures_stats_ig_nan.head()


# Now we have a time series dataframe!!. Let's make some plots to visualize the information

# In[ ]:


def line_plot(df, x, y, title, width, height):
    trace = go.Scatter(
        x = df[x],
        y = df[y],
        mode='lines',
        name='lines',
        marker = dict(
            color = '#1E90FF', 
        ), 
    )
    
    layout = go.Layout(
        title = go.layout.Title(
            text = title,
            x = 0.5
        ),
        font = dict(size = 14),
        width = width,
        height = height,
    )
    
    data = [trace]
    fig = go.Figure(data = data, layout = layout)
    py.iplot(fig, filename = 'line_plot')
line_plot(s5p_no2_pictures_stats_ig_nan, 'start_date', 'no2_emission_sum', 'NO2 emission by date', 1400, 600)


# * We have information of NO2 emission between 2018-07-01 and 2019-06-29
# * We can see a lot of peaks, is there a reasonable explanation for this low values? Maybee ignoring nan values is not correct.
# * Why do we have 387 observations? A year have 365 days
# * We have some duplicate dates

# In[ ]:


def line_plot_check_nan(df1, df2, x, y, title, width, height):
    
    trace1 = go.Scatter(
        x = df1[x],
        y = df1[y],
        mode='lines',
        name='with_nans',
        marker = dict(
            color = '#1E90FF', 
        ), 
    )
    
    df3 = df2.dropna()
    trace2 = go.Scatter(
        x = df3[x],
        y = df3[y],
        mode='markers',
        name='no_nans',
        marker = dict(
            color = 'red', 
        ), 
    )
    
    layout = go.Layout(
        title = go.layout.Title(
            text = title,
            x = 0.5
        ),
        font = dict(size = 14),
        width = width,
        height = height,
    )
    
    data = [trace1, trace2]
    fig = go.Figure(data = data, layout = layout)
    py.iplot(fig, filename = 'line_plot')
line_plot_check_nan(s5p_no2_pictures_stats_ig_nan, s5p_no2_pictures_stats, 'start_date', 'no2_emission_sum', 'NO2 emission by date', 1400, 600)


# * This peaks could be related with missing values!!!
# 
# * April 15 of 2019 is an outlier?

# In[ ]:


line_plot(s5p_no2_pictures_stats[s5p_no2_pictures_stats['start_date']!='2019-04-15'].dropna(), 'start_date', 'no2_emission_sum', 'NO2 emission by date', 1400, 600)


# Can we use only the values of the previous graph for our factor?, i think we cant because we are loosing too much information. We need to find a way to deal with this

# In[ ]:


n_duplicates_dates = s5p_no2_pictures_stats_ig_nan.shape[0] - s5p_no2_pictures_stats_ig_nan.drop_duplicates(subset = ['start_date', 'end_date']).shape[0]
print(f'We have {n_duplicates_dates} duplicate days')


# * Why do we have duplicate days?
# * What should we do with this days?

# Let's interpolate the nan values!!!

# In[ ]:


# this function will help us extract the no2 emission data in a tabular way
def read_s5p_no2_pictures_data_fill(only_no2_emissions = True):
    s5p_no2_pictures = []
    for num, i in tqdm(enumerate(no2_pictures_path), total = 387):
        temp_s5p_no2_pictures = {'start_date': [], 'end_date': [], 'data': []}
        temp_s5p_no2_pictures['start_date'] = no2_pictures_path[num][76:84]
        temp_s5p_no2_pictures['end_date'] = no2_pictures_path[num][92:100]
        # only no2 emissions
        if only_no2_emissions:
            image = tiff.imread(i)[:, :, 0 : 4]
            band1 = pd.DataFrame(image[: ,: , 0]).interpolate()
            band1.fillna(band1.mean(), inplace = True)
            band2 = pd.DataFrame(image[: ,: , 1]).interpolate()
            band2.fillna(band2.mean(), inplace = True)
            band3 = pd.DataFrame(image[: ,: , 2]).interpolate()
            band3.fillna(band3.mean(), inplace = True)
            band4 = pd.DataFrame(image[: ,: , 3]).interpolate()
            band4.fillna(band4.mean(), inplace = True)
            image = np.dstack((band1, band2, band3, band4))
            temp_s5p_no2_pictures['data'] = image
            temp_s5p_no2_pictures['no2_emission_sum'] = np.sum(image)
            temp_s5p_no2_pictures['no2_emission_mean'] = np.average(image)
            temp_s5p_no2_pictures['no2_emission_std'] = np.std(image)
            temp_s5p_no2_pictures['no2_emission_max'] = np.max(image)
            temp_s5p_no2_pictures['no2_emission_min'] = np.min(image)
            s5p_no2_pictures.append(temp_s5p_no2_pictures)
        # all Copernicus data
        else:
            temp_s5p_no2_pictures['data'] = tiff.imread(i)
            s5p_no2_pictures.append(temp_s5p_no2_pictures)
    s5p_no2_pictures = pd.DataFrame(s5p_no2_pictures)
    s5p_no2_pictures['start_date'] = pd.to_datetime(s5p_no2_pictures['start_date'])
    s5p_no2_pictures['end_date'] = pd.to_datetime(s5p_no2_pictures['end_date'])
    s5p_no2_pictures.sort_values('start_date', inplace = True)
    s5p_no2_pictures.reset_index(drop = True, inplace = True)
    return s5p_no2_pictures

s5p_no2_pictures_df_fill = read_s5p_no2_pictures_data_fill()


# In[ ]:


s5p_no2_pictures_stats_fill = s5p_no2_pictures_df_fill[[col for col in s5p_no2_pictures_df_fill.columns if col not in ['data']]]
del s5p_no2_pictures_df_fill
s5p_no2_pictures_stats_fill.head()


# In[ ]:


# drop nan values and check again for duplicate columns
s5p_no2_pictures_stats_fill = s5p_no2_pictures_stats_fill[s5p_no2_pictures_stats_fill['start_date']!='2019-04-15'].dropna()
# drop 2019-04-15 (probably an outlier or a rare event that can affect our factor calculation)
duplicate_columns = s5p_no2_pictures_stats_fill.shape[0] - s5p_no2_pictures_stats_fill.drop_duplicates(subset = ['start_date', 'end_date']).shape[0]
print(f'We have {duplicate_columns} duplicate columns')
print('We have {} days of data'.format(s5p_no2_pictures_stats_fill['start_date'].nunique()))


# Great! we have clean our NO2 dataset. Let's plot it again.

# In[ ]:


line_plot(s5p_no2_pictures_stats_fill, 'start_date', 'no2_emission_sum', 'NO2 emission by date', 1400, 800)


# * We have a nice and stable curve
# * Between August 13 of 2018 and March 9 of 2019  we can see a decrease in NO2 emissions (lower trend)
# * What can we do with the missing days? Can we predict them with an arima model?
# 
# Now that we have better NO2 information, let's calculate again the factor using the simple methodology

# In[ ]:


# get the mean NO2 emission between 2018/07/01 and 2019/06/29
sum_no2_emission = s5p_no2_pictures_stats_fill['no2_emission_sum'].mean()
# consider 14% of pollution is made from power plants electricity
sum_no2_emission_oe = sum_no2_emission * 0.14
# use the simplified emission factor formula (sum of estimated generation from Caol, Oil and Gas plants)
factor = sum_no2_emission_oe / p_type_sum
print(f'Simplified emissions factor for Puerto Rico is {factor} mol * h / m^2 * gw')


# Our factor changed a lot!!!.
# 
# Let's see how we can add more variable to the equation. Let's continue with the weather data!!

# # Weather Data
# 
# We also have pictures for this data were we can found different bands. Actually their are 9 bands. You can read more in the next link: https://developers.google.com/earth-engine/datasets/catalog/NOAA_GFS0P25

# In[ ]:


weather_path = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data/gfs/*'
weather_pictures_path = glob.glob(weather_path)
len(weather_pictures_path)
print('We have {} pictures of the global forecast system'.format(len(weather_pictures_path)))


# * We have more than one photo for each day. Probably the last 2 digits before .tif is the hour

# In[ ]:


tiff.imread(weather_pictures_path[0]).shape


# * Can only see 6 bands in this image. We are going to assume that their are the first 6 bands (plz correct me if i am wrong)

# In[ ]:


# this function will help us extract weather pictures in a tabular way
def read_weather_data():
    weather_pictures = []
    for num, i in tqdm(enumerate(weather_pictures_path), total = len(weather_pictures_path)):
        temp_weather_pictures = {'date': [], 'temperature_2m_above_ground': [], 'specific_humidity_2m_above_ground': [], 'relative_humidity_2m_above_ground': [], 
                                 'u_component_of_wind_10m_above_ground': [], 'v_component_of_wind_10m_above_ground': [], 'total_precipitation_surface': []}
        temp_weather_pictures['date'] = weather_pictures_path[num][68:-6]
        temp_weather_pictures['date'] = weather_pictures_path[num][68:-6]
        image = tiff.imread(i)
        temp_weather_pictures['temperature_2m_above_ground'] = image[ : , : , 0]
        temp_weather_pictures['specific_humidity_2m_above_ground'] = image[ : , : , 1]
        temp_weather_pictures['relative_humidity_2m_above_ground'] = image[ : , : , 2]
        temp_weather_pictures['u_component_of_wind_10m_above_ground'] = image[ : , : , 3]
        temp_weather_pictures['v_component_of_wind_10m_above_ground'] = image[ : , : , 4]
        temp_weather_pictures['total_precipitation_surface'] = image[ : , : , 5]
        temp_weather_pictures['temperature_2m_above_ground_mean'] = np.average(image[ : , : , 0])
        temp_weather_pictures['specific_humidity_2m_above_ground_mean'] = np.average(image[ : , : , 1])
        temp_weather_pictures['relative_humidity_2m_above_ground_mean'] = np.average(image[ : , : , 2])
        temp_weather_pictures['u_component_of_wind_10m_above_ground_mean'] = np.average(image[ : , : , 3])
        temp_weather_pictures['v_component_of_wind_10m_above_ground_mean'] = np.average(image[ : , : , 4])
        temp_weather_pictures['total_precipitation_surface_mean'] = np.average(image[ : , : , 5])
        
        weather_pictures.append(temp_weather_pictures)
    
    weather_pictures = pd.DataFrame(weather_pictures)
    weather_pictures['date'] = pd.to_datetime(weather_pictures['date'], infer_datetime_format  = True)
    weather_pictures.sort_values('date', inplace = True)
    weather_pictures.reset_index(drop = True, inplace = True)
    return weather_pictures

weather_pictures_df = read_weather_data()


# In[ ]:


weather_pictures_df.head()


# In[ ]:


# check missing values
img_columns = ['temperature_2m_above_ground', 'specific_humidity_2m_above_ground', 'relative_humidity_2m_above_ground', 
               'u_component_of_wind_10m_above_ground', 'v_component_of_wind_10m_above_ground', 'total_precipitation_surface']
weather_pictures_df[[col for col in weather_pictures_df.columns if col not in img_columns]].isnull().sum()


# Weather image dont have missing values, great!

# In[ ]:


weather_pictures_df_stats = weather_pictures_df[[col for col in weather_pictures_df.columns if col not in img_columns]]
n_duplicates = weather_pictures_df_stats.shape[0] - weather_pictures_df_stats['date'].nunique()
print(f'We have {n_duplicates} observations that belongs to a date with one or more records')


# Let's get the mean to have one observation for each day. Then let's plot them and check the graphs

# In[ ]:


weather_pictures_df_stats = weather_pictures_df_stats.groupby('date').mean().reset_index()
print('We have data for {} days'.format(weather_pictures_df_stats['date'].nunique()))
print('Our data start on {} and finish in {}'.format(weather_pictures_df_stats['date'].min(), weather_pictures_df_stats['date'].max()))
line_plot(weather_pictures_df_stats, 'date', 'temperature_2m_above_ground_mean', 'Temperature by Date', 1400, 800)


# * Temperature falls between October and June. Let's check the correlation between temperature and NO2 emission!

# In[ ]:


# Weather data have all the dates, on the other hand some days in the N02 dataframe are missing 
no2_weather = s5p_no2_pictures_stats_fill[['start_date', 'no2_emission_sum']].merge(weather_pictures_df_stats, left_on = 'start_date', right_on = 'date', how = 'left')
no2_tem_corr = no2_weather[['no2_emission_sum', 'temperature_2m_above_ground_mean']].corr().loc['no2_emission_sum', 'temperature_2m_above_ground_mean']
print(f'NO2 and temeprature have a correlation of: {no2_tem_corr}')


# In[ ]:


line_plot(weather_pictures_df_stats, 'date', 'specific_humidity_2m_above_ground_mean', 'Specific Humidity by Date', 1400, 800)


# In[ ]:


line_plot(weather_pictures_df_stats, 'date', 'relative_humidity_2m_above_ground_mean', 'Relative Humidity by Date', 1400, 800)


# In[ ]:


line_plot(weather_pictures_df_stats, 'date', 'u_component_of_wind_10m_above_ground_mean', 'U Component of Wind by Date', 1400, 800)


# In[ ]:


line_plot(weather_pictures_df_stats, 'date', 'v_component_of_wind_10m_above_ground_mean', 'V Component of Wind by Date', 1400, 800)


# In[ ]:


line_plot(weather_pictures_df_stats, 'date', 'total_precipitation_surface_mean', 'Total Precipitation Surface by Date', 1400, 800)


# In[ ]:


plt.figure(figsize = (14, 8))
sns.heatmap(no2_weather.corr(), annot = True, cmap = 'coolwarm')


# Temperature and relative humidity are the most correlated features with N02

# # Regression Analysis with Temperature and NO2
# 
# * In this experiment we are going make a regression model where we want to predict N02 emission
# 
# * We are only going to use weather data, later we can include the electricity produce by the power plants and get the coefficient as our factor!!
# 
# * Let's leave the last week as our test set

# In[ ]:


def preprocess_data(no2_weather, use_lags = True, use_time = True):
    reg_dataset = no2_weather[['date', 'temperature_2m_above_ground_mean', 'specific_humidity_2m_above_ground_mean', 'relative_humidity_2m_above_ground_mean', 'u_component_of_wind_10m_above_ground_mean', 
                               'v_component_of_wind_10m_above_ground_mean', 'total_precipitation_surface_mean', 'no2_emission_sum']]
    # get month for groupkfold validation
    reg_dataset['month'] = reg_dataset['date'].dt.month
    if use_time:
        # get day of week as feature
        reg_dataset['dayofweek'] = reg_dataset['date'].dt.dayofweek
        # one hot encoder 
        reg_dataset = pd.get_dummies(reg_dataset, columns = ['dayofweek'])
    if use_lags:
        # get no2_emissions lags
        reg_dataset['no2_emission_sum_t1'] = reg_dataset['no2_emission_sum'].shift(1)
        reg_dataset['no2_emission_sum_t2'] = reg_dataset['no2_emission_sum'].shift(2)
        reg_dataset['no2_emission_sum_t3'] = reg_dataset['no2_emission_sum'].shift(3)
        reg_dataset['no2_emission_rolling_mean_t1t3'] = (reg_dataset['no2_emission_sum_t1'] + reg_dataset['no2_emission_sum_t2'] + reg_dataset['no2_emission_sum_t3']) / 3
        # drop nan columns produce by the lags
        reg_dataset.dropna(inplace = True)
    # split train and test
    train = reg_dataset[reg_dataset['date'] < '2019-06-23']
    test = reg_dataset[reg_dataset['date'] >= '2019-06-23']
    features = [col for col in train.columns if col not in ['date', 'no2_emission_sum', 'month']]
    return train, test, features


def train_linear_regression(train, test, features, n_folds = 12):
    # 12 folds, each one representing 1 month
    target = 'no2_emission_sum'
    kfold = GroupKFold(n_folds)
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    all_coef = pd.DataFrame()
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train[features], groups = train['month'])):
        print(f'Training and evaluatin fold {fold}')
        x_train, y_train = train[features].iloc[trn_ind], train[target].iloc[trn_ind]
        x_val, y_val = train[features].iloc[val_ind], train[target].iloc[val_ind]
        # standarize train and eval
        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        test_scaled = scaler.transform(test[features])
        month = train[['month']].iloc[val_ind]['month'].unique()[0]
        model = LinearRegression().fit(x_train, y_train)
        fold_prediction = model.predict(x_val)
        fold_error = np.sqrt(metrics.mean_squared_error(y_val, fold_prediction))
        print(f'Our rmse for month {month} is {fold_error}')
        oof[val_ind] = fold_prediction
        predictions += model.predict(test_scaled) / n_folds
        coef = pd.DataFrame({'features': train[features].columns})
        coef['coef_'] = model.coef_
        all_coef = pd.concat([all_coef, coef])
    oof_rmse = np.sqrt(metrics.mean_squared_error(train[target], oof))
    test_error = np.sqrt(metrics.mean_squared_error(test[target], predictions))
    fig, ax = plt.subplots(2, 1, figsize = (14, 14))
    ax[0].plot(train['date'], train[target], color = 'red', label = 'real')
    ax[0].plot(train['date'], oof, color = 'blue', label = 'prediction')
    ax[0].set_title('out of fold prediction vs real target')
    ax[1].plot(test['date'], test[target], color = 'red', label = 'real')
    ax[1].plot(test['date'], predictions, color = 'blue', label = 'prediction')
    ax[1].set_title('test prediction vs real target')
    plt.show()
    print('The standard deviation for no2 emissions for each month is:')
    print(train.groupby('month')[target].std().reset_index())
    print(f'Our out of folds rmse is {oof_rmse}')
    print(f'Our test rmse is {test_error}')
    return oof, predictions, all_coef

def plot_coef(coef):
    plt.figure(figsize = (12, 8))
    # absolute for better visuals
    sns.barplot(abs(coef['coef_']), coef['features'], orient = 'h')
    plt.title('Feature coefficients')
    plt.show()

# train with lags and time features
train1, test1, features1 = preprocess_data(no2_weather, use_lags = True, use_time = True)
oof1, predictions1, all_coef1 = train_linear_regression(train1, test1, features1, n_folds = 12)
plot_coef(all_coef1)


# * There are some months that are hardest to predict.
# * Lags are the most important features

# In[ ]:


train2, test2, features2 = preprocess_data(no2_weather, use_lags = False, use_time = False)
oof2, predictions2, all_coef2 = train_linear_regression(train2, test2, features2, n_folds = 12)
plot_coef(all_coef2)


# * If we don't consider lags and time features our model perform badly.
# * Weather data is no that strong to predict NO2 emissions.

# # Still on work!!
# 
# * Have a lot of question about the data, need to read more about it so we can understand better the problem.
# 
# Next i will see how we can add electriciy generation of the power plants to our lineal regression
