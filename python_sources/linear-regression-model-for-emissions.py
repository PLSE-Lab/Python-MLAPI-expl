#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os
import folium
import rasterio as rio
import tifffile as tiff
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import mean_squared_error

print(os.listdir('/kaggle/input/ds4g-environmental-insights-explorer/eie_data'))
# Any results you write to the current directory are saved as output.
eie_data_path = '/kaggle/input/ds4g-environmental-insights-explorer/eie_data'


# # Global Power Plant Database
# 
# The island of Puerto Rico has multiple power plants for its energy generation. We will begin by seeing where these power plants are.

# In[ ]:


gpp_df = pd.read_csv(eie_data_path+'/gppd/gppd_120_pr.csv')
gpp_df.head()


# In[ ]:


#code source: https://www.kaggle.com/paultimothymooney/overview-of-the-eie-analytics-challenge
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

gpp_df = split_column_into_new_columns(gpp_df,'.geo','latitude',50,66)
gpp_df = split_column_into_new_columns(gpp_df,'.geo','longitude',31,48)
gpp_df['latitude'] = gpp_df['latitude'].astype(float)
a = np.array(gpp_df['latitude'].values.tolist()) # 18 insted of 8
gpp_df['latitude'] = np.where(a < 10, a + 10, a).tolist()
lat = 18.200178; lon = -66.664513
plot_points_on_map(gpp_df, 0, 425, 'latitude', lat, 'longitude', lon, 9)


# In[ ]:


print('There are ', gpp_df.shape[0],' power plants')


# In[ ]:


gpp_df.head()
#For the purposes of the model, we will use estimated power generation in gwh as a feature
total_power_generation_2017 = gpp_df['estimated_generation_gwh'].sum()
print('The total estimated power generation for all power plants in puerto for 2017 rico is: ', total_power_generation_2017, ' gwh')


# ## Sentinel-5 Precursor Data
# 
# The Sentinel-5 Precuror satellite has sensors to gauge NO2 concentrations. Below we will visualize the NO2 concentrations and display them over the island of Puerto Rico. 

# In[ ]:


no2_emissions_image = eie_data_path+'/s5p_no2/s5p_no2_20180701T161259_20180707T175356.tif'
latitude=18.1429005246921; longitude=-65.4440010699994
overlay_image_on_puerto_rico(no2_emissions_image,band_layer=1,lat=latitude,lon=longitude,zoom=8)


# In the image above, we are overlaying the first band of an image taken on July 7th, 2017 over the island of Puero Rico. The overlayed image (in red) displays the total vertical column of NO 2 in mol/m^2. The shade of red varies based on the quantity. Below there is a similar overlay for band 2 (measurements of the vertical column of NO 2 for the troposphere).

# In[ ]:


overlay_image_on_puerto_rico(no2_emissions_image,band_layer=2,lat=latitude,lon=longitude,zoom=8)


# ### First Simple Model
# 
# An Emissions Factor is a value that relates the quanity of a pollutant in the atmosphere with an activity associated with the release of that pollutant. 
# 
# We can calculate emissions using the following equation: 
# 
# E = A * EF * (1 - ER / 100 )
# 
# Here A refers to an activity, EF is the emissions factor, and ER is the overall emissions reduction efficiency (percentage value). 
# 
# If we solve for the Emissions Factor:
# 
# EF = (E * 100) / (A * (1-ER))
# 
# By solving for the EF, we can try to calculate the factor for the power plant data
# 
# From the Sentinel Satellite, NO2 emissions are measured in band1, band2, band3 and band4. The total emissions (E) for a given day (each tif file for emissions is a snapshot for a single day), can be calculated by summing these four bands).
# 
# The Activity for which we're interested in is the energy generation from power plants. One way to calculate the emissions factor for the energy generation would be to use the activity from all the power plants as A, and assume that the emissions reduction efficiency is 0. Therefore:
# 
# EF = (E) / (A)

# In[ ]:


activity = total_power_generation_2017
#Using the emissions snapshot from above
no2_emissions_sum = tiff.imread(no2_emissions_image)[:,:,0:4].sum()
emissions_factor = (no2_emissions_sum ) / (total_power_generation_2017)
print("Emissions Factor for power plant energy generation activity : ", emissions_factor)


# We have calculated the emissions factor to be 3.886 * 10^-5. This is a very rough estimate of the emissions factor. This model makes many assumptions:
# 
# - It does not take any spatial information into account, i.e. the original NO2 emissions data has measurements over the entire island of Puerto Rico. The emissions (as seen in the overlayed map), vary based on the location. Instead emissions over the entire island are summed. Similar to the NO2 emissions, the power plants total power generation does not take spatial information into account (the activity is not uniform over the entire island). Due to these assumptions, it is assumed that only energy generation activity from power plants contributes to the generation of NO2 emissions.
# 
# - The emissions are only from a single date. Emissions vary from day to day, and this information is lost. Similarly activity data is also not of a single day instead it is just a summation of all the activity from the power plants. 
# 

# ### Feature Engineering
# 
# To assess what factors contribute to emissions, a model can be built to predict the NO2 emissions. The features of the model can be activities that sources of NO2, and/or causes of the spread of NO2. 
# 
# We will create features from the NO2 emissions, weather, land assimilation data, and power plant data. 

# In[ ]:


#Get the date information from each files name
def get_dates(file_path, data_source):
    if data_source == 's5p':
        fname_only = file_path.split('/')
        dates_only = (fname_only[-1].split('_')[2], fname_only[-1].split('_')[3])
        start_date = dates_only[0][:8]
        end_date = dates_only[1][:8]
        return start_date, end_date
    elif data_source == 'gfs':
        file_name = file_path.split('/')[-1]
        date_w_extension = file_name.split('_')[-1]
        date = date_w_extension.split('.')[0][:8]
        return date
    elif data_source == 'gldas':
        file_name = file_path.split('/')[-1]
        date = file_name.split('_')[1]
        return date
    
#Read the data from the files and assemble a dataframe 
def read_data(data_path,data_source):
    data = []
    for file in os.listdir(data_path):
        file_path = data_path+file
        img = tiff.imread(file_path)
        img_to_add = {}
        if data_source == 's5p':
            start_date, end_date = get_dates(file_path, 's5p')
            img_to_add['start_date'] = start_date
            img_to_add['end_date'] = end_date
            img_to_add['no2_emissions_mean'] = np.nanmean(img[:,:,0:4])
        elif data_source == 'gfs':
            img_to_add['date'] = get_dates(file_path, 'gfs')
            img_to_add['temp'] = img[:,:,0]
            img_to_add['specific_humidity'] = img[:,:,1]
            img_to_add['relative_humidity'] = img[:, :, 2]
            img_to_add['u_component_wind'] = img[:, :, 3]
            img_to_add['v_component_wind'] = img[:, :, 4]
            img_to_add['total_precipation'] = img[:, :, 5]
        elif data_source == 'gldas':
            img_to_add['date'] = get_dates(file_path, 'gldas')
            for band in range(1, 13):
                img_to_add['band'+str(band)] = rio.open(file_path).read(band)    
                
        data.append(img_to_add)
        data_df = pd.DataFrame(data)
        
        if data_source == 's5p':
            data_df['start_date'] = pd.to_datetime(data_df['start_date'])
            data_df['end_date'] = pd.to_datetime(data_df['end_date'])
            data_df.sort_values('start_date', inplace = True)
            data_df.reset_index(drop = True, inplace = True)
        else: 
            data_df['date'] = pd.to_datetime(data_df['date'])
            data_df.sort_values('date', inplace=True)
       
    return data_df


# ##### Getting Labels for Regression Model from Sentinel Satellite

# In[ ]:


no2_path = eie_data_path+'/s5p_no2/'
print("Number of sentinel satellite pictures: ", len(os.listdir(no2_path)))


# Below, we'll get the the data from .tif files that contain NO2 measurements of the Sentinel Satellite. We'll create a pandas dataframe that will have the mean no2 emissions. 

# In[ ]:


emissions_mean = read_data(no2_path, 's5p')
emissions_mean.head()


# Below is a plot of the emissions mean (Y axis), and the start date on the x axis. We see that emissions peak just before May, followed by additonal peaks during June, July, and August.

# In[ ]:


emissions_mean.plot.line(x='start_date', y='no2_emissions_mean')


# ##### Features from Atmosphere Data (Global Forecast System 384-Hour Predicted Atmosphere Data) 

# In[ ]:


gfs_path = eie_data_path + '/gfs/'
print("We have " , len(gfs_path), " pictures of the global forecast system")
weather_df = read_data(gfs_path, 'gfs')
weather_df.head()


# In[ ]:


def get_weather_feature_stats(df):
    for c in [col for col in df.columns if col not in ['date']]:
        df[c] = df[c].apply(np.mean)
    return df
weather_mean_stats = get_weather_feature_stats(weather_df.copy())
weather_mean_stats.head()


# In the DataFrame above you can see that each date as a atmosphere feature associated with it, i.e. temperature, humidity, etc. 

# ##### Beginning to assemble the dataset for our model
# 
# Below, we'll start building our dataset for our model. The dataset will have m examples in total, where m is the number of dates that correspond to the NO2 emissions measurements. 

# In[ ]:


#As there are multiple weather measurements per day, group them by the date and take the mean
weather_features = weather_mean_stats.groupby('date').mean()
#Also group the emissions  by date 
emissions_mean = emissions_mean.groupby('start_date').mean()
#concatenate the mean emissions dataframe and the weather features
training_data = pd.concat([weather_features, emissions_mean], axis=1 , join='outer')
#Drop any NaN values
training_data.dropna(how='any',inplace=True)
training_data.head()


# ##### Features from Global Land Data Assimilation System

# In[ ]:


gldas_path = eie_data_path + '/gldas/'
print("We have ", len(os.listdir(gldas_path)), " pictures of land data")
gldas_df = read_data(gldas_path, 'gldas')
gldas_df.head()


# In[ ]:


def get_gldas_mean_stats(df):
    cols = [col for col in df.columns if col not in ['date']]
    print(cols)
    for c in cols:
        df[c] = df[c].apply(np.mean)
    return df
gldas_mean_stats = get_gldas_mean_stats(gldas_df.copy())
gldas_mean_stats['date'] = gldas_mean_stats['date'].apply(pd.Timestamp.date)
gldas_mean_stats = gldas_mean_stats.fillna(0)
gldas_mean_stats = gldas_mean_stats.groupby('date').mean()

training_data = pd.concat([training_data, gldas_mean_stats], axis=1 , join='outer')
training_data.dropna(how='any', inplace=True)
#Add the total power generation of all the power plants
training_data['total_power_generation'] = total_power_generation_2017
training_data.head()


# ### Linear Regression
# 
# For a regression model, our goal is to predict the NO2 emissions by using features engineered from power plant, atmosphere, and land use data. There are many factors that contribute to NO2 emissions. While there are sources of pollution such as electricty/power generation, traffic, and other industrial activities, conditions of land and the atmosphere can affect how pollutants remain in, and spread from the environment that they originate from.
# 
# For each given day we have the average of the NO2 emissions of that day, and the weather and land information as well. Features from weather data include the mean temperature, specific humidity, relative humidity, the U component of the wind, the V component of the wind, and the total precipation at surface. Similarly, the features from the land assimilation data include albedo, average surface skin temperature, plant canopy surface water, canpoy water evaporation, evaporation from bare soil, evapotranspiration, downward long-wave radiation flux, net long-wave radiation flux, potential evaporation rate, pressure, specific humidity, and heat flux (first 12 bands).There is not time series data available- so the total power generation of the entire island of Puerto Rico will be assumed to be constant (for the time being).
# 
# Measurements from s5p (NO2 emissions), weather, and land originally provide data for the entire island. For the purpose of the regression, we average the measurement accross the entire island. This does result in loss of spatial information, and is a much more coarse.

# In[ ]:


ds = training_data.copy()
train_df = ds.iloc[:324]
test_df = ds[324:]
X = train_df[[col for col in ds.columns if col not in ['no2_emissions_mean']]]
Y = train_df['no2_emissions_mean']
model = LinearRegression().fit(X,Y)
test_X = test_df[[col for col in ds.columns if col not in ['no2_emissions_mean']]]
test_Y = test_df['no2_emissions_mean']
preds = model.predict(test_X)
preds = pd.Series(preds)


# In[ ]:


plt.plot(preds.index, preds.values, color='red')


# In[ ]:


plt.plot(test_Y.index, test_Y.values)


# In[ ]:


rms = np.sqrt(mean_squared_error(test_Y, preds))


# In[ ]:


print(rms)


# Based on the above we can see that there is room for improvement for the regression model. Future model improvements can include using more features from power plant/activity, weather, and land use data in addition to incorporating spatial information of emissions.

# In[ ]:




